"""Full-scale, checkpointed candidate generation with a frozen blocking configuration.

Queries are processed in fixed-size chunks; each chunk is written to its own Parquet
file, and an interrupted run resumes by skipping chunks that already exist. Output
per (query, target source): the ranked candidate entity_ids and their scores, so later
stages can re-truncate (fixed K or score-relative) without re-running retrieval.

    python -m er.candidates --split train --fold 0 --config fuse_nab_pab_k100
    python -m er.candidates --split test --n-queries 20000 --config fuse_nab_pab_k100
    python -m er.candidates --evaluate --split train --fold 0 --config fuse_nab_pab_k100
"""
import argparse
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .blocking import TfidfParams, tfidf_candidates_by_country
from .data import parse_id_list
from .index import load_entity_ids
from .metrics import blocking_metrics
from .normalize import VERSION as NORM_VERSION
from .resources import Tracker, available_gb
from .splits import load_folds
from .store import store_path

TARGETS = ("source2", "source3")

# Frozen blocking configurations. "variants" are retrieved independently; with
# "fuse", each candidate keeps its best score across variants (late fusion).
CONFIGS = {
    # Phase 2 frozen choice (P2-E2 / P2-E3): literal + phonetic-only retrieval, late fusion,
    # cap_df 20k, retrieval depth 1000, top 100 per target source kept with scores.
    "fuse_nab_pab_r1000_k100": {"variants": [TfidfParams(namespaces="nab", top_r=1000, top_k=100),
                                             TfidfParams(namespaces="pab", top_r=1000, top_k=100)],
                                "fuse": True, "top_k": 100},
    "fuse_nab_pab_k100": {"variants": [TfidfParams(namespaces="nab", top_k=100),
                                       TfidfParams(namespaces="pab", top_k=100)],
                          "fuse": True, "top_k": 100},
    "nabp_p025_k100": {"variants": [TfidfParams(namespaces="nabp", top_k=100, ns_weights={"p": 0.25})],
                       "fuse": False, "top_k": 100},
    "nab_k100": {"variants": [TfidfParams(namespaces="nab", top_k=100)], "fuse": False, "top_k": 100},
}

SCHEMA = pa.schema([("source1_entity_id", pa.string()), ("country", pa.string()), ("target_source", pa.string()),
                    ("candidate_entity_ids", pa.list_(pa.string())), ("scores", pa.list_(pa.float32()))])


def run_dir(split: str, cfg: str, scope: str):
    return config.WORK_DIR / "candidates" / f"{split}_{scope}_{cfg}"


def load_queries(split: str, fold: int = None, n: int = None, seed: int = 13) -> pd.DataFrame:
    cols = ["entity_id", "country", "name_norm", "addr_norm"]
    s1 = pq.read_table(store_path(split, "source1"), columns=cols)
    if split == "train" and fold is not None:
        folds = load_folds(["entity_id", "fold"])
        idx = np.flatnonzero(folds["fold"].to_numpy() == fold)
        if not (s1.column("entity_id").take(pa.array(idx[:1000])).to_pylist()
                == folds["entity_id"].iloc[idx[:1000]].tolist()):
            raise ValueError("fold table and Source-1 store are out of row order")
    else:
        idx = np.arange(s1.num_rows)
    if n and n < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return s1.take(pa.array(idx)).to_pandas()


def _fuse_rows(lists_rows: list, lists_scores: list, k: int) -> tuple:
    rows = np.concatenate(lists_rows)
    scores = np.concatenate(lists_scores)
    if not len(rows):
        return rows.astype(np.int32), scores.astype(np.float32)
    order = np.lexsort((-scores, rows))          # group by row, best score first
    rows, scores = rows[order], scores[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = rows[1:] != rows[:-1]
    rows, scores = rows[first], scores[first]
    top = np.argsort(-scores, kind="stable")[:k]
    return rows[top].astype(np.int32), scores[top].astype(np.float32)


def _chunk_table(q: pd.DataFrame, source: str, ent_ids: pa.Array, rows_list, scores_list) -> pa.Table:
    lengths = np.array([len(r) for r in rows_list], dtype=np.int64)
    offsets = pa.array(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))
    flat_rows = np.concatenate(rows_list) if lengths.sum() else np.zeros(0, dtype=np.int32)
    flat_scores = np.concatenate(scores_list) if lengths.sum() else np.zeros(0, dtype=np.float32)
    ids = pa.ListArray.from_arrays(offsets, ent_ids.take(pa.array(flat_rows)))
    sc = pa.ListArray.from_arrays(offsets, pa.array(flat_scores.astype(np.float32)))
    n = len(q)
    return pa.Table.from_arrays([pa.array(q["entity_id"]), pa.array(q["country"]),
                                 pa.array([source] * n), ids, sc], schema=SCHEMA)


def generate(split: str, cfg_name: str, fold: int = None, n: int = None, seed: int = 13, chunk: int = 50_000):
    cfg = CONFIGS[cfg_name]
    scope = f"fold{fold}" if fold is not None else (f"sample{n}_seed{seed}" if n else "all")
    out = run_dir(split, cfg_name, scope)
    out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(split, fold, n, seed)
    n_chunks = (len(queries) + chunk - 1) // chunk
    meta = {"split": split, "config": cfg_name, "scope": scope, "n_queries": len(queries), "chunk": chunk,
            "n_chunks": n_chunks, "norm_version": NORM_VERSION,
            "variants": [v.__dict__ for v in cfg["variants"]], "fuse": cfg["fuse"], "top_k": cfg["top_k"]}
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"meta": meta, "chunks": {}}
    if progress["meta"] != meta:
        raise ValueError(f"{out} holds a run with different settings; use a new config name or delete it")
    ent_ids = {s: load_entity_ids(split, s) for s in TARGETS}
    print(f"[{out.name}] {len(queries)} queries in {n_chunks} chunks (avail {available_gb():.1f} GB)", flush=True)

    for c in range(n_chunks):
        path = out / f"chunk_{c:04d}.parquet"
        if path.exists():
            continue
        q = queries.iloc[c * chunk:(c + 1) * chunk].reset_index(drop=True)
        tables = []
        with Tracker() as t:
            for s in TARGETS:
                outs = tfidf_candidates_by_country(split, s, q, cfg["variants"])
                if cfg["fuse"]:
                    fused = [_fuse_rows([o[0][i] for o in outs], [o[1][i] for o in outs], cfg["top_k"])
                             for i in range(len(q))]
                    rows_list, scores_list = [f[0] for f in fused], [f[1] for f in fused]
                else:
                    rows_list, scores_list = outs[0]
                tables.append(_chunk_table(q, s, ent_ids[s], rows_list, scores_list))
        tmp = path.with_suffix(".tmp")
        pq.write_table(pa.concat_tables(tables), tmp, compression="zstd")
        tmp.replace(path)  # a chunk file exists only if it was written completely
        progress["chunks"][str(c)] = {"queries": len(q), "seconds": round(t.seconds, 1),
                                      "peak_rss_gb": round(t.peak_rss_gb, 2),
                                      "finished": dt.datetime.now().isoformat(timespec="seconds")}
        progress_path.write_text(json.dumps(progress, indent=2))
        done = len(progress["chunks"])
        print(f"  chunk {c + 1}/{n_chunks}: {t.seconds:.0f}s, peak {t.peak_rss_gb:.2f} GB ({done}/{n_chunks} done)",
              flush=True)
    return out


def load_ranked(out_dir) -> pd.DataFrame:
    return pq.read_table(sorted(out_dir.glob("chunk_*.parquet"))).to_pandas()


def evaluate(split: str, cfg_name: str, scope: str, ks=(10, 20, 30, 50, 100)) -> dict:
    """Blocking metrics of a finished run (train split) at several per-source candidate budgets.
    Truth comes from the fold table for exactly the queries present in the run."""
    out = run_dir(split, cfg_name, scope)
    tbl = pq.read_table(sorted(out.glob("chunk_*.parquet")), columns=["source1_entity_id", "candidate_entity_ids"])
    s1 = tbl.column("source1_entity_id").to_pylist()
    cand = tbl.column("candidate_entity_ids").to_pylist()
    del tbl
    ids = list(dict.fromkeys(s1))
    folds = load_folds(["entity_id", "matched_entity_ids"])
    folds = folds[folds["entity_id"].isin(set(ids))]
    truth = {e: frozenset(parse_id_list(m)) for e, m in zip(folds["entity_id"], folds["matched_entity_ids"])}
    if len(truth) != len(ids):
        raise ValueError("some queries have no ground-truth row")
    pool = sum(pq.ParquetFile(store_path(split, s)).metadata.num_rows for s in TARGETS)
    report = {"config": cfg_name, "scope": scope, "n_queries": len(ids), "per_k": {}}
    for k in ks:
        sets = {}
        for e, c in zip(s1, cand):
            sets.setdefault(e, set()).update(c[:k])
        m = blocking_metrics(sets, truth, ids, pool)
        report["per_k"][k] = {key: (round(v, 6) if isinstance(v, float) else v) for key, v in m.items()}
        print(f"  K={k:3d}/source: recall={m['pair_recall']:.4f} full_cov={m['entity_full_coverage']:.4f} "
              f"oracleF={m['oracle_f05']:.4f} cands={m['cands_mean']:.1f}", flush=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def describe(split: str, cfg_name: str, scope: str) -> dict:
    """Label-free candidate statistics per (country, target source): usable on the test split,
    where an unseen country (France) cannot be scored for recall. Compare against US/India."""
    out = run_dir(split, cfg_name, scope)
    df = pq.read_table(sorted(out.glob("chunk_*.parquet")), columns=["country", "target_source", "scores"]).to_pandas()
    df["n_cands"] = df["scores"].map(len)
    df["top1"] = df["scores"].map(lambda s: float(s[0]) if len(s) else np.nan)
    df["top1_minus_top2"] = df["scores"].map(lambda s: float(s[0] - s[1]) if len(s) > 1 else np.nan)
    rows = {}
    for (c, s), g in df.groupby(["country", "target_source"]):
        rows[f"{c}|{s}"] = {
            "queries": int(len(g)),
            "zero_cand_share": round(float((g["n_cands"] == 0).mean()), 5),
            "mean_cands": round(float(g["n_cands"].mean()), 2),
            "top1_p10_p50_p90": [round(float(x), 4) for x in g["top1"].quantile([0.1, 0.5, 0.9])],
            "top1_gap_p50": round(float(g["top1_minus_top2"].median()), 4),
        }
        print(f"  {c:8s} {s}: n={len(g):7d} zero={rows[f'{c}|{s}']['zero_cand_share']:.4f} "
              f"cands={rows[f'{c}|{s}']['mean_cands']:.1f} top1 p10/50/90={rows[f'{c}|{s}']['top1_p10_p50_p90']} "
              f"gap_p50={rows[f'{c}|{s}']['top1_gap_p50']}", flush=True)
    report = {"split": split, "config": cfg_name, "scope": scope, "by_country_source": rows}
    (out / "describe.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--config", default="fuse_nab_pab_r1000_k100", choices=list(CONFIGS))
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--describe", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    scope = f"fold{args.fold}" if args.fold is not None else (
        f"sample{args.n_queries}_seed{args.seed}" if args.n_queries else "all")
    if args.evaluate:
        evaluate(args.split, args.config, scope)
    elif args.describe:
        describe(args.split, args.config, scope)
    else:
        generate(args.split, args.config, args.fold, args.n_queries, args.seed, args.chunk)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
