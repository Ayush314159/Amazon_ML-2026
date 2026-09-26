"""Blocking experiments on a validation sample of Source-1 queries.

Every strategy is evaluated against the *full* training Source-2/3 pools (the
indexes cover all 10.3M records) so that recall and candidate volume are
realistic; only the query set is sampled.

    python -m er.blocking_experiments --n-queries 20000 --suite baseline
"""
import argparse
import csv
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .blocking import TfidfParams, key_candidates, rows_to_ids, tfidf_candidates_by_country
from .data import parse_id_list
from .index import KEY_FUNCS, load_entity_ids, load_manifest
from .metrics import blocking_metrics
from .normalize import VERSION as NORM_VERSION
from .resources import Tracker, available_gb
from .splits import VAL_FOLD, sample_queries
from .store import store_path

TARGETS = ("source2", "source3")
INDIC = "[ऀ-෿]"

RESULT_COLUMNS = [
    "run_id", "date", "strategy", "params", "n_queries", "pair_recall", "entity_full_coverage",
    "oracle_f05", "cands_mean", "cands_p50", "cands_p95", "cands_max", "zero_cand_share",
    "reduction_ratio", "seconds", "sec_per_1k_queries", "peak_rss_gb",
]

SUITES = {
    # every single-method family, plus unions
    "baseline": {
        "keys": list(KEY_FUNCS),
        "tfidf": {
            "tfidf_n": TfidfParams(namespaces="n", top_k=100),
            "tfidf_ab": TfidfParams(namespaces="ab", top_k=100),
            "tfidf_nab": TfidfParams(namespaces="nab", top_k=100),
            "tfidf_nab_norescore": TfidfParams(namespaces="nab", top_k=100, rescore=False),
        },
        "k_sweep": ("tfidf_nab", [5, 10, 20, 30, 50, 100]),
        "unions": {
            "keys_all": ["name_exact", "name_core", "addr_exact", "addr_set"],
            "tfidf_n+tfidf_ab": ["tfidf_n", "tfidf_ab"],
            "tfidf_nab+keys_all": ["tfidf_nab", "name_exact", "name_core", "addr_exact", "addr_set"],
        },
    },
    # after Indic transliteration (normalize v2) + phonetic-key namespace
    "v2": {
        "keys": list(KEY_FUNCS),
        "tfidf": {
            "tfidf_nab": TfidfParams(namespaces="nab", top_k=100),
            "tfidf_nabp": TfidfParams(namespaces="nabp", top_k=100),
            "tfidf_nabp_p0.5": TfidfParams(namespaces="nabp", top_k=100, ns_weights={"p": 0.5}),
            "tfidf_nabp_addr0.7": TfidfParams(namespaces="nabp", top_k=100, ns_weights={"a": 0.7, "b": 0.7}),
        },
        "k_sweep": ("tfidf_nabp", [5, 10, 20, 30, 50, 100]),
        "unions": {
            "keys_all": ["name_exact", "name_core", "addr_exact", "addr_set"],
            "tfidf_nabp+keys_all": ["tfidf_nabp", "name_exact", "name_core", "addr_exact", "addr_set"],
        },
    },
    # posting-length cap and retrieval depth
    "cap": {
        "keys": [],
        "tfidf": {
            "nab_cap2k": TfidfParams(cap_df=2_000, top_k=50),
            "nab_cap20k": TfidfParams(cap_df=20_000, top_k=50),
            "nab_cap100k": TfidfParams(cap_df=100_000, top_k=50, work_budget=6e7),
            "nab_cap20k_r1000": TfidfParams(cap_df=20_000, top_r=1000, top_k=50),
        },
        "k_sweep": None,
        "unions": {},
    },
}


def _as_sets(per_source: dict, n: int, k: int = None) -> list:
    """Union the per-source candidate lists (optionally truncated to top-k per source)."""
    out = [set() for _ in range(n)]
    for lists in per_source.values():
        for i, ids in enumerate(lists):
            out[i].update(ids if k is None else ids[:k])
    return out


def _evaluate(name, params, cand_sets, queries, truth, pool_size, seconds, peak, run_id):
    ids = queries["entity_id"].tolist()
    m = blocking_metrics(dict(zip(ids, cand_sets)), truth, ids, pool_size)
    row = {"run_id": run_id, "date": dt.date.today().isoformat(), "strategy": name,
           "params": f"norm_v{NORM_VERSION}; {params}",
           **{k: m[k] for k in RESULT_COLUMNS if k in m},
           "seconds": round(seconds, 2) if seconds is not None else "",
           "sec_per_1k_queries": round(1000 * seconds / len(ids), 3) if seconds else "",
           "peak_rss_gb": round(peak, 2) if peak is not None else ""}
    for k in ("pair_recall", "entity_full_coverage", "oracle_f05", "reduction_ratio", "zero_cand_share"):
        row[k] = round(row[k], 6)
    print(f"  {name:28s} recall={row['pair_recall']:.4f} full_cov={row['entity_full_coverage']:.4f} "
          f"oracleF={row['oracle_f05']:.4f} cands={row['cands_mean']:.1f} (p95 {row['cands_p95']:.0f}) "
          f"t={row['seconds']}s peak={row['peak_rss_gb']}GB", flush=True)
    return row


def _segments(cand_sets, queries, truth, target_attr) -> dict:
    """Pair recall broken down by query country, target source, and target text properties."""
    rows = []
    for e, c, cs in zip(queries["entity_id"], queries["country"], cand_sets):
        for t in truth[e]:
            a = target_attr.get(t, {})
            rows.append((c, t[:2], a.get("indic_name", False), a.get("empty_addr", False),
                         a.get("country") == c, t in cs))
    df = pd.DataFrame(rows, columns=["q_country", "t_source", "t_indic_name", "t_empty_addr", "same_country", "hit"])
    out = {}
    for col in ["q_country", "t_source", "t_indic_name", "t_empty_addr", "same_country"]:
        g = df.groupby(col)["hit"].agg(["mean", "size"])
        out[col] = {str(k): {"recall": round(float(v["mean"]), 4), "pairs": int(v["size"])} for k, v in g.iterrows()}
    return out


def _target_attributes(truth: dict) -> dict:
    """Country / script / empty-address flags for every true target of the sample (read once, filtered)."""
    wanted = sorted({t for s in truth.values() for t in s})
    attr = {}
    for src in TARGETS:
        prefix = "S2-" if src == "source2" else "S3-"
        ids = [t for t in wanted if t.startswith(prefix)]
        tbl = pq.read_table(store_path("train", src), columns=["entity_id", "country", "business_name", "addr_norm"],
                            filters=[("entity_id", "in", ids)]).to_pandas()
        indic = tbl["business_name"].str.contains(INDIC, regex=True)
        for e, c, ind, a in zip(tbl["entity_id"], tbl["country"], indic, tbl["addr_norm"]):
            attr[e] = {"country": c, "indic_name": bool(ind), "empty_addr": a == ""}
    return attr


def run(n_queries: int, suite: str, seed: int, save_candidates: str = None):
    spec = SUITES[suite]
    run_id = f"blk_{suite}_{n_queries}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    print(f"[{run_id}] avail {available_gb():.1f} GB", flush=True)

    queries = sample_queries(n_queries, VAL_FOLD, seed)
    truth = {e: frozenset(parse_id_list(m)) for e, m in zip(queries["entity_id"], queries["matched_entity_ids"])}
    pool_size = sum(load_manifest("train", s)["rows"] for s in TARGETS)
    print(f"  queries={len(queries)} true_pairs={sum(len(v) for v in truth.values())} pool={pool_size}", flush=True)
    target_attr = _target_attributes(truth)

    ent_ids = {s: load_entity_ids("train", s) for s in TARGETS}
    results, cands, ranked = [], {}, {}

    for k in spec["keys"]:
        per_source = {}
        with Tracker() as t:
            for s in TARGETS:
                rows = key_candidates("train", s, k, queries["name_norm"], queries["addr_norm"], max_group=50)
                per_source[s] = rows_to_ids(ent_ids[s], rows)
        cands[k] = _as_sets(per_source, len(queries))
        results.append(_evaluate(k, "max_group=50", cands[k], queries, truth, pool_size, t.seconds, t.peak_rss_gb, run_id))

    if spec["tfidf"]:
        names = list(spec["tfidf"])
        params = [spec["tfidf"][n] for n in names]
        per_variant = {n: {} for n in names}
        timings = {}
        with Tracker() as t:
            for s in TARGETS:
                outs = tfidf_candidates_by_country("train", s, queries, params, timings)
                for n, (rows, _scores) in zip(names, outs):
                    per_variant[n][s] = rows_to_ids(ent_ids[s], rows)
        load_share = timings.get("load", 0.0) / len(names)
        print(f"  (tfidf: index load {timings.get('load', 0):.1f}s shared by {len(names)} variants; "
              f"peak {t.peak_rss_gb:.2f} GB for the whole block)", flush=True)
        for v, n in enumerate(names):
            ranked[n] = per_variant[n]
            cands[n] = _as_sets(per_variant[n], len(queries))
            secs = timings[f"variant_{v}"] + load_share
            results.append(_evaluate(n, json.dumps(spec["tfidf"][n].__dict__), cands[n], queries, truth,
                                     pool_size, secs, t.peak_rss_gb, run_id))

    if spec.get("k_sweep"):
        base, ks = spec["k_sweep"]
        for k in ks:
            sets = _as_sets(ranked[base], len(queries), k)
            cands[f"{base}@{k}"] = sets
            results.append(_evaluate(f"{base}@{k}/source", f"top_k per source={k}", sets, queries, truth,
                                     pool_size, None, None, run_id))

    for uname, members in spec["unions"].items():
        sets = [set().union(*(cands[m][i] for m in members)) for i in range(len(queries))]
        cands[uname] = sets
        results.append(_evaluate(uname, "+".join(members), sets, queries, truth, pool_size, None, None, run_id))

    segments = {}
    for name in [n for n in ("tfidf_nab", "tfidf_nab@20", "tfidf_nab+keys_all", "tfidf_nabp", "tfidf_nabp@20",
                             "tfidf_nabp+keys_all", "nab_cap20k") if n in cands]:
        segments[name] = _segments(cands[name], queries, truth, target_attr)

    _write(run_id, results, segments, suite, n_queries)
    if save_candidates and save_candidates in cands:
        out = config.WORK_DIR / "blocking_runs" / f"{run_id}_{save_candidates.replace('@', '_at')}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"source1_entity_id": queries["entity_id"],
                      "candidate_entity_ids": [",".join(sorted(c)) for c in cands[save_candidates]]}).to_parquet(out)
        print(f"  saved candidates -> {out}")
    return results, segments, cands, queries, truth


def _write(run_id, results, segments, suite, n_queries):
    config.EXPERIMENT_DIR.mkdir(exist_ok=True)
    path = config.EXPERIMENT_DIR / "blocking_results.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(results)
    detail_dir = config.EXPERIMENT_DIR / "blocking_runs"
    detail_dir.mkdir(exist_ok=True)
    (detail_dir / f"{run_id}.json").write_text(json.dumps({"results": results, "segments": segments}, indent=2))
    log = config.EXPERIMENT_DIR / "experiment_log.csv"
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in results:
            w.writerow([f"{run_id}:{r['strategy']}", r["date"], "blocking", r["strategy"],
                        f"train fold {VAL_FOLD} sample; pools = full train S2+S3", n_queries, r["params"],
                        r["pair_recall"], "", "", "",
                        f"oracle_f05={r['oracle_f05']}; cands_mean={r['cands_mean']:.1f}; "
                        f"RR={r['reduction_ratio']}; sec={r['seconds']}; peak_gb={r['peak_rss_gb']}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=20_000)
    ap.add_argument("--suite", default="baseline", choices=list(SUITES))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--save-candidates", default=None)
    args = ap.parse_args()
    t0 = time.time()
    run(args.n_queries, args.suite, args.seed, args.save_candidates)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
