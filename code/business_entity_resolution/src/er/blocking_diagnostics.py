"""Why does blocking miss a true pair? Classifies every missed (query, target) pair of a
validation sample and saves raw-text examples per category.

Categories (for the TF-IDF name+address retriever):
- no_shared_feature: query and target share no hashed feature at all (cosine 0)
- no_rare_feature: they share features, but none with df <= cap_df, so the target
  cannot be reached through the posting lists used for retrieval
- lost_in_retrieval: reachable, but outside the top_r by retrieval score
- ranked_below_k: its full cosine is below the query's k-th kept candidate

    python -m er.blocking_diagnostics --n-queries 20000 --top-k 100
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq

from . import config
from .blocking import TfidfParams, _query_matrix, tfidf_candidates
from .data import parse_id_list
from .index import PartitionIndex, index_dir, load_entity_ids, load_manifest
from .resources import Tracker
from .splits import VAL_FOLD, sample_queries
from .store import store_path

TARGETS = ("source2", "source3")
INDIC = "[ऀ-෿]"


def diagnose(n_queries: int, top_k: int, seed: int = 13, cap_df: int = 20_000, n_examples: int = 12):
    p = TfidfParams(cap_df=cap_df, top_k=top_k)
    q = sample_queries(n_queries, VAL_FOLD, seed)
    truth = {e: set(parse_id_list(m)) for e, m in zip(q["entity_id"], q["matched_entity_ids"])}
    misses = []
    with Tracker() as t:
        for src in TARGETS:
            prefix = "S2-" if src == "source2" else "S3-"
            ids = load_entity_ids("train", src)
            row_of = {}
            wanted = {x for s in truth.values() for x in s if x.startswith(prefix)}
            tbl = pq.read_table(store_path("train", src), columns=["entity_id"]).column(0).to_pylist()
            for r, e in enumerate(tbl):
                if e in wanted:
                    row_of[e] = r
            del tbl
            for c, meta in load_manifest("train", src)["partitions"].items():
                sel = np.flatnonzero(q["country"].to_numpy() == c)
                if not len(sel):
                    continue
                index = PartitionIndex.load(index_dir("train", src) / meta["dir"])
                nn, aa = q["name_norm"].to_numpy()[sel], q["addr_norm"].to_numpy()[sel]
                rows, scores = tfidf_candidates(index, nn, aa, p)
                Q = _query_matrix(index, nn, aa, p.namespaces, p.ns_weights)
                for j, qi in enumerate(sel):
                    e = q["entity_id"].iat[qi]
                    got = set(ids.take(rows[j]).to_pylist()) if len(rows[j]) else set()
                    for tgt in truth[e]:
                        if not tgt.startswith(prefix) or tgt in got:
                            continue
                        local = np.searchsorted(index.rows, row_of[tgt])
                        if local >= len(index.rows) or index.rows[local] != row_of[tgt]:
                            misses.append((e, tgt, "other_partition", 0.0, None))
                            continue
                        qv, tv = Q[j], index.X[local]
                        shared = np.intersect1d(qv.indices, tv.indices)
                        cos = float(qv.multiply(tv).sum())
                        kth = float(scores[j][-1]) if len(scores[j]) >= top_k else 0.0
                        if not len(shared):
                            cat = "no_shared_feature"
                        elif (index.df[shared] > cap_df).all():
                            cat = "no_rare_feature"
                        elif cos > kth:
                            cat = "lost_in_retrieval"
                        else:
                            cat = "ranked_below_k"
                        misses.append((e, tgt, cat, cos, kth))
                del index
    n_true = sum(len(s) for s in truth.values())
    cats = Counter(m[2] for m in misses)

    # raw text for examples
    by_cat = defaultdict(list)
    for m in misses:
        if len(by_cat[m[2]]) < n_examples:
            by_cat[m[2]].append(m)
    need_s1 = {m[0] for v in by_cat.values() for m in v}
    need_t = {m[1] for v in by_cat.values() for m in v}
    raw = {}
    for src in config.SOURCES:
        tbl = pq.read_table(store_path("train", src), columns=["entity_id", "business_name", "business_address"],
                            filters=[("entity_id", "in", sorted(need_s1 | need_t))]).to_pylist()
        raw.update({r["entity_id"]: (r["business_name"], r["business_address"]) for r in tbl})
    # indic / empty-address share per category
    t_attr = {}
    for src in TARGETS:
        tbl = pq.read_table(store_path("train", src), columns=["entity_id", "business_name", "addr_norm"],
                            filters=[("entity_id", "in", sorted({m[1] for m in misses if m[1].startswith('S2-' if src == 'source2' else 'S3-')}))]).to_pandas()
        indic = tbl["business_name"].str.contains(INDIC, regex=True)
        for e, i, a in zip(tbl["entity_id"], indic, tbl["addr_norm"]):
            t_attr[e] = (bool(i), a == "")
    profile = defaultdict(Counter)
    for m in misses:
        i, a = t_attr.get(m[1], (False, False))
        profile[m[2]]["indic_target_name"] += i
        profile[m[2]]["empty_target_addr"] += a
        profile[m[2]]["n"] += 1

    report = {
        "n_queries": len(q), "true_pairs": n_true, "missed_pairs": len(misses),
        "recall": 1 - len(misses) / n_true, "top_k_per_source": top_k, "cap_df": cap_df,
        "seconds": round(t.seconds, 1), "peak_rss_gb": round(t.peak_rss_gb, 2),
        "categories": {k: {"pairs": v, "share_of_misses": round(v / len(misses), 4),
                           "indic_target_name": profile[k]["indic_target_name"],
                           "empty_target_addr": profile[k]["empty_target_addr"]} for k, v in cats.most_common()},
        "examples": {k: [{"query": raw.get(m[0]), "target": raw.get(m[1]), "target_id": m[1],
                          "cosine": round(m[3], 4), "kth_score": None if m[4] is None else round(m[4], 4)}
                         for m in v] for k, v in by_cat.items()},
    }
    out = config.REPO_ROOT / "analysis" / f"blocking_misses_k{top_k}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=20_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--cap-df", type=int, default=20_000)
    args = ap.parse_args()
    diagnose(args.n_queries, args.top_k, cap_df=args.cap_df)


if __name__ == "__main__":
    main()
