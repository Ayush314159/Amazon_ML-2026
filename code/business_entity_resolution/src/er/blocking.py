"""Candidate generation over the prebuilt indexes (see index.py).

All strategies return, per query, an array of *store row positions* in one target
source (optionally with scores). Nothing here scans the source data per query.
"""
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import scipy.sparse as sp

from .index import (KEY_FUNCS, KeyIndex, PartitionIndex, hash_features, hash_strings,
                    index_dir, l2_normalize_rows, load_manifest, record_features)

EMPTY_ROWS = np.zeros(0, dtype=np.int32)
EMPTY_SCORES = np.zeros(0, dtype=np.float32)


# ---- exact keys -------------------------------------------------------------

def key_candidates(split: str, source: str, key_name: str, name_norm, addr_norm, max_group: int = 50) -> list:
    """Store rows sharing the query's exact key; keys shared by > max_group records are skipped."""
    ki = KeyIndex.load(index_dir(split, source) / f"key_{key_name}.npz")
    fn = KEY_FUNCS[key_name]
    keys = [fn(n, a) for n, a in zip(name_norm, addr_norm)]
    res = ki.lookup(hash_strings(keys), max_group)
    return [r if k else EMPTY_ROWS for r, k in zip(res, keys)]


# ---- TF-IDF retrieval -------------------------------------------------------

@dataclass
class TfidfParams:
    namespaces: str = "nabp"     # feature families: n=name, a=address, b=house-number bigram, p=phonetic name key
    cap_df: int = 20_000         # features with df above this are not used to *retrieve* (still used to rescore)
    top_r: int = 300             # retrieved per query before rescoring
    top_k: int = 100             # kept per query after full-cosine rescoring
    rescore: bool = True
    work_budget: float = 3e7     # posting entries touched per sparse-matmul batch (bounds memory)
    ns_weights: dict = field(default_factory=dict)  # query-side multiplier per namespace (default 1.0)


def _query_matrix(index: PartitionIndex, name_norm, addr_norm, namespaces: str, ns_weights: dict = None):
    """Query TF-IDF vectors. Namespace weights apply on the query side only, so they can be
    tuned without rebuilding the index."""
    all_feats = [record_features(n, a) for n, a in zip(name_norm, addr_norm)]
    Q = None
    for ns in namespaces:
        w = (ns_weights or {}).get(ns, 1.0)
        if w == 0:
            continue
        Qn = hash_features([[f for f in fs if f[0] == ns] for fs in all_feats])
        Qn.data = index.idf[Qn.indices] * np.float32(w)
        Qn.data[index.df[Qn.indices] == 0] = 0.0  # features absent from the partition carry no signal
        Q = Qn if Q is None else Q + Qn
    Q = Q.tocsr()
    Q.eliminate_zeros()
    return l2_normalize_rows(Q)


def _retrieval_matrix(index: PartitionIndex, Q: sp.csr_matrix, cap_df: int) -> sp.csr_matrix:
    """Keep only features with df <= cap_df; a query left with none keeps its single rarest feature."""
    df = index.df[Q.indices]
    keep = df <= cap_df
    rows = np.repeat(np.arange(Q.shape[0]), np.diff(Q.indptr))
    has_kept = np.bincount(rows[keep], minlength=Q.shape[0]) > 0
    for r in np.flatnonzero(~has_kept):
        a, b = Q.indptr[r], Q.indptr[r + 1]
        if b > a:
            keep[a + np.argmin(df[a:b])] = True
    R = Q.copy()
    R.data = np.where(keep, R.data, 0).astype(np.float32)
    R.eliminate_zeros()
    return R


def _top(indices: np.ndarray, scores: np.ndarray, k: int):
    if len(scores) > k:
        sel = np.argpartition(-scores, k - 1)[:k]
        indices, scores = indices[sel], scores[sel]
    order = np.argsort(-scores, kind="stable")
    return indices[order], scores[order]


def tfidf_candidates(index: PartitionIndex, name_norm, addr_norm, p: TfidfParams) -> tuple:
    """Returns (list of store-row arrays, list of score arrays), both ranked by score."""
    Q = _query_matrix(index, name_norm, addr_norm, p.namespaces, p.ns_weights)
    R = _retrieval_matrix(index, Q, p.cap_df)
    n = R.shape[0]
    row_of = np.repeat(np.arange(n), np.diff(R.indptr))
    work = np.bincount(row_of, weights=index.df[R.indices].astype(np.float64), minlength=n)
    P = index.P
    out_rows, out_scores = [], []
    start = 0
    while start < n:
        end, acc = start, 0.0
        while end < n and (end == start or acc + work[end] <= p.work_budget):
            acc += work[end]
            end += 1
        S = (R[start:end] @ P).tocsr()
        tops = []
        for i in range(end - start):
            a, b = S.indptr[i], S.indptr[i + 1]
            tops.append(_top(S.indices[a:b], S.data[a:b], p.top_r) if b > a else (EMPTY_ROWS, EMPTY_SCORES))
        if p.rescore:
            qi = np.repeat(np.arange(start, end), [len(t[0]) for t in tops])
            docs = np.concatenate([t[0] for t in tops]) if len(qi) else EMPTY_ROWS
            full = (np.asarray(index.X[docs].multiply(Q[qi]).sum(axis=1)).ravel().astype(np.float32)
                    if len(qi) else EMPTY_SCORES)
            pos = 0
            for t in tops:
                m = len(t[0])
                d, s = _top(t[0], full[pos:pos + m], p.top_k) if m else (EMPTY_ROWS, EMPTY_SCORES)
                pos += m
                out_rows.append(index.rows[d])
                out_scores.append(s)
        else:
            for d, s in tops:
                d, s = _top(d, s, p.top_k) if len(d) else (EMPTY_ROWS, EMPTY_SCORES)
                out_rows.append(index.rows[d])
                out_scores.append(s)
        del S
        start = end
    return out_rows, out_scores


def tfidf_candidates_by_country(split: str, source: str, queries, params: list, timings: dict = None) -> list:
    """Run one or more TfidfParams variants, loading each partition index only once.

    Each query is routed to the partition of its own `country` string. A country with
    no partition in this source (possible for any unseen label) searches all
    partitions and keeps the best-scoring candidates. Returns one (rows, scores)
    pair of per-query lists per variant. `timings` (optional) accumulates seconds:
    'load' for index loading, and 'variant_<i>' for each variant's retrieval.
    """
    import time
    timings = timings if timings is not None else {}
    parts = load_manifest(split, source)["partitions"]
    n = len(queries)
    outs = [([EMPTY_ROWS] * n, [EMPTY_SCORES] * n) for _ in params]
    countries = queries["country"].to_numpy()
    name_norm = queries["name_norm"].to_numpy()
    addr_norm = queries["addr_norm"].to_numpy()
    routed = np.isin(countries, list(parts))
    fallback = np.flatnonzero(~routed)
    pooled = [[[] for _ in fallback] for _ in params]

    for c, meta in parts.items():
        sel = np.flatnonzero(countries == c)
        if not len(sel) and not len(fallback):
            continue
        t0 = time.perf_counter()
        index = PartitionIndex.load(index_dir(split, source) / meta["dir"])
        _ = index.P  # build posting lists once, charged to load time
        timings["load"] = timings.get("load", 0.0) + time.perf_counter() - t0
        for v, p in enumerate(params):
            t0 = time.perf_counter()
            if len(sel):
                r, s = tfidf_candidates(index, name_norm[sel], addr_norm[sel], p)
                for j, i in enumerate(sel):
                    outs[v][0][i], outs[v][1][i] = r[j], s[j]
            if len(fallback):
                r, s = tfidf_candidates(index, name_norm[fallback], addr_norm[fallback], p)
                for j in range(len(fallback)):
                    pooled[v][j].append((r[j], s[j]))
            timings[f"variant_{v}"] = timings.get(f"variant_{v}", 0.0) + time.perf_counter() - t0
        del index

    for v, p in enumerate(params):
        for j, i in enumerate(fallback):
            rr = np.concatenate([x[0] for x in pooled[v][j]]) if pooled[v][j] else EMPTY_ROWS
            ss = np.concatenate([x[1] for x in pooled[v][j]]) if pooled[v][j] else EMPTY_SCORES
            outs[v][0][i], outs[v][1][i] = _top(rr, ss, p.top_k) if len(rr) else (EMPTY_ROWS, EMPTY_SCORES)
    return outs


def rows_to_ids(entity_ids: pa.Array, rows_list: list) -> list:
    """Convert per-query store-row arrays to per-query lists of entity_id strings."""
    lengths = [len(r) for r in rows_list]
    flat = np.concatenate(rows_list) if rows_list and sum(lengths) else EMPTY_ROWS
    ids = entity_ids.take(pa.array(flat)).to_pylist() if len(flat) else []
    out, pos = [], 0
    for m in lengths:
        out.append(ids[pos:pos + m])
        pos += m
    return out
