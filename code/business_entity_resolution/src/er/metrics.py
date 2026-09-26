"""Competition metric (entity-level macro F_0.5) and blocking-quality metrics.

Metric definition (amazon_ml.pdf): F_0.5 is computed per Source-1 entity and
macro-averaged over all Source-1 entities. A singleton (no true matches) scores
1.0 for an empty prediction and 0.0 for any prediction.
"""
from typing import Iterable, Mapping

import numpy as np

BETA = 0.5


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return 0.0 if denom == 0 else (1 + b2) * precision * recall / denom


def entity_scores(pred: Iterable, truth: Iterable) -> tuple:
    """Return (precision, recall, f0.5) for one Source-1 entity.

    Precision/recall are NaN where undefined (empty prediction / empty truth);
    the F score is always defined.
    """
    pred, truth = set(pred), set(truth)
    if not truth:
        return (np.nan, np.nan, 1.0) if not pred else (0.0, np.nan, 0.0)
    if not pred:
        return np.nan, 0.0, 0.0
    tp = len(pred & truth)
    p, r = tp / len(pred), tp / len(truth)
    return p, r, f_beta(p, r)


def macro_f05(pred: Mapping, truth: Mapping, ids: Iterable = None) -> dict:
    """Macro F_0.5 over `ids` (default: every key of `truth`). Missing predictions count as empty."""
    ids = list(truth.keys()) if ids is None else list(ids)
    f = np.empty(len(ids))
    p = np.empty(len(ids))
    r = np.empty(len(ids))
    singleton = np.zeros(len(ids), dtype=bool)
    for i, e in enumerate(ids):
        t = truth.get(e, ())
        p[i], r[i], f[i] = entity_scores(pred.get(e, ()), t)
        singleton[i] = len(t) == 0
    n_pred_nonempty = sum(1 for e in ids if pred.get(e))
    return {
        "n_entities": len(ids),
        "f05": float(f.mean()) if len(ids) else float("nan"),
        "f05_singletons": float(f[singleton].mean()) if singleton.any() else float("nan"),
        "f05_non_singletons": float(f[~singleton].mean()) if (~singleton).any() else float("nan"),
        "mean_precision": float(np.nanmean(p)) if np.isfinite(p).any() else float("nan"),
        "mean_recall": float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"),
        "singleton_share": float(singleton.mean()) if len(ids) else float("nan"),
        "pred_nonempty_share": n_pred_nonempty / len(ids) if len(ids) else float("nan"),
    }


def blocking_metrics(candidates: Mapping, truth: Mapping, ids: Iterable, pool_size: int) -> dict:
    """Quality of a candidate set, independent of any matcher.

    - pair_recall: share of all true (S1, target) pairs present among the candidates
      (the recall ceiling for any downstream matcher).
    - entity_full_coverage: share of non-singleton entities whose *every* true match is a candidate.
    - oracle_f05: macro F_0.5 of a perfect matcher restricted to these candidates
      (predicts candidates ∩ truth; predicts empty for singletons).
    - reduction_ratio: 1 - candidate pairs / (n_queries * pool_size).
    """
    ids = list(ids)
    n_true = n_hit = full = nonsingle = 0
    counts = np.empty(len(ids), dtype=np.int64)
    oracle = np.empty(len(ids))
    for i, e in enumerate(ids):
        c = candidates.get(e, ())
        c = c if isinstance(c, (set, frozenset)) else set(c)
        t = truth.get(e, ())
        counts[i] = len(c)
        if t:
            nonsingle += 1
            hit = len(c & set(t)) if c else 0
            n_true += len(t)
            n_hit += hit
            full += hit == len(t)
            oracle[i] = f_beta(1.0, hit / len(t)) if hit else 0.0
        else:
            oracle[i] = 1.0
    total = int(counts.sum())
    n = len(ids)
    return {
        "n_queries": n,
        "true_pairs": n_true,
        "pair_recall": n_hit / n_true if n_true else float("nan"),
        "entity_full_coverage": full / nonsingle if nonsingle else float("nan"),
        "oracle_f05": float(oracle.mean()) if n else float("nan"),
        "cands_mean": float(counts.mean()) if n else 0.0,
        "cands_p50": float(np.percentile(counts, 50)) if n else 0.0,
        "cands_p95": float(np.percentile(counts, 95)) if n else 0.0,
        "cands_max": int(counts.max()) if n else 0,
        "cands_total": total,
        "zero_cand_share": float((counts == 0).mean()) if n else float("nan"),
        "reduction_ratio": 1.0 - total / (n * pool_size) if n and pool_size else float("nan"),
    }
