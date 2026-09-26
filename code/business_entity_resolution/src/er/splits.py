"""Validation folds over training Source-1 entities.

Folds are a deterministic function of the entity_id (crc32 mod 5), so every
stage (blocking, classifier training, threshold tuning) sees the same split
without needing a stored random seed. Fold 0 is the validation fold used for
all blocking/matching evaluation; folds 1-4 are available for classifier
training. The fold table also caches each entity's truth list and country so
later stages never re-parse the 122 MB ground-truth TSV.

    python -m er.splits
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .data import fold_of, load_ground_truth, parse_id_list
from .store import store_path

N_FOLDS = 5
VAL_FOLD = 0


def folds_path():
    return config.SPLIT_DIR / "train_s1_folds.parquet"


def build_folds() -> pd.DataFrame:
    gt = load_ground_truth()
    s1 = pq.read_table(store_path("train", "source1"), columns=["entity_id", "country"]).to_pandas()
    df = s1.merge(gt, left_on="entity_id", right_on="source1_entity_id", how="left", validate="1:1")
    if df["source1_entity_id"].isna().any():
        raise ValueError("Source-1 entities without a ground-truth row")
    df = df.drop(columns=["source1_entity_id"])
    df["fold"] = fold_of(df["entity_id"], N_FOLDS)
    df["n_matches"] = df["matched_entity_ids"].map(lambda s: len(parse_id_list(s))).astype(np.int16)
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), folds_path(), compression="zstd")
    return df


def load_folds(columns=None) -> pd.DataFrame:
    if not folds_path().exists():
        build_folds()
    return pq.read_table(folds_path(), columns=columns).to_pandas()


def sample_queries(n: int, fold: int = VAL_FOLD, seed: int = 13) -> pd.DataFrame:
    """Random sample of Source-1 queries from one fold, with normalized text and truth.
    Columns: entity_id, country, name_norm, addr_norm, matched_entity_ids, n_matches."""
    folds = load_folds(["entity_id", "fold", "matched_entity_ids", "n_matches"])
    pool = folds.index[folds["fold"] == fold].to_numpy()
    if n and n < len(pool):
        pool = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    s1 = pq.read_table(store_path("train", "source1"), columns=["entity_id", "country", "name_norm", "addr_norm"])
    q = s1.take(pa.array(pool)).to_pandas()
    sel = folds.loc[pool, ["entity_id", "matched_entity_ids", "n_matches"]].reset_index(drop=True)
    if not (q["entity_id"].to_numpy() == sel["entity_id"].to_numpy()).all():
        raise ValueError("fold table and Source-1 store are out of row order")
    q["matched_entity_ids"] = sel["matched_entity_ids"].to_numpy()
    q["n_matches"] = sel["n_matches"].to_numpy()
    return q


if __name__ == "__main__":
    df = build_folds()
    print(df.groupby("fold").agg(n=("entity_id", "size"), singleton_share=("n_matches", lambda s: (s == 0).mean()),
                                 mean_matches=("n_matches", "mean")).round(4))
