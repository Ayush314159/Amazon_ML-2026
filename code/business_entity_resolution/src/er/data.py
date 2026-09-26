"""Reading competition TSVs, ground truth, validation folds, and submission files."""
import csv
import zlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path, usecols=None, nrows=None, chunksize=None):
    """Read a competition TSV exactly as written: tab-separated, no quoting, no NA inference."""
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        quoting=csv.QUOTE_NONE, usecols=usecols, nrows=nrows, chunksize=chunksize,
        encoding="utf-8",
    )


def parse_id_list(s: str) -> tuple:
    return tuple(t for t in s.split(",") if t) if s else ()


def load_ground_truth() -> pd.DataFrame:
    """Columns: source1_entity_id, matched_entity_ids (raw string)."""
    return read_tsv(config.ground_truth_path())


def truth_map(gt: pd.DataFrame, ids: Iterable = None) -> dict:
    if ids is not None:
        gt = gt[gt["source1_entity_id"].isin(set(ids))]
    return {e: frozenset(parse_id_list(m)) for e, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def fold_of(entity_ids: Iterable, n_folds: int = 5) -> np.ndarray:
    """Deterministic fold assignment from the entity_id string (stable across runs and machines)."""
    return np.fromiter((zlib.crc32(e.encode()) % n_folds for e in entity_ids), dtype=np.int8)


# ---- submission files -------------------------------------------------------

def write_id_lists(path, mapping: Mapping, ids: Iterable, column: str) -> None:
    """Write one row per Source-1 id; an empty list becomes an empty field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{column}\n")
        for e in ids:
            vals = mapping.get(e, ())
            seen = dict.fromkeys(vals)  # de-duplicate, keep order
            f.write(f"{e}\t{','.join(seen)}\n")


def validate_submission(matching_path, candidate_path, s1_ids: Iterable, target_ids: set) -> list:
    """Local re-implementation of the rules in amazon_ml.pdf (the official
    utils/validate_submission.py is not in this repo yet). Returns a list of issues."""
    issues = []
    s1_ids = list(s1_ids)
    s1_set = set(s1_ids)
    parsed = {}
    for label, path, col in (("matching", matching_path, "matched_entity_ids"),
                             ("candidate", candidate_path, "candidate_entity_ids")):
        df = read_tsv(path)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{label}: header must be source1_entity_id<TAB>{col}, got {list(df.columns)}")
            continue
        dup = df["source1_entity_id"].duplicated().sum()
        if dup:
            issues.append(f"{label}: {dup} duplicate source1_entity_id rows")
        missing = s1_set - set(df["source1_entity_id"])
        if missing:
            issues.append(f"{label}: {len(missing)} Source-1 entities missing")
        extra = set(df["source1_entity_id"]) - s1_set
        if extra:
            issues.append(f"{label}: {len(extra)} unknown source1_entity_id values")
        m = {}
        n_dup_in_list = n_bad = 0
        for e, s in zip(df["source1_entity_id"], df[col]):
            lst = parse_id_list(s)
            n_dup_in_list += len(lst) != len(set(lst))
            n_bad += sum(1 for t in lst if t not in target_ids)
            m[e] = set(lst)
        if n_dup_in_list:
            issues.append(f"{label}: {n_dup_in_list} rows with duplicate ids inside the list")
        if n_bad:
            issues.append(f"{label}: {n_bad} ids that are not Source-2/3 ids of this split")
        parsed[label] = m
    if "matching" in parsed and "candidate" in parsed:
        not_subset = sum(1 for e, s in parsed["matching"].items() if not s <= parsed["candidate"].get(e, set()))
        if not_subset:
            issues.append(f"{not_subset} rows whose matches are not a subset of their candidates")
    return issues
