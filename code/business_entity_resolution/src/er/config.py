"""Filesystem locations. Override the repository root with the ER_ROOT env var."""
import os
from pathlib import Path

# src/er/config.py -> parents: er, src, business_entity_resolution, code, <repo root>
REPO_ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[4]))

# The competition package ships data under dataset/{train,test}; this repo keeps
# train/ and test/ at the root. Accept either layout.
DATA_DIR = REPO_ROOT / "dataset" if (REPO_ROOT / "dataset").exists() else REPO_ROOT
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

WORK_DIR = Path(os.environ.get("ER_WORK", REPO_ROOT / "work"))
STORE_DIR = WORK_DIR / "store"
INDEX_DIR = WORK_DIR / "index"
SPLIT_DIR = WORK_DIR / "splits"

EXPERIMENT_DIR = REPO_ROOT / "experiments"

SOURCES = ("source1", "source2", "source3")


def source_path(split: str, source: str) -> Path:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    return base / f"{split}_{source}.tsv"


def ground_truth_path() -> Path:
    return TRAIN_DIR / "train_ground_truth.tsv"
