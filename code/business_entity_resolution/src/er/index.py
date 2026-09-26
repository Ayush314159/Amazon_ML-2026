"""Reusable retrieval structures over a normalized store (built once, loaded per experiment).

1. KeyIndex: exact-match lookup on a derived string key. Stored as sorted uint64
   key hashes + int32 row positions (~12 bytes/record); lookups are searchsorted.
2. PartitionIndex: hashed TF-IDF inverted index for one (source, country) partition.
   Features are namespaced name tokens ("n:"), address tokens ("a:"),
   house-number bigrams ("b:") and phonetic name keys ("p:", see normalize.phonetic_key). Rows are L2-normalized TF-IDF vectors (X, N x F);
   posting lists are X transposed (P, F x N), so a query's cost is the total length
   of the posting lists it touches, not the partition size.

Partitions are keyed by whatever `country` strings occur in the store; no country
value is special-cased.

    python -m er.index --split train --sources source2 source3
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher

from . import config
from .normalize import addr_set_key, name_core_key, phonetic_key
from .resources import Tracker, available_gb
from .store import store_path

N_FEATURES = 2 ** 24
_HASHER = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False, dtype=np.float32)

KEY_FUNCS = {
    "name_exact": lambda n, a: n,
    "name_core": lambda n, a: name_core_key(n),
    "addr_exact": lambda n, a: a,
    "addr_set": lambda n, a: addr_set_key(a),
}


NAMESPACES = "nabp"  # name token, address token, house-number bigram, phonetic name key


def record_features(name_norm: str, addr_norm: str) -> list:
    nt = name_norm.split()
    at = addr_norm.split()
    feats = ["n:" + t for t in nt]
    feats += ["a:" + t for t in at]
    feats += ["b:" + at[i] + "_" + at[i + 1] for i in range(len(at) - 1) if at[i].isdigit()]
    feats += ["p:" + k for k in {phonetic_key(t) for t in nt} if k]
    return feats


def hash_features(feature_lists) -> sp.csr_matrix:
    X = _HASHER.transform(feature_lists)
    X.sum_duplicates()
    X.data[:] = 1.0  # binary term presence
    return X


def idf_from_df(df: np.ndarray, n_docs: int) -> np.ndarray:
    return (np.log((n_docs + 1.0) / (df.astype(np.float64) + 1.0)) + 1.0).astype(np.float32)


def l2_normalize_rows(X: sp.csr_matrix, chunk: int = 500_000) -> sp.csr_matrix:
    """In-place row L2 normalization, processed in row chunks to avoid full-size temporaries."""
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        a, b = X.indptr[s], X.indptr[e]
        lengths = np.diff(X.indptr[s:e + 1])
        local = np.repeat(np.arange(e - s, dtype=np.int32), lengths)
        d = X.data[a:b]
        sq = np.bincount(local, weights=d.astype(np.float64) ** 2, minlength=e - s)
        inv = np.zeros_like(sq)
        nz = sq > 0
        inv[nz] = 1.0 / np.sqrt(sq[nz])
        d *= inv[local].astype(np.float32)
    return X


def hash_strings(keys) -> np.ndarray:
    return pd.util.hash_array(np.asarray(keys, dtype=object)).astype(np.uint64)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s) or "blank"


def index_dir(split: str, source: str):
    return config.INDEX_DIR / f"{split}_{source}"


# ---------------------------------------------------------------------------
class KeyIndex:
    def __init__(self, hashes: np.ndarray, rows: np.ndarray):
        self.hashes, self.rows = hashes, rows

    def lookup(self, query_hashes: np.ndarray, max_group: int) -> list:
        """For each query hash, the store row positions sharing that key (empty if > max_group)."""
        lo = np.searchsorted(self.hashes, query_hashes, side="left")
        hi = np.searchsorted(self.hashes, query_hashes, side="right")
        out = []
        for a, b in zip(lo, hi):
            out.append(self.rows[a:b] if 0 < b - a <= max_group else self.rows[:0])
        return out

    def save(self, path):
        np.savez(path, hashes=self.hashes, rows=self.rows)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["hashes"], z["rows"])


class PartitionIndex:
    def __init__(self, X: sp.csr_matrix, df: np.ndarray, rows: np.ndarray):
        self.X = X            # N x F, L2-normalized TF-IDF
        self.df = df          # document frequency per hashed feature (length F)
        self.rows = rows      # partition-local row -> store row position
        self.n_docs = X.shape[0]
        self.idf = idf_from_df(df, self.n_docs)
        self._P = None

    @property
    def P(self) -> sp.csr_matrix:
        if self._P is None:
            self._P = self.X.T.tocsr()
        return self._P

    @classmethod
    def build(cls, name_norm, addr_norm, rows, chunk: int = 200_000):
        parts = []
        for s in range(0, len(rows), chunk):
            parts.append(hash_features(record_features(n, a) for n, a in
                                       zip(name_norm[s:s + chunk], addr_norm[s:s + chunk])))
        X = sp.vstack(parts, format="csr")
        df = np.bincount(X.indices, minlength=N_FEATURES).astype(np.int32)
        idf = idf_from_df(df, X.shape[0])
        X.data = idf[X.indices]
        return cls(l2_normalize_rows(X).tocsr(), df, np.asarray(rows, dtype=np.int32))

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "X.npz", self.X, compressed=False)
        nz = np.flatnonzero(self.df)
        np.savez(path / "meta.npz", df_idx=nz.astype(np.int32), df_val=self.df[nz], rows=self.rows)

    @classmethod
    def load(cls, path):
        X = sp.load_npz(path / "X.npz").tocsr()
        m = np.load(path / "meta.npz")
        df = np.zeros(N_FEATURES, dtype=np.int32)
        df[m["df_idx"]] = m["df_val"]
        return cls(X, df, m["rows"])


# ---------------------------------------------------------------------------
def build_source_indexes(split: str, source: str) -> dict:
    """Build every key index and every country partition index for one store.

    Only the normalized columns are loaded (raw text stays on disk)."""
    out = index_dir(split, source)
    out.mkdir(parents=True, exist_ok=True)
    report = {"split": split, "source": source, "key_indexes": {}, "partitions": {}}
    table = pq.read_table(store_path(split, source), columns=["country", "name_norm", "addr_norm"])
    name_norm = table.column("name_norm").to_pylist()
    addr_norm = table.column("addr_norm").to_pylist()
    country = table.column("country").combine_chunks()
    del table
    report["rows"] = len(name_norm)

    for kname, fn in KEY_FUNCS.items():
        with Tracker() as t:
            keys = [fn(n, a) for n, a in zip(name_norm, addr_norm)]
            h = hash_strings(keys)
            nonblank = np.fromiter((bool(k) for k in keys), dtype=bool, count=len(keys))
            del keys
            rows = np.flatnonzero(nonblank).astype(np.int32)
            h = h[nonblank]
            order = np.argsort(h, kind="stable")
            KeyIndex(h[order], rows[order]).save(out / f"key_{kname}.npz")
            _, counts = np.unique(h, return_counts=True)
        report["key_indexes"][kname] = {
            "seconds": round(t.seconds, 1), "distinct_keys": int(len(counts)),
            "max_group": int(counts.max()), "keys_with_group_gt_20": int((counts > 20).sum()),
        }
        print(f"    key {kname}: {report['key_indexes'][kname]}", flush=True)

    for c in pc.unique(country).to_pylist():
        rows = np.flatnonzero(pc.equal(country, c).to_numpy(zero_copy_only=False)).astype(np.int32)
        with Tracker() as t:
            idx = PartitionIndex.build([name_norm[i] for i in rows], [addr_norm[i] for i in rows], rows)
            idx.save(out / f"part_{_safe(c)}")
        report["partitions"][c] = {
            "dir": f"part_{_safe(c)}", "docs": int(idx.n_docs), "nnz": int(idx.X.nnz),
            "seconds": round(t.seconds, 1), "peak_rss_gb": round(t.peak_rss_gb, 2),
        }
        print(f"    partition {c!r}: {report['partitions'][c]}", flush=True)
        del idx
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


def load_manifest(split: str, source: str) -> dict:
    return json.loads((index_dir(split, source) / "manifest.json").read_text())


def load_entity_ids(split: str, source: str) -> pa.Array:
    """Compact Arrow array of entity_ids in store row order (~16 bytes/record)."""
    return pq.read_table(store_path(split, source), columns=["entity_id"]).column(0).combine_chunks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sources", nargs="+", default=["source2", "source3"])
    args = ap.parse_args()
    for s in args.sources:
        t0 = time.time()
        print(f"[+] indexing {args.split}_{s} (avail {available_gb():.1f} GB)", flush=True)
        build_source_indexes(args.split, s)
        print(f"    done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
