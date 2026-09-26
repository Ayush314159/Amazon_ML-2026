"""Normalized record store: each source TSV is parsed and normalized once, in
chunks, and written to Parquet. Everything downstream reads these stores
(column subsets, row-group streaming) instead of re-parsing the raw TSVs.

    python -m er.store --split train --sources source1 source2 source3
"""
import argparse
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .data import read_tsv
from .normalize import normalize_addr, normalize_name
from .resources import Tracker, available_gb

CHUNK = 250_000
SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("country", pa.string()),
    ("business_name", pa.string()), ("business_address", pa.string()),
    ("name_norm", pa.string()), ("addr_norm", pa.string()),
])


def store_path(split: str, source: str):
    return config.STORE_DIR / f"{split}_{source}.parquet"


def _normalize_chunk(df):
    df = df.copy()
    df["name_norm"] = [normalize_name(x) for x in df["business_name"]]
    df["addr_norm"] = [normalize_addr(x) for x in df["business_address"]]
    return pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA, preserve_index=False)


def build_store(split: str, source: str, workers: int = 4) -> dict:
    out = store_path(split, source)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    n = 0
    with Tracker() as t:
        chunks = read_tsv(config.source_path(split, source), chunksize=CHUNK)
        with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer, Pool(workers) as pool:
            # imap keeps at most a few chunks in flight -> bounded memory
            for table in pool.imap(_normalize_chunk, chunks):
                writer.write_table(table)
                n += table.num_rows
    tmp.replace(out)
    return {"split": split, "source": source, "rows": n, "seconds": round(t.seconds, 1),
            "peak_rss_gb": round(t.peak_rss_gb, 2), "path": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--sources", nargs="+", default=list(config.SOURCES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for s in args.sources:
        if store_path(args.split, s).exists() and not args.force:
            print(f"[skip] {args.split}_{s} store exists")
            continue
        t0 = time.time()
        print(f"[+] building {args.split}_{s} store (avail {available_gb():.1f} GB)", flush=True)
        print(build_store(args.split, s, args.workers), flush=True)
        print(f"    done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
