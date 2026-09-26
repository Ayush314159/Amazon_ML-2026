# Business Entity Resolution — pipeline code

Status: validation framework + blocking (candidate generation) implemented and frozen
(see `analysis/blocking_analysis.md`). The matching model and submission writer are not built yet.

All commands run from `code/business_entity_resolution/src`. Data is read from
`<repo>/train`, `<repo>/test` (or `<repo>/dataset/{train,test}` if present).
Intermediate artifacts go to `<repo>/work/` (override with `ER_WORK`).

```bash
pip install -r ../requirements.txt

# 1. normalize each source once -> work/store/{split}_{source}.parquet   (~2.5 min train)
python -m er.store --split train

# 2. validation folds (deterministic crc32(entity_id) % 5; fold 0 = validation)
python -m er.splits

# 3. retrieval indexes for the candidate pools -> work/index/{split}_{source}/   (~6 min train)
python -m er.index --split train --sources source2 source3

# 4. blocking experiments on a validation sample (appends to experiments/*.csv)
python -m er.blocking_experiments --n-queries 20000 --suite confirm
python -m er.blocking_diagnostics --n-queries 20000 --top-k 20

# 5. frozen blocking at full scale: checkpointed, resumable (re-run the same command after an interruption)
python -m er.candidates --split train --fold 0                       # validation fold -> work/candidates/
python -m er.candidates --evaluate --split train --fold 0            # recall / oracle F0.5 at 10..100 per source
python -m er.candidates --split test --n-queries 20000 --seed 13     # after building test stores + indexes
python -m er.candidates --describe --split test --n-queries 20000 --seed 13   # label-free per-country stats

# unit tests (stdlib unittest)
cd .. && python -m unittest discover -s tests
```

## Modules (`src/er/`)

| Module | Purpose |
|---|---|
| `config.py` | paths |
| `data.py` | TSV reading (`sep="\t"`, no quoting), ground truth parsing, folds, submission writer + local validator |
| `normalize.py` | country-agnostic normalization, Indic→Latin transliteration, phonetic key, blocking keys |
| `metrics.py` | entity-level macro F0.5 (competition metric) and blocking metrics |
| `store.py` | chunked, parallel normalization of each source into Parquet |
| `splits.py` | validation folds + query sampling |
| `index.py` | exact-key indexes and per-(source, country) hashed TF-IDF inverted indexes |
| `blocking.py` | candidate retrieval over those indexes |
| `candidates.py` | frozen blocking config (`fuse_nab_pab_r1000_k100`), chunked + resumable generation, evaluation, label-free description |
| `blocking_experiments.py`, `blocking_diagnostics.py` | evaluation harness and miss analysis |
| `resources.py` | runtime / peak-memory tracking |

Resource notes: the largest single step (index build for one ~5M-row source) peaks
at ~3.7 GB RSS; retrieval peaks at ~3.5 GB. Run one heavy step at a time.
