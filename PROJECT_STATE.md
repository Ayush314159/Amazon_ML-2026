# PROJECT_STATE.md — Business Entity Resolution Challenge (Amazon ML Challenge 2026)

Last updated: 2026-09-26
Status: **INSPECTION COMPLETE — no modeling started.**

This file is the single source of truth for "where are we right now." Update it at the
end of every work session: what changed, what the current best validation F_0.5 is,
and what the next concrete step is. Do not let it go stale.

---

## 1. Repository

- Remote: `https://github.com/Ayush314159/Amazon_ML-2026.git`, branch `main`.
- Git history: 2 commits (`Initial commit`, `Add EDA and analysis files`).
- `.gitignore` is **empty** — the four large `.tsv` files (1.27 GB total) are currently
  **untracked**, not ignored. Risk: someone runs `git add .` and commits 1.27 GB of data
  to the repo. See Risks (§6).
- `LICENSE`: MIT, copyright Ayush Pandey, 2026.
- Flat layout today — no `dataset/`, `src/`, `code/`, `output/`, or `utils/` directories
  exist yet. The competition's expected submission structure (see §2) has to be built
  from scratch; nothing currently on disk matches it.

Current top-level contents:

```
AmznML/
├── .git/
├── .gitignore                 (empty)
├── LICENSE                    (MIT)
├── amazon_ml.pdf              (official problem statement, 8 pages)
├── amazon_ml_profile.html     (ydata-profiling report, ~1.7 MB)
├── train2_profile.html        (ydata-profiling report on a second dataset, ~2.4 MB)
├── run_comprehensive_eda.py   (380-line standalone EDA script)
├── EDA.ipynb                  (633-cell-source notebook, same analyses, richer prose)
├── train_ground_truth.tsv     (122 MB, untracked)
├── train_source1.tsv          (201 MB, untracked)
├── train_source2.tsv          (467 MB, untracked)
└── train_source3.tsv          (481 MB, untracked)
```

Missing relative to what the competition package expects: `dataset/train/`,
`dataset/test/` (no test files have been provided/downloaded yet), `utils/validate_submission.py`,
`Documentation_template.md`. **None of these exist in the repo yet** — confirm with the
competition portal whether test files and the validator script are provided separately
and not yet downloaded, or whether they need to be requested.

## 2. Competition (from `amazon_ml.pdf`, read in full)

- **Task**: Business Entity Resolution. Source 1 is the deduplicated reference set. For
  every S1 record, find all matching records in Source 2 and/or Source 3 (0, 1, or many).
- **Files**: tab-separated (`sep="\t"`). Ground truth: `source1_entity_id`,
  `matched_entity_ids` (comma-separated, empty = no match).
- **Test set** additionally contains **France**, absent from training (train = US, India
  only). Country must be handled as an **open string set** — no hardcoding to 2 values,
  no filtering/one-hot restricted to `{US, India}`. Every test S1 entity, including
  French ones, must appear in the submission.
- **Required outputs** (`output/` folder in the final zip):
  - `matching_results.tsv` — final matches, **the only file scored** on the leaderboard.
  - `candidate_pairs.tsv` — the exact candidate set fed to the final matching model
    (last blocking stage output, not an earlier raw pass). Not scored directly, but
    used to audit blocking recall/reduction ratio, and **every ID in
    `matching_results.tsv` must appear in `candidate_pairs.tsv`** (subset constraint).
- **Hard format rules** (checked by `utils/validate_submission.py`, not yet present
  locally — must be obtained before final packaging):
  - Exactly one row per test S1 entity in both output files.
  - Empty `matched_entity_ids` / `candidate_entity_ids` allowed (singletons).
  - No duplicate IDs within a list; no duplicate `source1_entity_id` rows.
  - IDs must be S2-/S3- only, must exist in the test set, no S1 self-matches.
  - Matches ⊆ candidates.
- **Metric**: **F_0.5**, per-entity, **macro-averaged** across all S1 entities.
  `F_0.5 = 1.25·P·R / (0.25·P + R)` — precision weighted 2× over recall. A correctly
  predicted singleton (no true matches, predicted empty) scores **1.0**; predicting any
  match for a true singleton scores **0.0**. This means over-predicting is punished
  harder than under-predicting, and singleton accuracy is worth real points.
- **Model constraint**: final model must be **MIT or Apache-2.0 licensed** and
  **≤ 8B parameters**. Must verify license of any pretrained model before using it (no
  gated/research-only checkpoints, nothing over 8B params).
- **Strictly prohibited**: any external lookups — commercial ER APIs, government
  business-registry lookups, geocoding APIs, any internet-sourced data augmentation.
  Everything must be derivable from the three provided files only. Enforcement is by
  code/methodology audit; violation = disqualification.
- **Deliverable**: a zip with `output/` (both TSVs), `code/business_entity_resolution/`
  (runnable `src/`, `README.md`, `requirements.txt`), and a filled
  `Documentation_template.md` (not yet present locally — need to obtain this template)
  describing methodology, blocking strategy, model/features, and licensing.
- **Leaderboard**: public subset during the challenge, private subset revealed after;
  final ranking uses the private leaderboard, but you submit predictions for the full
  test set both times.

Video walkthrough referenced in the PDF (link not yet followed — text-only inspection
so far).

## 3. Datasets (training only — no test files present yet)

All counts verified by direct inspection of the full files (not the EDA notebook's
100k-row sample), 2026-09-25/26.

| File | Rows | Columns |
|---|---|---|
| `train_ground_truth.tsv` | 2,206,821 | `source1_entity_id`, `matched_entity_ids` |
| `train_source1.tsv` | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_source2.tsv` | 5,034,616 | same 4 columns |
| `train_source3.tsv` | 5,285,603 | same 4 columns |

Key facts established by direct computation on the full ground-truth file:

- Every S1 entity has exactly one ground-truth row (2,206,821 unique `source1_entity_id`,
  no duplicates).
- **5.58% of S1 entities are singletons** (zero true matches) — non-trivial to get right
  under F_0.5, since these are worth 1.0 for correct empty predictions.
- Total true match pairs: 7,638,365. Mean ≈ 3.46 matches per S1 entity when matches exist;
  distribution peaks at 3–4 matches, ranges 0–11.
- Matches split roughly evenly between S2 and S3 per query (means ~1.7 S2 + ~1.8 S3), no
  query is matched to only-S2 or only-S3 exclusively as a rule — mixed-source ground
  truth is common.
- **Every one of the 7,638,365 matched S2/S3 target IDs is unique across the whole
  ground truth** — no S2 or S3 record is claimed as a match by more than one S1 entity.
  This is a hard structural property of the *training* data and a strong candidate
  post-processing signal (assign each target to at most one query) — **must be
  re-verified on real predictions, not assumed to hold by construction on unseen test
  data**, but worth testing during blocking/matching design.
- Country distribution: source1 ≈ 60% US / 40% India; source2 and source3 similar splits.
  No France rows exist anywhere in training (confirms PDF's claim).
- Real matched-pair examples (pulled directly from the data) show: casing differences,
  OCR-style single-letter typos (`GTRDH` for "Grand", `PRlVATE` for "Private" using a
  lowercase L), accented character corruption (`ópportunity`, `Éar`), abbreviation
  swaps (Drive/Dr, Limited/Ltd, `&`/`+`/`and`), reordered address components, house
  number drift (7622 → 762, 00132 → 132), and business-name rewordings that still refer
  to the same entity by matching on address (e.g. "Vision Health P.C." vs. "Vision P.C.
  Center" at the identical address).
- Non-Latin scripts appear in Indian records (Hindi, Tamil, Kannada) describing
  businesses that also appear transliterated/translated to English elsewhere in the
  data — this is a real signal a name-only matcher would miss without some form of
  script normalization built from the training data itself (no external transliteration
  services, per the no-external-data rule — must confirm whether a static, locally
  installed transliteration *library* — e.g. an offline character-mapping table — counts
  as "external data" or is acceptable as a deterministic algorithm; flagged as an open
  question, see §6).

No test files (`dataset/test/*.tsv`) exist in this repository yet. All numbers above are
training-set only; nothing about the France subset or overall test-set size/shape is
known firsthand.

## 4. Existing code

- **`run_comprehensive_eda.py`** (380 lines): a standalone script, structured as
  reusable functions — `load_data`, `audit_data_hygiene`, `analyze_entity_ids`,
  `analyze_country_distribution`, `analyze_business_names`, `analyze_business_addresses`,
  `analyze_ground_truth`, `analyze_pair_similarity`. Loads with `SAMPLE_ROWS` (partial
  loads for speed). Purely descriptive/analytical — no blocking, no features, no model
  code, no train/val split logic, no submission-format code exists anywhere in the repo.
- **`EDA.ipynb`** (633 source-lines across cells): the same analyses in notebook form
  with markdown narration ("Phase 1" through "Phase 10+"), using a 100,000-row sample by
  default (`SAMPLE_ROWS = 100_000`). Covers hygiene/null audit, entity-ID format and
  cross-source collision checks, country profiling, business-name NLP stats (length,
  casing, legal-suffix frequency), address structural stats (PO box, unit, postal-code
  detection), ground-truth matching topology, and a Phase 10 "true match pair similarity"
  analysis using Jaccard token overlap on name/address for known matched pairs. No
  modeling cells exist past this exploratory analysis.
- **`amazon_ml_profile.html`**, **`train2_profile.html`**: pre-generated ydata-profiling
  (pandas-profiling) HTML reports, presumably from a subset load — not yet reviewed in
  the browser; flagged as unread reference material, not a blocker.
- No blocking/candidate-generation code, no feature-engineering code, no model training
  code, no prediction/inference code, no `utils/validate_submission.py`, no
  `requirements.txt`, and no submission scaffold (`output/`, `code/business_entity_resolution/`)
  exist anywhere in this repo as of this inspection.

## 5. Resources (Python environment, hardware)

- Python 3.13.7.
- **12 logical CPU cores.**
- **16 GB total RAM**, but only **~5.9 GB currently available** (64% already in use by
  other processes on this machine at inspection time) — re-check available RAM before
  running full 1.27 GB × multiple-dataframe pipelines; may need chunked/streaming
  processing or to close other applications.
- **No GPU** — `torch.cuda.is_available()` is `False`, no `nvidia-smi`. All modeling
  (including any transformer/embedding work) must run on CPU. This is a significant
  constraint on Stage-2 neural options (e.g. a cross-encoder fine-tune) — expect it to
  be slow at full 2.2M-query scale; plan to prototype on samples and consider whether
  full-scale neural re-ranking is feasible in the available compute budget at all.
- 238 GB free disk space — not a constraint.

Installed ML/data libraries relevant to this task (checked via `pip list`):

| Installed | Notes |
|---|---|
| `pandas` 2.2.3, `numpy` 2.2.2, `scipy` 1.15.2, `pyarrow` 19.0.1 | tabular/data handling; Parquet available via pyarrow |
| `scikit-learn` 1.6.1 | TF-IDF, nearest neighbors, standard baselines |
| `xgboost` 3.0.2, `catboost` 1.2.8 | gradient-boosted trees for the pair classifier |
| `faiss-cpu` 1.15.0 | approximate nearest-neighbor search for embedding-based blocking |
| `sentence-transformers` 5.7.0, `transformers` 5.15.0, `torch` 2.13.0 (CPU build), `tensorflow` 2.20.0 | neural embeddings / transformer models available, CPU-only |
| `networkx` 3.6.1 | useful for graph-based post-processing (e.g. one-target-per-query assignment, connected components) |
| `jupyter_client`/`jupyter_core` | notebook execution |

**Notably not installed** (would need to be added, and license-checked, before use):
`lightgbm` (mentioned as the natural choice in earlier planning but not present —
`xgboost`/`catboost` are available now instead), `rapidfuzz`, `jellyfish`, `polars`,
`usaddress`, any indic-transliteration package, `recordlinkage`/`dedupe`. None of these
are yet a blocker since no modeling code has been written; just noting the gap.

## 6. Risks

1. **Empty `.gitignore` with 1.27 GB of untracked data files sitting in the repo root.**
   A careless `git add .` / `git add -A` will attempt to commit the training data to
   GitHub. Needs a `.gitignore` entry for `*.tsv` (or a `data/` convention) before any
   commit that touches these files.
2. **No test data present.** All dataset facts in §3 are training-only. The France
   subset's actual naming/address conventions are completely unknown until test files
   are obtained — any France-handling logic will have to be built defensively (generic,
   script/format-agnostic) and validated only against hand-constructed synthetic
   examples until real test data arrives.
3. **Missing competition artifacts**: `utils/validate_submission.py` and
   `Documentation_template.md` are referenced by the PDF but not present in this repo.
   Final packaging cannot be verified locally without the validator script — must be
   obtained (from the competition portal/resource package) before a submission is
   attempted.
4. **No GPU**, only ~6 GB free RAM at last check. Any plan involving transformer
   embeddings or fine-tuning at full 2.2M×10.3M scale needs a compute-budget check
   first; may need to lean more heavily on classical (TF-IDF/rapidfuzz/tree-model)
   methods for Stage 1/2 and reserve neural components for a smaller reranking pass.
5. **Open question on the transliteration/normalization boundary.** The no-external-
   data rule bans external lookups/APIs/geocoding but doesn't explicitly address
   whether a locally-run, pre-trained transliteration model (e.g. from
   `sentence-transformers`/`transformers`, already on disk, no network call) counts as
   "external data augmentation." Needs a judgment call or clarification before relying
   on it; safest default is to treat only *network calls at runtime* as prohibited, and to
   document clearly in the methodology write-up whichever choice is made.
6. **Model license/size compliance is a hard disqualifying constraint** (MIT/Apache-2.0,
   ≤8B params) but has not yet been checked against any specific model choice — must be
   verified per-model before it's adopted, not after the fact.
7. **No modeling code, blocking code, feature code, or submission-format code exists
   yet.** Everything in `src/` has to be written from scratch; current repo state is
   exploration only.

## 7. Recommended next step

Per governing instructions in `CLAUDE.md`: **do not start building the blocking or
matching model yet.** The immediately useful, low-risk next actions are:

1. Add a `.gitignore` entry for the raw `.tsv` data files (Risk #1) before any further
   git operations.
2. Obtain the missing competition artifacts — test files, `utils/validate_submission.py`,
   `Documentation_template.md` — since none of the local pipeline can be verified for
   format compliance without them.
3. Write the entity-level macro F_0.5 scorer as a standalone, tested utility (no model
   dependency) — this is safe, small, and every later modeling decision needs it.
4. Only after 1–3, begin scaffolding `code/business_entity_resolution/src/` structure
   (empty modules for normalization, blocking, features, model, inference) and start
   the blocking/candidate-generation design, since the PDF explicitly flags blocking
   recall as the ceiling on everything downstream.

This file should be updated the moment any of the above changes.
