# PROJECT_STATE.md — Business Entity Resolution Challenge (Amazon ML Challenge 2026)

Last updated: 2026-09-26
Status: **PHASE 1 (data understanding) COMPLETE. No blocking, feature, or model code
written yet.**

This file is the single source of truth for "where are we right now." Update it at the
end of every work session: what changed, what the current best validation F_0.5 is,
and what the next concrete step is. Do not let it go stale.

---

## 1. Repository

- Remote: `https://github.com/Ayush314159/Amazon_ML-2026.git`.
- A documentation/governance PR (`project-setup-docs` branch: `CLAUDE.md`,
  the first version of this file, `experiments/experiment_log.csv`, and a
  populated `.gitignore`) was pushed to origin but **not yet merged** — GitHub CLI
  (`gh`) is not installed on this machine, so the PR had to be opened manually via the
  link GitHub prints on push. Check whether that PR has since been merged before
  branching again from `main`; this session's Phase 1 work (below) should go out as a
  follow-up PR/commit, ideally after confirming `project-setup-docs` landed.
- `.gitignore` now excludes `*.tsv` (raw data), `*_profile.html` (regenerable
  ydata-profiling reports), and `analysis/*.parquet` (regenerable analysis caches,
  see §3). The small JSON analysis summaries under `analysis/` **are** tracked —
  they're the actual Phase 1 findings and are cheap to version.
- `LICENSE`: MIT, copyright Ayush Pandey, 2026.
- **Test files have now been added** (`test/test_source{1,2,3}.tsv`), and the training
  files were moved into `train/`. Layout as of this session:

```
AmznML/
├── .git/
├── .gitignore
├── LICENSE
├── CLAUDE.md
├── PROJECT_STATE.md            (this file)
├── amazon_ml.pdf               (official problem statement, 8 pages)
├── amazon_ml_profile.html      (ydata-profiling report, pre-existing, ~1.7 MB)
├── train2_profile.html         (ydata-profiling report, pre-existing, ~2.4 MB)
├── run_comprehensive_eda.py    (pre-existing 380-line EDA script)
├── EDA.ipynb                   (pre-existing sampled-EDA notebook)
├── experiments/
│   └── experiment_log.csv      (header-only scaffold, no modeling runs yet)
├── analysis/                   (NEW this session — Phase 1 findings, see §3)
│   ├── dataset_profile.md
│   ├── ground_truth_analysis.md
│   ├── train_test_shift.md
│   ├── profile_basic.json                       (tracked — full-dataset stats, all 6 files)
│   ├── ground_truth_stats.json                   (tracked — full ground-truth topology)
│   ├── pair_similarity_summary.json              (tracked — sampled true-match similarity)
│   ├── curated_examples.json                     (tracked — noise-phenomenon examples)
│   ├── ambiguity_stats_lightweight.json          (tracked — sampled repeated-name stats)
│   ├── ambiguity_examples.json                   (tracked — confirmed cross-query ambiguity)
│   ├── france_deep_dive_sampled.json             (tracked — France characterization)
│   ├── shift_diff.json                           (tracked — full train-vs-test diff)
│   ├── gt_cache.parquet                          (gitignored — 92MB regenerable cache)
│   └── train_source2_repeated_name_groups.parquet (gitignored — regenerable sample cache)
├── train/
│   ├── train_ground_truth.tsv  (122 MB)
│   ├── train_source1.tsv       (201 MB)
│   ├── train_source2.tsv       (467 MB)
│   └── train_source3.tsv       (481 MB)
└── test/
    ├── test_source1.tsv        (167 MB, 1,732,544 rows)
    ├── test_source2.tsv        (486 MB, 4,887,273 rows)
    └── test_source3.tsv        (483 MB, 5,082,316 rows)
```

Still missing relative to what the competition package expects: `utils/validate_submission.py`
and `Documentation_template.md`. **Neither exists in the repo yet** — must be obtained
before final packaging/validation can be done locally (see Risks §6).

## 2. Competition (from `amazon_ml.pdf`, read in full) — unchanged, restated for reference

- **Task**: Business Entity Resolution. Source 1 is the deduplicated reference set. For
  every S1 record, find all matching records in Source 2 and/or Source 3 (0, 1, or many).
- **Metric**: **F_0.5**, per-entity, macro-averaged. `F_0.5 = 1.25·P·R / (0.25·P + R)` —
  precision weighted 2× over recall. Singletons scored 1.0 if predicted empty, 0.0 if any
  match is predicted.
- **Country is an open string set**: train = {US, India}; test additionally has
  **France**, now directly confirmed present at ~14–15% of every test file (§3).
- **Outputs**: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (audited;
  matches must be a subset of candidates). Hard format rules apply — see `CLAUDE.md` §3.
- **Model constraint**: MIT/Apache-2.0 licensed, ≤8B parameters.
- **No external data lookups** of any kind.
- Full detail retained in `CLAUDE.md` (permanent rules) rather than duplicated here.

## 3. Phase 1 findings — data understanding (this session)

Full detail lives in three new files under `analysis/`; only the highlights that change
how Phase 2 (blocking) and later stages should be designed are repeated here.

**[analysis/dataset_profile.md](analysis/dataset_profile.md)** — full-dataset profiling
of all 6 source files:
- Source 1 (train and test) is confirmed genuinely deduplicated: **zero** duplicate
  `entity_id`s and **zero** rows sharing an identical (name, address, country) triple, in
  both splits. Source 2/3 are not — ~0.6–1.0% of their rows are exact duplicates of
  another row in the same file.
- **Source 1 is Latin-script-only, in both train and test (0.0% Devanagari/Tamil/Kannada
  in `business_name`, always).** Source 2/3 carry native-script names for a real share of
  Indian businesses (up to ~6.3%). Transliteration/script-normalization is therefore only
  strictly required on the Source 2/3 side.
- **Postal/PIN codes are rare everywhere** — only ~6.5% of US/India addresses contain
  anything ZIP/PIN-shaped, even in training. This is a general property of the free-text
  `business_address` field, not a France-only gap (France is even lower, ~0.4–0.5%, see
  `train_test_shift.md`). Postal-code matching cannot be a primary blocking key.
- ~18% of a 1M-row `train_source3` sample share a normalized business name with another
  row; cross-referencing against the ground truth confirmed **16 real cases of the same
  (normalized) name legitimately belonging to different true businesses** — name alone
  is not a safe blocking key for common/generic names.
- A **distinct encoding-corruption artifact** (literal `�` replacement characters
  embedded in otherwise-normal Latin names) was found in both training and France test
  examples — this must be treated as corrupted data (droppable/wildcard character), not
  confused with genuine script/transliteration handling.

**[analysis/ground_truth_analysis.md](analysis/ground_truth_analysis.md)** — full
ground-truth topology (2,206,821 rows) plus a stratified 12,000-query similarity sample:
- Singleton 5.58% / one-match 5.40% / multi-match 89.02% (matches prior session's
  numbers, now cross-checked against source1's country column too — singleton rate is
  ~identical for US and India, 5.58% each).
- 80.5% of non-singleton entities have matches in **both** S2 and S3 — a matcher must
  search both pools, not just one.
- **Zero of 19,437 sampled true-match pairs had an identical name AND identical
  address.** Only 4.7%/2.2% have byte-identical name/address; even after normalization +
  legal-suffix stripping, 54% still don't have an identical name. **74% of true matches
  would be missed by naive exact-normalized-name blocking**, and 10% of true matches
  share *zero* common name tokens at all (name-Jaccard = 0 at the 10th percentile).
- **The stronger signal flips by country**: for US pairs, name similarity is stronger
  (mean Jaccard 0.70) and address noisier (0.55); for India, address is stronger (0.69)
  and name noisier (0.54, more transliteration/trade-name variation). A single fixed
  name/address weighting is unlikely to be optimal across both.
- Every one of the 7.6M matched targets in the training ground truth is claimed by
  exactly one query (no target reused) — a candidate post-processing heuristic
  (one-target-per-query greedy/Hungarian assignment) to test empirically, **not** to
  assume holds on test data.

**[analysis/train_test_shift.md](analysis/train_test_shift.md)** — France, confirmed and
characterized directly (not inferred):
- France is **~14–15% of every test file**, and its introduction also shifts the
  US/India ratio itself: train is 60/40 US-majority, test is ~45/55 India-majority even
  before counting France.
- French business names are shorter on average (19–21 chars vs. 24–26 for
  US/India-majority populations) — driven by shorter legal-suffix abbreviations
  (SARL, SAS, EURL, SASU, SA, SCI — **zero presence in any training file**, so any
  suffix list built only from training data will not recognize them).
  Normalization's legal-suffix table must be extended with these forms.
  postal/PIN coverage: **~0.4–0.5%**, even lower than the already-sparse US/India rate.
- Accented-Latin/non-ASCII upticks in Source 1 test data (0% in training, several % in
  test) are attributable almost entirely to France, not a broader shift in the US/India
  population — the same accent-normalization work already motivated for Indian name
  variants directly benefits France too.

## 4. Existing code — unchanged from prior session

`run_comprehensive_eda.py` and `EDA.ipynb` remain purely descriptive/sampled EDA
(no blocking, feature, model, or submission code). No changes made to either this
session; Phase 1's real analysis work now lives in `analysis/` instead, computed against
full data or clearly-labeled full/bounded samples rather than a fixed 100k-row sample.

## 5. Resources — updated with this session's operational lesson

- Python 3.13.7, 12 logical CPU cores, 16 GB total RAM, no GPU (`torch.cuda.is_available()`
  is `False`), 238 GB free disk. Same as previously recorded.
- **Operational finding from this session: a naive "load full file + run ~25 vectorized/
  apply-based column statistics" profiling pass on the 5M+/5.3M-row source2/source3
  files pushed available RAM down to ~1.3–1.6 GB free** (of 16 GB total, with ~10GB
  already used by other processes on this machine) while completing successfully in
  ~180–190 seconds each. It did **not** crash, but it was close enough to the ceiling,
  and slow enough in wall-clock time, that a **user-directed correction stopped a
  second, heavier round of full-file profiling mid-run** (a same-shaped ambiguity/
  repeated-name pass over source2 and source3's full name+address columns, plus a
  planned full chunked scan of all three test files for France characterization).
  **Lesson, now codified as a standing rule in `CLAUDE.md`**: default to a single
  bounded sample (e.g. `nrows=1_000_000`, confirmed acceptable since row order in every
  file here is shuffled/interleaved, not grouped) for exploratory statistics, and
  reserve a full-file pass for numbers that are cheap (ground truth, 122MB) or that a
  sample genuinely cannot answer reliably. Never run more than one memory-heavy pass
  concurrently.
- What was salvaged vs. redone after that correction: `profile_basic.json` (the first,
  already-completed full-dataset pass over all 6 files) and `ground_truth_stats.json` /
  `pair_similarity_summary.json` / `curated_examples.json` (all completed before the
  stop) were kept as-is, not recomputed. Only the unfinished ambiguity/France work was
  redesigned to be sample-based and re-run (33.6 seconds total, no memory pressure) —
  see `analysis/dataset_profile.md` Methodology and `analysis/train_test_shift.md`
  Methodology for the full account.
- Installed ML/data libraries: unchanged from prior session (`pandas`, `numpy`, `scipy`,
  `pyarrow`, `scikit-learn`, `xgboost`, `catboost`, `faiss-cpu`, `sentence-transformers`,
  `transformers`, `torch` CPU-only, `networkx`). Still not installed: `lightgbm`,
  `rapidfuzz`, `polars`, `usaddress`, any indic-transliteration package.

## 6. Risks — updated

1. ~~Empty `.gitignore`~~ **Resolved**: `.gitignore` now excludes raw `.tsv`, profiling
   HTML, and regenerable analysis parquet caches.
2. ~~No test data present~~ **Resolved**: test files are now in the repo and have been
   profiled (§3). France is now directly characterized, not unknown.
3. **Still missing**: `utils/validate_submission.py` and `Documentation_template.md`.
   Final packaging/validation cannot be done locally without these — must be obtained.
4. **No GPU, and confirmed-tight RAM headroom under full-dataset profiling** (as low as
   ~1.3GB free mid-run, §5). Phase 2/3 (blocking, feature computation, model training)
   at full 2.2M-query / 10.3M-candidate scale must be designed with chunking/sampling
   from the start, not retrofitted after an OOM. Prototype every new heavy step on a
   bounded sample first, as a hard rule now (`CLAUDE.md`).
5. **Open question on the transliteration/normalization boundary** (locally-run
   pretrained transliteration model vs. "external data augmentation") — unresolved,
   carried over from prior session. Needs a judgment call before Phase 2/3 relies on it.
6. **Model license/size compliance** — unresolved, carried over. Must be checked
   per-model before adoption, not after.
7. **The `project-setup-docs` PR/branch's merge status is unconfirmed** (§1) —
   check before creating a new branch for this session's `analysis/` additions, to avoid
   branching from a stale `main`.
8. **No blocking, feature, model, inference, or submission-format code exists yet.**
   Phase 1 (data understanding) is complete; nothing downstream has been started.

## 7. Recommended next step

Phase 1 is complete. Per `CLAUDE.md`'s staged process, the next stage is **blocking /
candidate-generation design** (Phase 2) — not model training. Concretely:

1. Commit and push `analysis/` (this session's findings) — confirm the
   `project-setup-docs` PR's status first (§1, §6.7) and branch accordingly.
2. Obtain `utils/validate_submission.py` and `Documentation_template.md` (§6.3) —
   still blocking final packaging, independent of modeling progress.
3. Write the entity-level macro F_0.5 scorer as a standalone utility (recommended last
   session, still not done — genuinely the next piece of code, before any blocking
   logic, so every later decision can be checked against it).
4. Design blocking using this session's concrete findings rather than generic
   assumptions: exact/normalized-name matching alone recovers a small minority of true
   matches (§3) and postal-code signal is sparse everywhere (§3) — so blocking should
   combine normalized token-overlap on **both** name and address (not name-only), likely
   via TF-IDF/character-n-gram nearest-neighbor search (`scikit-learn`/`faiss-cpu`, both
   installed) rather than exact-match blocking keys, with country used as a candidate
   filter/feature (not a hardcoded branch) given the confirmed train/test country-mix
   shift (§3). Design and prototype on a bounded sample first, per §5's resource lesson.
5. Track recall@candidates on a held-out validation slice as blocking is built —
   per `CLAUDE.md`, this is the ceiling on everything downstream and must be measured,
   not assumed.

This file should be updated the moment any of the above changes.
