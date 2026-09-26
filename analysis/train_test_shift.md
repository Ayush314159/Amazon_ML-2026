# Train / Test Distribution Shift

Last updated: 2026-09-26
Scope: comparing each `train_sourceN.tsv` against the corresponding `test_sourceN.tsv`,
using the already-computed full-dataset profiles (`profile_basic.json` — see
`dataset_profile.md` for how those were produced) plus a dedicated France-focused
sample (`france_deep_dive_sampled.json`, 1,000,000-row sample per test file — see
Methodology). No country value is hardcoded in any pipeline logic anywhere in this
analysis or the project's rules (`CLAUDE.md`); France is treated throughout as "whatever
string value appears in the `country` column," discovered empirically below, not
assumed.

## 1. Row counts

| Source | Train rows | Test rows | Test/Train ratio |
|---|---|---|---|
| Source 1 | 2,206,821 | 1,732,544 | 0.785 |
| Source 2 | 5,034,616 | 4,887,273 | 0.971 |
| Source 3 | 5,285,603 | 5,082,316 | 0.962 |

Source 1's test set is proportionally smaller relative to train (78.5%) than Source 2/3
(96–97%) — the test candidate pools (S2/S3) are nearly as large as train's, while the
query set (S1) shrank more. This affects candidate-pool density per query and should be
factored into blocking-width choices (candidates-per-query ratios will differ between
train-based tuning and the real test run).

## 2. Country distribution shift — France confirmed and characterized

**Directly confirmed**: `country` in every training file contains exactly `{US, India}`;
every test file additionally contains `France`, at a **consistent ~14–15% of every test
file** regardless of source:

| Source | Train: US / India | Test: US / India / **France** |
|---|---|---|
| 1 | 60.0% / 40.0% | 38.3% / 46.8% / **15.0%** |
| 2 | 59.9% / 40.1% | 38.3% / 47.3% / **14.4%** |
| 3 | 60.0% / 40.0% | 38.3% / 47.3% / **14.4%** |

**Key finding — this is not simply "train's distribution plus a France slice."** The
US/India *ratio itself* shifts: train is 60/40 US-majority, test is ~45/55 India-majority
even before counting France. Any component of the pipeline implicitly tuned to a 60/40
split (e.g. a fixed blend of country-specific heuristics, or validation thresholds
picked assuming train's mix) will see a meaningfully different distribution at test
time. This reinforces `CLAUDE.md`'s standing rule to treat country as an open,
data-driven signal rather than encode assumptions about its distribution.

## 3. France-specific characterization (sampled)

Because France has zero training examples, its characteristics were profiled directly
from the test files via a 1,000,000-row sample per file (see Methodology) rather than
inferred from training data.

| Metric | test_source1 (France rows) | test_source2 | test_source3 |
|---|---|---|---|
| France rows in 1M sample | 149,496 (15.0%) | 143,763 (14.4%) | 143,260 (14.3%) |
| Name length, mean | 19.4 | 21.1 | 21.1 |
| Address length, mean | 50.1 | 39.5 | 40.0 |
| Name has accented Latin char | 15.7% | 24.5% | 23.9% |
| Address has a 5-digit number (postal-code shaped) | **0.40%** | **0.49%** | **0.53%** |

Legal-suffix vocabulary within the France subset (% of France rows containing the
token):

| Suffix | test_source1 | test_source2 | test_source3 |
|---|---|---|---|
| SARL | 28.2% | 19.9% | 19.8% |
| SAS | 20.2% | 14.1% | 13.8% |
| EURL | 6.6% | 5.9% | 5.8% |
| SASU | 4.2% | 4.5% | 4.4% |
| SA | 4.8% | 4.9% | 4.8% |
| SCI | 3.2% | 3.9% | 3.8% |

**Key finding — French business names are meaningfully shorter than US/India names**
(19.4–21.1 mean characters vs. 23.8–25.7 for the overall test population in
`dataset_profile.md` §3), likely because French legal-suffix abbreviations (SARL, SAS)
are shorter than the US/India equivalents ("Limited", "Private Limited",
"Incorporated"). Any name-length-based feature or heuristic calibrated on US/India data
will see systematically shorter names for France.

**Key finding — French addresses almost never contain a postal code in this dataset
(~0.4–0.5%), even lower than the already-low US/India rate (~6.5%, see
`dataset_profile.md` §5).** A blocking/feature strategy that leans on postal-code
matching — already weak for US/India — has **essentially no signal for France at all**.
Country-agnostic candidate generation (name/address token overlap, structural
similarity) is not just "more robust" for France, it is close to necessary, since the
sparse structured signals (ZIP/PIN) that exist for the other two countries barely exist
here.

**Key finding — France introduces a distinct, unfamiliar legal-suffix vocabulary**
(SARL/SAS/EURL/SASU/SA/SCI) with **zero presence in training data** (`SARL` was 0.0% in
every training file per `dataset_profile.md` §5's underlying data). Any legal-suffix
list hardcoded from training data observations (LLC/Inc/Ltd/Pvt/Limited/...) will not
recognize these tokens as suffixes at all unless the suffix list itself is
extended — a clear, concrete action item for the normalization stage.

Two example France records straight from the sample (note the `�` corruption in the
first — see `dataset_profile.md` §7, this is encoding corruption, not a France-specific
phenomenon):

```
"Lège-Cap-Ferret Societe SARL" (test_source1, corrupted to "L�ge-Cap-Ferret..." in the file)
"Comité des Enfant" (test_source2)
```

## 4. Scalar shift summary (percentage-point change, test minus train)

Selected shifts with the largest magnitude (full detail in `shift_diff.json`):

| Signal | Source 1 shift | Source 2 shift | Source 3 shift |
|---|---|---|---|
| addr non-ASCII % | +4.2 | (see json) | (see json) |
| addr accented-Latin % | +4.2 | (see json) | (see json) |
| name accented-Latin % | +2.4 | (see json) | (see json) |
| addr has US-ZIP-shaped number % | −2.3 | (see json) | (see json) |
| name Title Case % | −4.1 | (see json) | (see json) |

The accented-Latin and non-ASCII upticks in Source 1 (which is 0% non-ASCII and 0%
accented-Latin in *training*, per `dataset_profile.md` §5) are attributable entirely to
France — this is the clearest single confirmation that France's introduction is the
dominant driver of train/test shift in the name/address text itself, rather than a
broader distributional change within the US/India population.

## 5. What this means for the pipeline (do not hardcode country)

- Normalization's abbreviation/legal-suffix tables must include the French forms
  identified in §3, discovered from data, not from an assumption about which countries
  exist — the same discovery process (scan for capitalized short tokens at the end of a
  name, above some frequency threshold) should in principle surface a fourth country's
  vocabulary if one appeared, without code changes.
- Any accent-handling/normalization step already needed for Indian name variants
  (`dataset_profile.md` §5, §7) also directly benefits France — one Unicode-normalization
  + accent-stripping function serves both, so this is largely already covered by
  work motivated for other reasons.
- Postal-code-based features/blocking keys should be treated as **low-coverage
  auxiliary signals everywhere**, not just absent for France — §3 shows the "US/India
  only have 6.5% postal-code coverage" finding from `dataset_profile.md` generalizes:
  France just makes an already-weak signal weaker still (0.4–0.5%).
- Validation splits used to tune blocking width / classifier thresholds should account
  for the fact that test's country mix (and therefore difficulty mix, given
  `ground_truth_analysis.md` §5's country-dependent name/address-similarity balance) is
  not the same as train's — a held-out validation slice that matches train's 60/40
  US/India split may give an overly optimistic read on how the pipeline performs on the
  actual test distribution. See `PROJECT_STATE.md` for the recommended validation
  strategy.

## Methodology

- §1, §2 (distribution), §4 are computed by diffing the already-complete, full-dataset
  `profile_basic.json` (see `dataset_profile.md` — Methodology) — no new data loading
  was needed for these sections.
- §3 (France characterization) required a *test-only* profile that training data cannot
  provide. Rather than a full chunked scan of all ~11.7M test rows (previously
  attempted, then explicitly stopped as too slow/heavy for this machine — see
  `PROJECT_STATE.md`), a single bounded read of the **first 1,000,000 rows** of each
  test file was used instead. Row order was spot-checked and found to already interleave
  countries throughout the file (not grouped/sorted), so this behaves as an
  approximately random ~14–20% sample of each file (1M of 1.73M/4.89M/5.08M rows) — large
  enough that the France sub-sample within it (143k–149k rows) is itself a substantial,
  reliable base for the percentages reported. This completed for all three files in
  under 35 seconds total with no memory pressure, versus the multi-minute-per-file,
  memory-heavy full scan it replaced.
