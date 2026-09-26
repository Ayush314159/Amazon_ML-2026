# Ground Truth Analysis

Last updated: 2026-09-26
Scope: `train/train_ground_truth.tsv` (2,206,821 rows), analyzed in full, joined
against `train_source1`'s `country` column (also in full — a single lightweight
`entity_id -> country` mapping, not a full name/address load). Name/address
similarity in §5 is estimated from a **stratified 12,000-query sample** (see
Methodology) with lookups resolved via targeted chunked scans, not a full load of
Source 2/3.

## 1. Overview

- **2,206,821 Source-1 entities**, one ground-truth row each (matches `train_source1`'s
  row count exactly; every S1 entity has a label).
- **7,638,365 total true match pairs** across all S1 entities.
- Mean matches per entity (including singletons): **3.46**.

## 2. Singleton / one-match / multi-match breakdown

| Category | Count | % of all S1 entities |
|---|---|---|
| Singleton (0 matches) | 123,247 | **5.58%** |
| Exactly 1 match | 119,157 | 5.40% |
| 2+ matches ("multi-match") | 1,964,417 | 89.02% |

Full match-count distribution (count of S1 entities by number of true matches):

| # matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| # entities | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |

The distribution is unimodal, peaking at 3 matches, with a long right tail up to 11.
**Singletons (5.58%) are a meaningfully sized, distinct regime** — under the
competition's F_0.5-per-entity-macro-average metric, getting this 5.58% right (by
predicting an empty list) is worth exactly as much as getting a well-matched multi-match
entity right, and a false positive on any of these 123,247 entities scores a hard 0.0 on
that entity regardless of how good the rest of the submission is.

## 3. Matches by data source (S2 vs. S3)

| | Total matches | Per-entity distribution (0 / 1 / 2 / 3 / 4 / 5 / 6+) |
|---|---|---|
| Source 2 | 3,693,619 | 287,745 / 789,108 / 652,779 / 333,957 / 119,078 / 24,154 / 0 |
| Source 3 | 3,944,746 | 266,276 / 716,417 / 668,375 / 372,443 / 145,116 / 35,378 / 2,816 |

Matching modality (does an entity's true matches come from S2 only, S3 only, or both?):

| Modality | Count | % |
|---|---|---|
| Only Source 2 | 143,029 | 6.48% |
| Only Source 3 | 164,498 | 7.45% |
| Both S2 and S3 | 1,776,047 | 80.48% |
| (Singleton, neither) | 123,247 | 5.58% |

**Key finding**: the overwhelming majority (80.5%) of non-singleton entities have
matches in **both** Source 2 and Source 3 — a matcher that only searches one of the two
candidate pools, or weights them very differently, will systematically underperform.
S2 and S3 contribute almost equal total volume (3.69M vs. 3.94M), with S3 contributing
slightly more.

## 4. Uniqueness of matched targets (training data only — see caveat)

Across all 7,638,365 matched-target references in the training ground truth, **every
single one is a distinct `entity_id`** — zero Source-2 or Source-3 record is claimed as
a true match by more than one Source-1 entity in this training set.

**Caveat, stated plainly**: this is a property observed in the *training* ground truth,
not a rule guaranteed to hold for the *test* set's unknown true matches, and it
must not be assumed as a modeling constant. It is, however, a legitimate and useful
**post-processing heuristic to test at validation time**: if the trained matcher ever
assigns the same S2/S3 candidate to two different S1 queries with different confidence
scores, resolving the conflict in favor of the higher-scoring query (Global bipartite
"greedy" or Hungarian-style assignment) is a reasonable tie-break to try, and its effect
on validation F_0.5 should be measured empirically rather than assumed to help.

## 5. True-match name and address similarity (sampled)

Computed over a **stratified sample of 12,000 Source-1 queries** (4,000 singleton,
4,000 exactly-one-match, 4,000 multi-match — sampled this way specifically so the
similarity distribution isn't dominated by high-match-count entities), yielding
**19,437 true-match pairs** after resolving every sampled query and its matched targets
via a targeted chunked scan of `train_source1/2/3` (only the needed ~31,000 specific
`entity_id`s were pulled out of the ~12.5M-row files — not a full load).

| Metric | Value |
|---|---|
| Exact raw string match, name | 4.70% |
| Exact raw string match, address | 2.17% |
| Exact match after normalization (lowercase, accent-strip, punctuation-strip), name | 26.0% |
| Exact match after normalization, address | 8.47% |
| Exact match after stripping legal suffixes (llc/inc/ltd/pvt/…) as well, name | 46.0% |
| Mean name token-Jaccard | 0.640 |
| Median name token-Jaccard | 0.667 |
| 10th-percentile name token-Jaccard | **0.0** |
| Mean name Levenshtein ratio | 0.716 |
| Mean address token-Jaccard | 0.608 |
| 10th-percentile address token-Jaccard | 0.25 |

**Key finding — exact/near-exact matching alone recovers a small minority of true
matches.** Only 4.7% of true-match pairs have byte-identical names, and even after
normalization + legal-suffix stripping, 54% of true matches still don't have an
identical name string. **Zero of the 19,437 sampled true-match pairs had an identical
name AND an identical address** (the `exact_match` bucket in `curated_examples.json` is
empty) — a blocking or matching strategy that requires exact agreement on both fields
will recover essentially nothing.

**Key finding — the tail is genuinely hard, not just "slightly fuzzy."** The 10th
percentile of name token-Jaccard similarity across true matches is **0.0** — meaning at
least 10% of true-match pairs share *no* common word at all in their business name
(e.g. `"Torres Coffee LLC"` truly matching `"t0rrescoffee.com"` — see
`curated_examples.json`, `hardest_true_matches_low_similarity`). Separately,
**74.0% of sampled true matches would be missed entirely by a naive "block only on
exact normalized name" strategy**, and **14.8% have name-Jaccard below 0.2** — i.e. even
a fairly loose bag-of-words name-overlap blocking threshold will still miss a
meaningful share of true matches unless address similarity or another signal is used to
recover them.

### By country (India vs. US, sampled pairs)

| | n pairs | mean name-Jaccard | mean addr-Jaccard | normalized-exact-name % |
|---|---|---|---|---|
| India | 7,770 | 0.544 | 0.691 | 19.3% |
| US | 11,667 | 0.704 | 0.553 | 30.5% |

**Key finding — the hardest signal flips by country.** For US pairs, **name** is the
stronger/more literal signal (higher name-Jaccard, higher exact-normalized-name rate)
while **address** is noisier (lower addr-Jaccard) — consistent with US addresses being
short, structured, and prone to component reordering/abbreviation. For India, it's the
reverse: **address** is the stronger signal (0.691 mean Jaccard) and **name** is noisier
(lower Jaccard, lower exact-match rate) — consistent with more name variation
(transliteration, trade names, longer descriptive names) but more literal address
copying. **A single fixed weighting of name-vs-address similarity is unlikely to be
optimal across both countries**; country-aware (or, per `CLAUDE.md`, country-agnostic
but feature-driven — e.g. let the model learn per-country weighting implicitly via a
country-derived feature) calibration is worth testing in Stage 2 feature design.

## 6. Curated positive examples by phenomenon

Concrete examples for each noise phenomenon named in the problem statement were pulled
from the same 12,000-query sample and saved to `curated_examples.json` (up to 6 examples
per bucket): `likely_typo`, `legal_suffix_abbreviation`, `normalized_exact_name`,
`punctuation_change`, `token_reordering`, `address_variation_name_close`,
`transliteration_script_mismatch`, `partial_or_missing_address`,
`hardest_true_matches_low_similarity`. Two are worth calling out specifically:

- The `transliteration_script_mismatch` bucket's first example
  (`"Ardath Martin Freedom Inc"` vs. `"ARDATH MARTIN FR�EDOM INC"`) is **not** a real
  transliteration case — it is character-encoding corruption (see
  `dataset_profile.md` §7). The bucket's detection rule (one side has a non-ASCII
  byte, the other doesn't) catches both real transliteration and mojibake; a genuine
  transliteration example would need script-specific detection (Devanagari/Tamil/
  Kannada ranges specifically) rather than "any non-ASCII," which is left as follow-up
  work rather than something resolved here.
- `hardest_true_matches_low_similarity` (e.g. `"Torres Coffee LLC"` ↔
  `"t0rrescoffee.com"`) is the most direct evidence that **address must carry real
  weight in the final classifier** — these pairs are only resolvable through the shared
  address, since the names have almost nothing in common as strings.

## 7. Repeated names / ambiguous businesses cross-referenced against ground truth

See `dataset_profile.md` §6 for the sampling methodology. Cross-referencing sampled
repeated-normalized-name groups in Source 2/3 against the full ground truth found **16
confirmed cases** where members of the same repeated-name group are true matches for
**different** S1 queries — i.e. genuine "which one of these same-named businesses is
the right one" ambiguity exists in the data, not just formatting noise, and address is
what disambiguates it. Full list in `ambiguity_examples.json`.

## Methodology

- §1–4 are **full-dataset** (all 2,206,821 ground-truth rows, all match tokens parsed
  and counted; the country join uses `train_source1`'s `(entity_id, country)` column
  pair only, not name/address, keeping that step lightweight).
- §5–7 are **sample-based by design**: resolving true-match name/address similarity for
  *every* one of 7.6M true-match pairs would require loading full name/address text for
  both sides of each pair — effectively a large fraction of the 12.5M-row training
  corpus in memory at once. Per the project's resource constraints, a stratified
  12,000-query sample (chosen to balance singleton/one/multi-match representation, not
  just uniform-random) plus a targeted chunked scan for exactly the ~31,000
  `entity_id`s needed was used instead, completing in under a minute with a small,
  bounded memory footprint. This is treated as a reliable estimate, not a substitute
  for full-dataset counts where those were already cheap enough to compute exactly
  (§1–4).
- Cached intermediate artifact: `gt_cache.parquet` (the ground truth plus each query's
  `country`, 92MB) is kept in the repository root (gitignored) so downstream analysis
  and future modeling code can reload it in ~1 second instead of re-parsing the 122MB
  TSV.
