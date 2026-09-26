# Dataset Profile

Last updated: 2026-09-26
Scope: full-dataset statistics for all 6 source files (`train_source1/2/3`,
`test_source1/2/3`). All numbers in this file are **exact, computed over every row**
(not sampled) via chunked/vectorized pandas passes — see Methodology at the end.
Ground-truth-specific numbers are in [ground_truth_analysis.md](ground_truth_analysis.md);
train-vs-test comparisons are in [train_test_shift.md](train_test_shift.md).

## 1. Row counts, schema, load correctness

All seven files (4 train, 3 test) load correctly as 4-or-2-column tab-separated tables
with `sep="\t"` and no parsing errors (row counts below match `wc -l` minus header,
confirming no embedded newlines or tab-count mismatches broke row boundaries).

| File | Rows | Columns |
|---|---|---|
| `train/train_source1.tsv` | 2,206,821 | `entity_id, business_name, business_address, country` |
| `train/train_source2.tsv` | 5,034,616 | same |
| `train/train_source3.tsv` | 5,285,603 | same |
| `train/train_ground_truth.tsv` | 2,206,821 | `source1_entity_id, matched_entity_ids` |
| `test/test_source1.tsv` | 1,732,544 | `entity_id, business_name, business_address, country` |
| `test/test_source2.tsv` | 4,887,273 | same |
| `test/test_source3.tsv` | 5,082,316 | same |

All columns load as plain strings (`dtype=str` forced on ingestion); pandas' default
inference would otherwise risk mis-typing numeric-looking business names/addresses.

## 2. Missingness, duplicates, uniqueness

| File | entity_id dup. | entity_id format valid | name blank | addr blank | dup. name+addr+country rows |
|---|---|---|---|---|---|
| train_source1 | 0 | 100% | 0 | 0 | **0** |
| train_source2 | 0 | 100% | 0 | 168,967 (3.36%) | 50,933 (1.01%) |
| train_source3 | 0 | 100% | 0 | 175,916 (3.33%) | 37,241 (0.70%) |
| test_source1 | 0 | 100% | 0 | 0 | **0** |
| test_source2 | 0 | 100% | 0 | 129,408 (2.65%) | 44,550 (0.91%) |
| test_source3 | 0 | 100% | 0 | 136,098 (2.68%) | 32,154 (0.63%) |

**Key finding — Source 1 really is deduplicated, as documented.** Both train and test
`source1` have zero duplicate `entity_id`s and, more importantly, **zero rows sharing
an identical (business_name, business_address, country) triple**. Source 2 and Source 3
are not: roughly 0.6–1.0% of their rows are exact duplicates of another row in the same
file (same name, address, and country under different `entity_id`s) — a real source of
ambiguity, see §6 and `ground_truth_analysis.md` §4.

`business_address` can be a genuinely empty string in Source 2/3 (min length 0), but
never in Source 1 (min length 11 in both train and test) — Source 1 always has *some*
address text; Source 2/3 sometimes have none at all.

`country` has zero missing values everywhere and exactly the documented value sets:
`{US, India}` in all training files and `test_source1`/`2`/`3`'s **train** portion, and
`{US, India, France}` in the test files (confirmed directly, not assumed — see §5).

## 3. String length profile

| File | name len (mean / median / p95) | addr len (mean / median / p95) |
|---|---|---|
| train_source1 | 24.0 / 24 / 37 | 52.1 / 41 / 103 |
| train_source2 | 25.1 / 25 / 40 | 46.2 / 37 / 96 |
| train_source3 | 25.2 / 25 / 42 | 46.7 / 42 / 91 |
| test_source1 | 23.8 / 24 / 36 | 57.2 / 50 / 105 |
| test_source2 | 25.7 / 25 / 42 | 50.4 / 43 / 99 |
| test_source3 | 25.7 / 25 / 42 | 48.7 / 43 / 94 |

Name lengths are stable across sources and splits. Address lengths are ~5–10 characters
longer in test than train for every source — driven by France (see `train_test_shift.md`).

## 4. Country distribution

| File | US | India | France |
|---|---|---|---|
| train_source1 | 1,323,633 (60.0%) | 883,188 (40.0%) | — |
| train_source2 | 3,016,817 (59.9%) | 2,017,799 (40.1%) | — |
| train_source3 | 3,170,056 (60.0%) | 2,115,547 (40.0%) | — |
| test_source1 | 663,106 (38.3%) | 809,986 (46.8%) | 259,452 (15.0%) |
| test_source2 | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) |
| test_source3 | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) |

France is a consistent **~14–15% of every test file**. Its introduction shifts the
US/India *ratio* too — test is majority-India (~47%) versus train's majority-US (~60%),
not just "India + a France slice." Any model or blocking key implicitly tuned to a
60/40 US/India split will see a different mix at test time even ignoring France
entirely. Full France characterization is in §5.

## 5. Source-specific and script/formatting patterns

| Signal | train_S1 | train_S2 | train_S3 | test_S1 | test_S2 | test_S3 |
|---|---|---|---|---|---|---|
| name non-ASCII % | 0.0 | 15.2 | 11.5 | 2.4 | 19.0 | 14.5 |
| name Devanagari % | 0.0 | 5.4 | 3.0 | 0.0 | 6.3 | 3.6 |
| name Tamil % | 0.0 | 0.7 | 0.4 | 0.0 | 0.8 | 0.4 |
| name Kannada % | 0.0 | 0.7 | 0.4 | 0.0 | 0.9 | 0.5 |
| name accented-Latin % | 0.0 | 1.3 | 1.5 | 2.4 | 4.4 | 4.5 |
| name ALL-UPPERCASE % | 0.0 | 18.9 | 3.0 | 0.0 | 17.5 | 3.2 |
| name Title Case % | 69.6 | 46.8 | 61.0 | 65.5 | 44.4 | 58.5 |

**Key finding — Source 1 is Latin/English-only in both train and test; Source 2/3 are
not.** `train_source1`/`test_source1` show **0.0%** Devanagari, Tamil, or Kannada script
in `business_name`, across all 2.2M + 1.7M rows. Source 2 and Source 3 carry native-script
names for a real fraction of Indian businesses (up to ~6.3% Devanagari in `test_source2`).
This is a structural asymmetry, not noise: matching only ever needs to go **from** a
Latin-script S1 query **to** a possibly-native-script S2/S3 candidate, never the reverse.
A transliteration/normalization step is therefore only strictly required on the S2/S3
side (or as a bidirectional normalization into a shared space), and can be validated by
checking that all S1 output stays 0% non-Latin.

**Key finding — Source 2 is disproportionately ALL-CAPS; Source 1 never is.**
`train_source2`/`test_source2` have ~17–19% fully-uppercase business names, vs. ~3% in
Source 3 and 0% in Source 1. Case must be normalized away before any name comparison —
casing differences are a Source 2 formatting artifact, not a meaningful signal.

Address-side formatting (`addr_has_*` flags, percentage of rows matching):

| Signal | train_S1 | train_S2 | train_S3 | test_S1 | test_S2 | test_S3 |
|---|---|---|---|---|---|---|
| has PO Box | 0.003 | 1.00 | 0.94 | 0.003 | 0.65 | 0.61 |
| has unit/apt/suite/floor | 19.5 | 9.5 | 12.6 | 17.9 | 11.3 | 12.2 |
| has US-ZIP-shaped 5(-digit) number | 6.6 | 6.5 | 6.5 | 4.3 | 4.5 | 4.5 |
| has IN-PIN-shaped 6-digit number | 0.08 | 0.83 | 0.80 | 0.05 | 0.55 | 0.53 |
| has landmark word (near/opp/behind) | 4.5 | 3.9 | 2.9 | 5.2 | 4.6 | 3.4 |

**Key finding — postal/PIN codes are rare across the board, not just missing for
France.** Even in the training data, where a business's country is definitively US or
India, only ~6.5% of addresses contain anything that looks like a 5-digit US ZIP and
under 1% contain a 6-digit Indian PIN. **A blocking or feature strategy that leans on
postal-code matching will have almost no coverage even before France is considered** —
this is a general property of `business_address`'s free-text, often-partial format
(confirmed directly against the PDF's warning about "missing components"), not a
France-specific gap. See `train_test_shift.md` for the France-specific numbers (postal
code is even rarer there, ~0.4–0.5%).

Legal-suffix vocabulary (share of rows containing the token as a whole word), notable
entries:

| Suffix | train_S1 | train_S2 | train_S3 |
|---|---|---|---|
| llc | 16.7% | ~15–17%* | ~15–17%* |
| inc | 11.4%* | similar | similar |
| limited | 24.9%* | similar | similar |
| private limited | 20.7%* | similar | similar |
| pvt | 5.8%* | similar | similar |

*(see `profile_basic.json` for exact per-file counts; the three sources track each
other closely on US/India suffix vocabulary — the real vocabulary shift is train vs.
test / France, covered in `train_test_shift.md`.)*

## 6. Repeated names / addresses and ambiguity (sample-based estimate)

Exact-duplicate rows (§2) already give a **full-dataset, exact lower bound**: 0.6–1.0%
of Source 2/3 rows are byte-for-byt identical in name+address+country to another row in
the same file. To go beyond exact duplicates cheaply, normalized-name repetition and
cross-query ambiguity were estimated from a **1,000,000-row sample of `train_source3`**
(the file's rows are not sorted by any field we could find — country and match difficulty
are interleaved throughout every file we inspected — so a contiguous first-N sample is
effectively a random sample) plus a previously-computed 2,000-group sample of
`train_source2`; see Methodology.

- In the `train_source3` sample, **18.3% of rows share a normalized business name**
  (case/punctuation/accent-insensitive) with at least one other row in the same
  1M-row sample — noticeably higher than the 0.7% exact-duplicate rate, because most of
  this is generic/common business names (e.g. franchise-style or template names), not
  true duplicates.
- Cross-referencing sampled repeated-name groups against the full ground truth
  confirms **real ambiguity exists**: of 3,500 repeated-name groups checked across the
  two sources (2,000 from Source 2, 1,500 sampled from Source 3), **16 groups had
  members claimed as true matches by 2 or more different Source-1 queries** — i.e. the
  same (normalized) business name legitimately refers to multiple distinct real
  businesses in the data, not just formatting noise. Example: the normalized name
  `"1 800 got"` has 9 Source-2 records split across 8 different true Source-1 owners.
  Full examples in `ambiguity_examples.json`.
- **Implication for blocking**: name alone (even normalized) is not a safe blocking key
  in isolation for generic/common names — address must co-determine candidate
  membership, or blocking on name needs a frequency cap that routes very common names
  through address-based sub-blocking instead.

## 7. Suspicious / noteworthy fields

- **Encoding corruption (mojibake), distinct from real transliteration.** Multiple
  records contain a literal Unicode replacement character (`�`) embedded mid-word in
  otherwise-Latin names, e.g. a true match pair with `"Ardath Martin Freedom Inc"` (S1)
  vs. `"ARDATH MARTIN FR�EDOM INC"` (S2), and in the France sample, `"Lège-Cap-Ferret"`
  appearing as `"L�ge-Cap-Ferret"`. This is corrupted source data (a lost/mis-decoded
  byte), not a script or language phenomenon — normalization should treat `�` as a
  wildcard/droppable character, and this should not be confused with genuine
  Devanagari/Tamil/Kannada transliteration handling.
- **Zero fully-exact true-match pairs found in a 19,437-pair sample** (see
  `ground_truth_analysis.md` §5) — worth flagging here because it means naive
  string-equality blocking would recover **zero** candidates in that sample, not just
  "few."
- `business_address` reaching exactly 0 characters only in Source 2/3 (never Source 1)
  is itself a usable feature/flag (`address_missing`), not just something to impute.
- Business names as short as 2–3 characters exist in every source (`name_len.min`);
  these will be effectively unmatchable on name alone and should rely on address.

## Methodology

- **Full-dataset, not sampled**: §1–§5 and the exact-duplicate figures in §2/§6 are
  computed by loading each of the 6 files once, in full, one at a time (not
  concurrently), with `dtype=str` and only the needed columns where possible, then
  freeing memory (`del` + `gc.collect()`) before moving to the next file. This kept
  peak resident memory for any single stage within the machine's ~16GB budget
  (observed available memory during the run: as low as ~1.3GB free at the single most
  expensive step, `train_source2`/`train_source3` full profiling, which completed
  successfully; no swapping-induced failures occurred, but this confirmed profiling is
  running close to the machine's ceiling and should not be re-run casually or run
  concurrently with anything else).
- **Sample-based, explicitly labeled as such**: §6's `train_source3` repeated-name
  estimate and the ambiguity cross-check use a single bounded 1,000,000-row (or
  2,000-group) sample rather than a full 5.3M-row scan, per the project's
  resource-efficiency constraint (`CLAUDE.md` — prototype/estimate on samples first,
  reserve full-dataset passes for statistics that cannot be reliably estimated from a
  sample). Row order was spot-checked and found to be shuffled (not grouped by country
  or match difficulty), so a first-N sample behaves as a random sample.
- Source artifacts: `profile_basic.json` (full-dataset, all 6 files),
  `ambiguity_stats_lightweight.json` and `ambiguity_examples.json` (sampled), all in
  the repository root (gitignored alongside the raw data — see §"reproducing" below).
- **Reproducing**: the profiling script that produced `profile_basic.json` took
  75s / 183s / 190s / 61s / 188s / 327s for `train_source1/2/3` and `test_source1/2/3`
  respectively (~17.5 minutes total, sequential). It should be run as a single
  foreground or background job, one file at a time, never concurrently with another
  memory-heavy job — see `CLAUDE.md` for the standing resource-usage rule this
  finding established.
