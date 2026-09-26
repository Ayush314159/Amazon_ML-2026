# Blocking (Candidate Generation) Analysis — Phase 2

Last updated: 2026-09-26. Status of the phase: see `PROJECT_STATE.md`.

All numbers below are copied from `experiments/blocking_results.csv`,
`experiments/blocking_runs/*.json`, `analysis/blocking_misses_k20.json` and
`work/candidates/*/evaluation.json`. Evidence labels: **VERIFIED** = measured;
**INFERRED** = derived from measurements; **HYPOTHESIS** = untested.

## 1. What blocking has to do, and how it is measured

For each Source-1 query we must return a short, ranked list of Source-2/3
candidates drawn from pools of 10,320,219 (train) records. Anything missing
from this list can never be matched later, so **pair recall** (share of true
(query, target) pairs present among the candidates) is the ceiling for the
whole pipeline. Metrics (implemented in `src/er/metrics.py`):

| Metric | Meaning |
|---|---|
| pair recall | share of true pairs present among candidates |
| entity full coverage | share of non-singleton queries whose *every* true match is a candidate |
| oracle F0.5 | macro F0.5 of a perfect matcher restricted to the candidates (singletons predicted empty); the best score any matcher could reach on top of this blocking |
| candidates / query | the downstream cost driver (mean, p95) |
| reduction ratio | 1 − candidate pairs / (queries × pool). Reported but **not informative here**: it is ≥ 0.99998 for every strategy because the pool is huge |

Validation protocol: deterministic folds `crc32(entity_id) % 5`; fold 0
(441,611 queries) is validation. Prototypes use random fold-0 samples (seed
13 unless stated) but always search the **full** train pools, so recall and
candidate volume are realistic. Sample sizes: 5k (17,264 true pairs), 20k
(69,276 true pairs).

## 2. Architecture (VERIFIED working)

```
S1 query (normalized)
  -> hashed features: n: name tokens | a: address tokens | b: house-number bigrams | p: phonetic name keys
  -> route by the query's own `country` string to that (source, country) partition index
     (partitions discovered from data; unseen country -> search all partitions)
  -> retrieval: sparse product with posting lists of features whose df <= 20k (top 1000)
  -> rescoring: full TF-IDF cosine on those 1000 (all features)
  -> two searches: literal (n,a,b) and phonetic-only (p,a,b); late fusion keeps each
     candidate's best score; top 100 per target source, stored with scores
```

Built once per source and reused by every experiment: normalized Parquet
stores (`work/store`), exact-key indexes and per-country TF-IDF inverted indexes
(`work/index`). No experiment scans the raw data per query.

## 3. Experiment record

### P2-B0 — baseline families (norm v1, 20k sample, seed 13)

| Strategy | Pair recall | Full coverage | Oracle F0.5 | Cands/query |
|---|---|---|---|---|
| exact normalized name | 28.76% | 4.26% | 0.5200 | 7.6 |
| order-insensitive name core (no legal forms) | 48.87% | 15.82% | 0.6949 | 12.1 |
| exact normalized address | 12.40% | 1.12% | 0.2697 | 0.5 |
| order-insensitive address | 18.64% | 2.00% | 0.3646 | 0.8 |
| all four exact keys (union) | 61.47% | 24.85% | 0.8098 | 16.2 |
| TF-IDF name only, 100/source | 57.62% | 37.04% | 0.6903 | 193.7 |
| TF-IDF address only, 100/source | 91.27% | 75.61% | 0.9675 | 199.8 |
| **TF-IDF name+address, 100/source** | **98.04%** | **94.19%** | **0.9927** | 200 |
| same, without full-cosine rescoring | 97.65% | 93.12% | 0.9914 | 200 |
| name TF-IDF ∪ address TF-IDF (separate searches) | 97.91% | 94.16% | 0.9917 | 391.2 |
| name+address TF-IDF ∪ all exact keys | 98.14% | 94.39% | 0.9931 | 211.6 |

Decisions: exact keys alone are far too weak (≤ 61%) and add only +0.10pp on
top of TF-IDF → not part of the blocker (they remain useful as *features* in
Phase 3). A joint name+address vector beats two separate searches at half the
volume. Rescoring helps (+0.39pp).

Segments (TF-IDF name+address, 100/source): **every one of the 69,276 true
pairs shares the query's country** → per-country partitioning costs zero
recall (VERIFIED). Weak segments: target names in Indic script 87.88% (4,949
pairs), empty target address 91.74% (3,027 pairs), India queries 96.37% vs
US 99.14%.

### P2-D1 — why are true pairs missed? (norm v1, 20k, 20/source; 2,200 misses)

| Cause | Share of misses |
|---|---|
| reachable but ranked below the 20th candidate | 87.45% (617 Indic-script, 376 empty-address) |
| shares only very common features (df > 20k) | 9.64% (140 Indic-script) |
| lost at the retrieval-depth cut (top 300) | 2.91% |
| shares no feature at all | 0 |

→ The bottleneck is ranking, not coverage. Examples show Source-2/3 names in
Devanagari / Kannada / Bengali / Tamil that are phonetic spellings of English
words ("एपेक्स सिस्टम्स प्रा. लि." = "Apex Systems Pvt Ltd"), while Source 1 is
Latin-only.

### P2-N2 — normalization v2 (Indic → Latin transliteration + phonetic key)

Deterministic in-code transliteration: one offset table for the nine
ISCII-aligned Brahmic Unicode blocks (no external data or model); phonetic key
= coarse consonant skeleton ("systems"/"sistams" → `stms`). Transliteration
alone: name+address TF-IDF 98.04% → 98.13% at 100/source (20k, seed 13).

### P2-E1 — does excluding legal-form words from phonetic keys fix the Latin-script loss? (5k)

Hypothesis: legal-form words (`ltd→lt`, `pvt→pft`, ...) pollute the phonetic
namespace. The bug was real (VERIFIED in code), but fixing it changed nothing:
+phonetic at weight 1 went 98.04% → 98.03%. **Hypothesis rejected** — those
keys already had near-zero IDF and were never used for retrieval. Fix kept
(harmless, principled, regression-tested).

Segment view (same 5k): phonetic keys help Indic-script targets (91.34% →
92.90%) and empty-address targets (88.31% → 92.38%) but hurt US queries
(99.08% → 98.45%). INFERRED cause: phonetic keys double-count name evidence
for every record, pushing same-named businesses at other addresses ahead.

### P2-E1b — adding phonetic evidence without double counting (5k)

| Variant (100/source) | Recall | Indic-script | US queries | Empty address |
|---|---|---|---|---|
| name+address (literal) | 98.23% | 91.34% | 99.08% | 88.31% |
| + phonetic weight 0.25 | 98.42% | 92.41% | 99.14% | 90.85% |
| + phonetic weight 0.5 | 98.34% | 92.82% | 98.94% | 91.99% |
| + phonetic weight 0.75 | 98.20% | 92.82% | 98.71% | 92.38% |
| phonetic-only (p,a,b) | 96.70% | — | — | — |
| **late fusion literal + phonetic-only** | **98.47%** | **94.88%** | 99.07% | 88.18% |

### P2-E2 — confirmation at 20k on two samples (seed 13 = baseline sample; seed 29, 855 queries overlap)

| Per source | literal, seed 13 / 29 | phon. 0.25, seed 13 / 29 | fusion, seed 13 / 29 |
|---|---|---|---|
| 10 | 95.46 / 95.38 | **95.89 / 95.80** | 95.57 / 95.54 |
| 20 | 96.77 / 96.81 | **97.07 / 97.09** | 96.99 / 97.04 |
| 30 | 97.22 / 97.28 | **97.50 / 97.55** | 97.47 / 97.53 |
| 50 | 97.69 / 97.75 | 97.86 / 97.95 | **97.96 / 98.01** |
| 100 | 98.13 / 98.20 | 98.24 / 98.33 | **98.39 / 98.45** |

All orderings replicate within ±0.1pp on the second sample. Fusion keeps its
Indic-script advantage (94.3–94.4% vs 90.8–90.9% literal). Known weakness of
fusion: empty-address targets at small budgets (20/source: 79.0–80.2% vs
85.1–86.1% for phonetic weight 0.25).

Score-relative truncation (keep candidates ≥ r × best score) was **rejected**:
at r = 0.3 it gives 97.64–97.65% with 83 candidates/query, no better than a
fixed K of similar volume, and with a p95 of 200 it is less predictable.

### P2-E3 / E3b — retrieval cap and depth (fusion, 5k)

| Setting | Recall @100 | @50 | @20 | Time (both searches / 5k) |
|---|---|---|---|---|
| cap 5k | 96.66% | 96.44% | 95.93% | 18 s |
| cap 20k, depth 300 | 98.47% | 98.03% | 97.14% | 53 s |
| **cap 20k, depth 1000** | **98.64%** | **98.12%** | **97.20%** | 60 s |
| cap 20k, depth 1000, ≥3 rarest features up to df 200k | 98.64% | 98.12% | 97.20% | 59 s |
| cap 50k, depth 1000 | 98.77% | 98.22% | 97.23% | 109 s |
| cap 100k, depth 300 | 98.80% | 98.24% | 97.23% | 229 s |

Decisions: depth 1000 is kept (+0.17pp for ~+13% time). The "≥3 rarest
features" rule had **no effect** (identical numbers): almost every query already
has ≥3 features under the cap — hypothesis rejected. Higher caps add
+0.13–0.16pp at 1.9–4.3× the cost; rejected given test-set scale on this
laptop (INFERRED ~20 h for cap 100k on 1.73M test queries).
Timing note: run-to-run wall-clock varies by up to ~50% for identical work
(other processes on the laptop); timings are indicative only.

### P2-E4 — operating budget

Candidates are stored as the **top 100 per source with scores**; the final
per-source budget is chosen in Phase 3 from measured feature/matcher cost.
The recall/volume curve for that decision is in P2-E2 / P2-E5.

### P2-S — smoke test of the checkpointed generator (2,000 train queries, seed 7)

Resume after deleting one chunk regenerated only that chunk; schema, score
ordering and uniqueness checks passed. Recall 98.70% @100, 98.08% @50,
97.02% @20 — consistent with the 5k estimates.

Profiling (one partition, one search): 53% sparse product, 22% rescoring, 13%
top-k selection; **multi-threading gives no speed-up** (scipy holds the GIL:
4.65 s with 4 threads vs 5.0 s single-threaded). Parallelism would need
separate processes sharing memory-mapped indexes — deferred to Phase 6, where
the 1.73M test queries make it worthwhile.

### P2-E5 — full validation fold (441,611 queries) — see §4

### P2-E6 — test split / open-set (France) check — see §5

## 4. Full-fold result (P2-E5)

_Pending — filled in from `work/candidates/train_fold0_fuse_nab_pab_r1000_k100/evaluation.json`._

## 5. Test split / France check (P2-E6)

_Pending._

## 6. Open items and risks

- Empty-address targets are the weakest segment under fusion at small budgets.
- Runtime on the laptop is single-core bound; test-set generation needs either
  ~5 h single-process (INFERRED) or process-level parallelism.
- France cannot be measured for recall (no labels); only distributional checks.
