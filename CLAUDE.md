# CLAUDE.md — Permanent Rules for the Business Entity Resolution Challenge

This file is the standing contract for how this project is worked on. It does not
describe the current state of the project (see `PROJECT_STATE.md` for that) — it
describes rules that hold for the entire duration of the challenge and should not need
to change as work progresses. If a rule here ever needs to change, that's a deliberate
edit the user should make or approve, not an incidental one.

## 1. Problem shape (do not re-derive these — they are fixed by the competition)

- Source 1 (`S1-*`) is the deduplicated reference set. For every S1 entity, the task is
  to find all matching records in Source 2 (`S2-*`) and/or Source 3 (`S3-*`).
- An S1 entity may have **zero, one, or many** matches, drawn from S2 and/or S3 in any
  combination.
- **Country is an open-set string feature, not a fixed category.** Training data
  contains only `US` and `India`; the test set additionally contains `France`. Never
  hardcode, filter, one-hot-encode, or branch logic on a closed `{US, India}` set
  anywhere in the pipeline. Every piece of country-aware logic (address parsing,
  postal-code patterns, blocking keys) must degrade gracefully to a generic/unknown
  country rather than fail or silently drop the record.
- Every test S1 entity — including French ones — must receive exactly one row in the
  output, even if no candidates or matches are found for it.

## 2. Evaluation metric — internalize this before writing any modeling code

- The competition metric is **F_0.5**, computed **per S1 entity**, then **macro-averaged**
  across all entities:
  `F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
- This weights precision **2× over recall**. False merges (matching two different
  businesses) cost more than missed matches. Every threshold, blocking width, and
  post-processing decision should be made with this asymmetry in mind — when genuinely
  unsure whether a candidate is a match, the metric rewards leaving it out.
- **Singletons matter and are graded**: an S1 entity with zero true matches scores
  **1.0** if predicted as empty, and **0.0** if any match is predicted for it. Do not
  treat "no match" as a degenerate or low-value case — it is worth full credit and is
  ~5.6% of the training entities.
- Build and use a standalone F_0.5 macro-scorer against the training ground truth for
  all internal validation. Do not substitute plain F1, micro-averaged precision/recall,
  or pairwise accuracy as a proxy — they behave differently from the actual metric and
  will mislead threshold/design decisions.

## 3. Output contract — hard requirements, not stylistic preferences

Two tab-separated files are required in `output/`:

- **`matching_results.tsv`** — the only file scored on the leaderboard. Columns:
  `source1_entity_id`, `matched_entity_ids` (comma-separated, no quoting, empty string
  for no matches).
- **`candidate_pairs.tsv`** — the exact candidate set produced by the last
  blocking/filtering stage, i.e. precisely what gets fed into the final matching model
  for inference. Not what an earlier, wider blocking pass produced if there are multiple
  filtering stages downstream of it. Columns: `source1_entity_id`, `candidate_entity_ids`.

Non-negotiable formatting rules (all checked by `utils/validate_submission.py` once
available):

- Exactly one row per S1 entity in the relevant dataset (train or test) — no missing
  entities, no duplicate `source1_entity_id` rows.
- No duplicate IDs within any single ID list.
- `matched_entity_ids` and `candidate_entity_ids` must only contain S2-/S3- IDs that
  exist in the same dataset split being scored — never an S1 ID, never an ID absent from
  that split.
- **`matching_results.tsv`'s matches must be a subset of `candidate_pairs.tsv`'s
  candidates for the same S1 entity, always.** If a final match did not appear as a
  candidate, that is a pipeline bug, not an acceptable outcome — fix the bug, don't
  patch the output file to hide it.
- Run `utils/validate_submission.py` locally before every leaderboard upload. A failed
  format check means the submission is not scored at all, regardless of match quality.

## 4. Fair play — hard constraints, not suggestions

- **No external data lookups of any kind**: no commercial entity-resolution APIs, no
  government business-registry lookups, no geocoding APIs, no internet-sourced data
  augmentation, no calling out to any network service at runtime to help resolve or
  normalize a record. Everything the pipeline uses must be derivable from
  `train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, and the corresponding test files
  alone, or from a pretrained model's own locally-installed weights (no live API calls
  to a hosted model either).
- **Final model license and size**: the model actually used for the final submission
  must be **MIT or Apache-2.0 licensed** and have **at most 8B parameters**. Check the
  license of any pretrained model or library component before adopting it into the
  final pipeline — do not assume a popular model qualifies without checking. If a model
  is used during exploration/prototyping but doesn't meet this bar, it cannot be part
  of what's actually submitted.
- Document every methodology and licensing choice in `Documentation_template.md` — the
  top submissions' code and docs are reviewed in detail, so the write-up needs to hold
  up to scrutiny, not just describe intentions vaguely.

## 5. Working process for this project

- **Do not implement a final end-to-end solution speculatively.** Work in the staged
  order: normalization/parsing → blocking/candidate generation → pair features → pair
  classifier → post-processing → format validation. Each stage should be checked
  (against the F_0.5 scorer and/or the format validator) before building heavily on top
  of it.
- **Blocking is a hard ceiling on recall.** Whatever a query's true match is, if
  blocking never retrieves it as a candidate, no downstream model can recover it. Treat
  recall@candidates as its own tracked metric, separate from final F_0.5, and invest in
  it before tuning the classifier.
- Prototype and iterate on a **sample** (tens of thousands of S1 queries), not the full
  2.2M, before scaling any experiment to the full training set. Full-scale runs are
  expensive on this machine (12 CPU cores, ~16 GB RAM, **no GPU** — see
  `PROJECT_STATE.md` §5 for current numbers, which should be re-checked periodically
  rather than assumed to be fixed).
- Do not speculate about facts that can be directly checked (row counts, column
  contents, library availability, hardware) — inspect and report the real numbers.
  `PROJECT_STATE.md` should reflect what was actually verified, not assumptions.
- Keep `PROJECT_STATE.md` current: update it at the end of any session that changes the
  repo, the data understanding, the environment, or the current best validation score.
  It should always answer "what's the state right now and what's next," without
  requiring someone to reread the whole conversation history.
- Log every modeling experiment in `experiments/experiment_log.csv` — one row per run,
  with enough detail (approach, data scope, key hyperparameters, validation F_0.5, and
  any recall@candidates figure) that a later reader can tell what was tried and why it
  did or didn't help, without re-running it.
- Keep the raw `.tsv` data files out of git (`.gitignore`) — they are large (1.27 GB
  combined) and are provided by the competition, not something to version.
- Before using any new third-party library, prefer ones already installed
  (`pandas`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`, `xgboost`, `catboost`,
  `faiss-cpu`, `sentence-transformers`, `transformers`, `torch`, `networkx`) — check
  `PROJECT_STATE.md` §5 for the current list before assuming something needs to be
  installed.
