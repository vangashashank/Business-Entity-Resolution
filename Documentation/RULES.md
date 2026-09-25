# RULES.md

## 1. Libraries — Use
- `pandas`, `numpy` — data handling
- `rapidfuzz` — string similarity (preferred over `python-Levenshtein`: faster, MIT
  licensed)
- `jellyfish` — phonetic encoding (metaphone/soundex)
- `scikit-learn` — TF-IDF, NearestNeighbors, LogisticRegression, RandomForest,
  train/test utilities
- `xgboost` or `lightgbm` — gradient boosted matcher (pick one, don't maintain both)
- `sentence-transformers` — **only if** blocking recall plateaus below ~95% with
  classical methods; must use an MIT/Apache-2.0 model, ≤8B params (e.g.
  `all-MiniLM-L6-v2`, `bge-small-en-v1.5` — both are far under the limit and safe)
- `matplotlib`, `seaborn` — EDA and error-analysis plots
- `tqdm` — progress bars for anything iterating over the full candidate set

## 2. Libraries / Approaches — Avoid
- **Any external API, database, or geocoding service** (Google Maps API, OpenCorporates,
  business registry lookups, etc.) — explicitly banned by the challenge rules, instant
  disqualification if detected.
- **Any closed-license or >8B-parameter pretrained model** — check license text before
  pulling any model off Hugging Face, not just the model card headline.
- Hardcoding `country in ["US", "India"]` anywhere in the pipeline — France (and any
  future unseen country) must flow through the same code path.
- `pd.read_csv(...)` without `sep="\t"` — this challenge's files are tab-separated;
  a silent default-comma read produces a single garbage column.
- Over-fitting the threshold to the validation split by eye — always sweep and pick by
  the scorer, not by "this looks about right."
- Keeping a change in the pipeline "because it seemed reasonable" if it didn't move
  validation F0.5 in the error-analysis loop — revert it.

## 3. Error Handling Conventions
- Every module (`preprocessing.py`, `blocking.py`, etc.) validates its input schema on
  load (expected columns present, no unexpected nulls in `entity_id`) and fails loudly
  with a clear message — never silently drops rows.
- `scoring.py` is built and hand-verified against a 2–3-entity toy example **before**
  it's used to judge anything else. A silently wrong scorer invalidates every decision
  made downstream of it.
- After **every** change that touches candidate generation, feature extraction, model,
  threshold, or output formatting: re-run `utils/validate_submission.py` locally before
  moving on. Don't batch up changes and validate once at the end.
- `main` (or your single working branch) always has a working, validator-passing
  pipeline end-to-end. If mid-change and running low on time/focus, don't leave it
  broken — commit the last-known-good state first.
- Empty candidate/match lists are a valid, expected output for singletons — never
  treat an empty list as an error condition in downstream code.

## 4. Boundaries for the AI Pair-Programmer
- The AI (Claude Code / whatever vibecoding tool) may write and edit code inside
  `src/`, `notebooks/`, `colab_notebooks/` freely.
- The AI must **never** fabricate or estimate a validation F0.5/precision/recall
  number — it only reports numbers actually produced by running `scoring.py`.
- The AI must **not** silently change the folder structure, output file schema, or
  column names defined in `Architecture.md` / the challenge README without flagging
  it explicitly first.
- The AI must **not** add an external API call, geocoding call, or any network
  dependency beyond installing packages — this is a hard fair-play violation.
- Before pulling in any pretrained model, the AI must state its license and parameter
  count and confirm it's MIT/Apache-2.0 and ≤8B params.
- After each meaningful step, the AI updates `memory.md` (completed items, file
  currently being worked on, next step) — don't wait until the end of a session.
- Large refactors (e.g. switching blocking strategy, swapping the model family) get
  called out explicitly before being made, not folded silently into an unrelated fix.
- If a run of `validate_submission.py` fails, that's fixed before any other work
  continues — it's a hard blocker, not a "note for later."
