# memory.md

> **Update this file at the end of every work session, before closing the AI chat.**
> This is the single source of truth the AI reads first in every new session — if it's
> stale, the AI will guess wrong about what's done. Paste (or have the AI read) this
> file at the start of every new session.

---

## Last Updated
`2026-09-26 12:05` — Phase 1 Preprocessing & Blocking Candidate Recall Measured

## Current Phase
Phase 1 — Preprocessing & Blocking *(update as you move through Phases.md)*

## Currently Being Worked On
- File: `src/blocking.py` & `evaluate_candidate_recall.py` (Completed)
- Status: Preprocessing normalization and 3-strategy blocking implemented and verified. Candidate recall measured on validation split at 94.23% (8,633/9,162 matches recovered across 2,500 validation S1 entities and 208,988 candidate pool records).

## Completed ✅
- [x] Repo scaffolding created
- [x] `requirements.txt` frozen
- [x] TSVs loaded, schema confirmed
- [ ] Noise-pattern table built (awaiting user review)
- [x] Stratified val split created + committed (`dataset/val_split_ids.txt`)
- [x] Normalization pass (`preprocessing.py`)
- [x] Blocking v1 (TF-IDF)
- [x] Blocking v2 (token/address key)
- [x] Blocking v3 (phonetic)
- [x] Candidate recall measured: 94.23%
- [ ] `scoring.py` built + hand-verified on toy example
- [ ] Feature set v1
- [ ] Baseline LogReg model
- [ ] First end-to-end run validated
- [ ] Feature set v2
- [ ] Model iteration (RF / XGBoost / LightGBM)
- [ ] Threshold sweep + chosen threshold: ___
- [ ] Error-analysis loop pass 1 / 2 / 3
- [ ] France sanity check
- [ ] Final test-set inference run
- [ ] `validate_submission.py` → PASS
- [ ] Leaderboard upload confirmed SCORED
- [ ] Packaging (README, requirements.txt, Documentation_template.md filled)
- [ ] Final zip built and checked against required structure

## Key Decisions Log
*(append, don't rewrite — this is a history)*
- `2026-09-26`: Replicated Architecture.md layout using junctions for dataset, utils, and src without deleting user's existing directories (Documentation, student_resource, code); hardlinked memory.md to root.
- `2026-09-26`: Phase 0 EDA completed.
  - TSV shapes verified (Train: GT=2,206,821, S1=2,206,821, S2=5,034,616, S3=5,285,603; Test: S1=1,732,544, S2=4,888,610, S3=5,142,398).
  - Open-set country verified: France appears ONLY in test set (~15%), Train is ~60% US, ~40% India.
  - Match-count distribution: 0 matches (singletons) = 5.58% (123,247), 1 match = 5.40%, 2 matches = 17.00%, 3+ matches = 72.01%.
  - Stratified 80/20 train/validation split created (441,365 val entities) and locked to `dataset/val_split_ids.txt` (never regenerated).
  - Sampled and printed 40 real matched pairs side-by-side highlighting transliteration (Devanagari, Telugu), URLs/domains, abbreviations, typos, and missing address components.
- `2026-09-26`: Phase 1 Preprocessing & Blocking completed:
  - `src/preprocessing.py`: Implemented Unicode-aware normalization preserving Indic scripts & accents (Hindi/Telugu/French), legal-suffix canonicalization, and thoroughfare standardizations while avoiding colliding linguistic stop words.
  - `src/blocking.py`: Implemented 3 strategies (TF-IDF char (2,4) n-grams top-20, sorted significant address tokens + numeric keys, phonetic metaphone on name tokens) partitioned by country.
  - Candidate recall measured on validation set: **94.23%** (8,633/9,162 true matches recovered) with an average of 89.1 candidates per S1 entity.

## Known Issues / TODO
- *(nothing yet)*

## Next Immediate Step
Build `src/scoring.py` (local macro F0.5 evaluation metric) and hand-verify on 2-3 toy examples before feature engineering.

---

## Reference Numbers (fill in as you get them — don't let the AI guess these)
- Candidate recall (blocking): 94.23% (TF-IDF top-20: 81.86%, Address: 75.69%, Phonetic: 49.57%; avg candidates/entity: 89.1)
- Best validation F0.5: ___ (model: ___, threshold: ___)
- Singleton rate in train ground truth: 5.58%
- Public leaderboard F0.5: ___
