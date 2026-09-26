# memory.md

> **Update this file at the end of every work session, before closing the AI chat.**
> This is the single source of truth the AI reads first in every new session — if it's
> stale, the AI will guess wrong about what's done. Paste (or have the AI read) this
> file at the start of every new session.

---

## Last Updated
`2026-09-27` - Phase 4 Error Analysis Complete for Now

## Current Phase
Phase 4 - Integration / Error-Analysis Loop (complete for now)

## Currently Being Worked On
- File: Phase 4 validation error analysis
- Status: Phase 4 is complete for now. The locked baseline is Random Forest v2 at threshold 0.70 (macro F0.5=0.951156) on the canonical singleton-inclusive validation candidate set.

## Canonical Validation Candidate File
- `output/candidate_pairs_val.tsv` is the one canonical validation candidate file: the 2,500-S1 seed-42 sample from locked `student_resource/dataset/val_split_ids.txt`, including 152 singletons; 8,685 true-match pairs; 95.4174% candidate recall.
- The former 9,162-pair, matched-only file was **renamed/moved, not deleted**, to `output/archive/candidate_pairs_val_matched_only_legacy.tsv`. It is historical and must not be used for current validation comparisons.

## Completed ✅
- [x] Repo scaffolding created
- [x] `requirements.txt` frozen
- [x] TSVs loaded, schema confirmed
- [ ] Noise-pattern table built (awaiting user review)
- [x] Stratified val split created + committed (`student_resource/dataset/val_split_ids.txt`)
- [x] Normalization pass (`src/preprocessing.py`)
- [x] Blocking v1 (TF-IDF)
- [x] Blocking v2 (token/address key)
- [x] Blocking v3 (phonetic)
- [x] Candidate recall measured: 94.23%
- [x] TF-IDF top-k tradeoff: k=30 measured at 94.31% recall and 96.9 candidates/entity (2,500 S1 sample)
- [x] TF-IDF top-k tradeoff: k=40 measured at 94.30% recall and 105.0 candidates/entity (2,500 S1 sample)
- [x] Candidate union reserves up to 60 candidates per strategy before round-robin overflow; k=20 recall improved from 94.23% (8,633/9,162) to 95.30% (8,731/9,162), with 89.1 candidates/entity.
- [x] Rechecked three prior tie-rank pairs after union fix: all remain missing; their S1 output rows contain 43 and 59 candidates, below the 200 cap.
- [x] `scoring.py` macro F0.5 scorer implemented and hand-checked on 3 singleton/match edge cases (macro result 0.611111).
- [x] Re-ran visible toy check for `src/scoring.py`: hand and scorer values match for all 3 entities; macro F0.5=0.611111.
- [x] Feature set v2 implemented: first-token match, numeric-address token match, suffix-stripped normalized name match, and address component-count difference.
- [x] Feature/model comparison including singletons (2,500 validation S1; 152 singletons): LR v1 F0.5=0.821195, LR v2=0.851902, RF v2=0.947164, XGBoost v2=0.945500; candidate recall=95.4174%; threshold=0.5 for all.
- [x] `experiments.md` comparison includes old pre-correction scores and corrected matched-only reference scores, with explicit validation-scope caveat.
- [x] `src/threshold.py` sweeps 0.30–0.95 in 0.05 steps via `src/scoring.py`, plots using `Documentation/design.md` palette, and marks the grid maximum; chart at `output/threshold_sweep.png`.
- [x] Baseline LR threshold sweep measured 0.5 F0.5=0.821195 / P=0.813734 / R=0.918306 versus grid maximum 0.95 F0.5=0.922110 / P=0.939610 / R=0.899198 (same 2,500 S1 validation sample, 152 singletons). Maximum is at the grid boundary; production default remains 0.5 pending broader sweep.
- [x] Selected RF v2 threshold sweep: 22 points from 0.30–0.90 by 0.05 then 0.91–0.99 by 0.01. Best=0.75, F0.5=0.951503 / P=0.971190 / R=0.913899; 0.50 baseline F0.5=0.947164 / P=0.959219 / R=0.931150. Shallow peak/plateau across 0.65–0.75, then F0.5 declines; default stays 0.5 pending Phase 4 review.
- [x] Production threshold locked at 0.70 on RF v2: F0.5=0.951156, precision=0.968562, recall=0.919385. It retains more recall than the 0.75 grid maximum and serves as the Phase 4 error-analysis baseline.
- [x] Added `src/error_analysis.py` to reuse an existing candidate TSV, calculate v2 features, score with RF v2 at threshold 0.70, compare against validation ground truth, and export FP/FN examples. It does not import or invoke blocking code. Source records are loaded in chunks for the requested IDs.
- [x] Error-analysis sample scored on the canonical singleton-inclusive `output/candidate_pairs_val.tsv`: candidate recall 95.4174%, macro F0.5=0.951156, precision=0.968562, recall=0.919385; 173 FP, 297 model FNs, and 398 blocking-miss pairs across 2,500 S1s.
- [x] RF v2 feature ablation on canonical candidates: tested `numeric_token_match AND name_tfidf_cosine < 0.3` at threshold 0.70 with the same seed-42 training sample. Trial F0.5=0.950745, precision=0.968385, recall=0.918572 versus baseline 0.951156 / 0.968562 / 0.919385. Reverted the feature because F0.5 decreased; removed the trial script and model artifact.
- [x] Confirmed the archived matched-only sample is a different S1 sample: it shares only 15 of 2,500 S1 IDs with the canonical set and has no singletons. Both samples' IDs are subsets of locked `val_split_ids.txt`, but only the singleton-inclusive set matches the Phase 2/3 evaluation cohort.
- [x] `src/inference.py` now defaults to `artifacts/random_forest_v2.joblib` and threshold 0.70; 200-S1 smoke inference generated 22,392 candidates and 27 non-empty match rows, and the smoke fixture validator passed with `--check-ids`.
- [x] Smoke output validator PASS after model/feature work.
- [x] Re-ran corrected baseline inference smoke on 200 test S1s spanning US/India/France: 22,392 candidate pairs; 200 matching rows (104 empty, 96 non-empty); validator PASS. Full test-set run remains pending.
- [x] `validate_submission.py --check-ids` PASS was against a 200-entity smoke fixture only; this does NOT mean the pipeline is submission-ready. Full validation must run with `--check-ids` against `student_resource/dataset/test/` and a `matching_results.tsv` covering all 1,732,544 test S1 entities. Do this in Phase 6 after threshold tuning and the full error-analysis loop.
- [x] `scoring.py` built + hand-verified on toy example
- [ ] Feature set v1
- [ ] Baseline LogReg model
- [ ] First end-to-end run validated
- [ ] Feature set v2
- [ ] Model iteration (RF / XGBoost / LightGBM)
- [x] RF v2 threshold sweep through 0.99 complete; threshold 0.70 locked as the Phase 4 baseline.
- [x] Phase 4 error-analysis loop completed for now; locked baseline is RF v2 at threshold 0.70, macro F0.5=0.951156. The tested numeric/name disagreement feature was rejected and reverted.
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
  - Stratified 80/20 train/validation split created (441,365 val entities) and locked to `student_resource/dataset/val_split_ids.txt` (never regenerated).
  - Sampled and printed 40 real matched pairs side-by-side highlighting transliteration (Devanagari, Telugu), URLs/domains, abbreviations, typos, and missing address components.
- `2026-09-26`: Phase 1 Preprocessing & Blocking completed:
  - `src/preprocessing.py`: Implemented Unicode-aware normalization preserving Indic scripts & accents (Hindi/Telugu/French), legal-suffix canonicalization, and thoroughfare standardizations while avoiding colliding linguistic stop words.
  - `src/blocking.py`: Implemented 3 strategies (TF-IDF char (2,4) n-grams top-20, sorted significant address tokens + numeric keys, phonetic metaphone on name tokens) partitioned by country.
  - Candidate recall measured on validation set: **94.23%** (8,633/9,162 true matches recovered) with an average of 89.1 candidates per S1 entity.
- `2026-09-26`: Consolidated repository layout to single source of truth:
  - Removed top-level `dataset` and `utils` duplicate junctions; `student_resource/` is now the sole canonical location for raw data and `validate_submission.py`.
  - Replaced `src` junction with a real, top-level working directory containing `src/preprocessing.py` and `src/blocking.py`.
  - Removed premature `code/` packaging directory (to be created cleanly in Phase 7).

## Known Issues / TODO
- About 40 blocking-miss false negatives remain unaddressed: their true matches were never generated as candidates, so feature/model changes cannot recover them. Revisiting them requires a `src/blocking.py` change.
- Four transliteration-related false negatives were identified in error analysis and remain unaddressed.
- `numeric_token_overlap_despite_weak_name_similarity` was tested and rejected (macro F0.5 decreased by about 0.0004). Do not re-attempt this exact feature.

## Next Immediate Step
Proceed to the Phase 5 France/open-set sanity check.

---

## Reference Numbers (fill in as you get them — don't let the AI guess these)
- Candidate recall (blocking): 94.23% (TF-IDF top-20: 81.86%, Address: 75.69%, Phonetic: 49.57%; avg candidates/entity: 89.1)
- Top-k sweep (2,500 S1 sample): k=20: 94.23% / 89.1 candidates per entity; k=30: 94.31% / 96.9; k=40: 94.30% / 105.0. Default remains 20 pending user choice.
- Best validation F0.5: ___ (model: ___, threshold: ___)
- Singleton rate in train ground truth: 5.58%
- Public leaderboard F0.5: ___
