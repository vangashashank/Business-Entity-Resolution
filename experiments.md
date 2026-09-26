# experiments.md

Log every change and experiment here. Track metrics rigorously to guide pipeline decisions.

| change | candidate_recall | val_F0.5 | precision | recall | notes | date |
|---|---:|---:|---:|---:|---|---|
| Setup / EDA | - | - | - | - | Repository scaffolding and stratified split setup. | 2026-09-26 |
| Logistic Regression (feature set v1) | 95.4174% | 0.821195 | 0.813734 | 0.918306 | Similarity metric changed from fuzz.ratio to normalized Levenshtein; threshold=0.5; 2,500 held-out S1 entities, including 152 singletons. | 2026-09-26 |
| Logistic Regression (feature set v2) | 95.4174% | 0.851902 | 0.846726 | 0.924581 | Similarity metric changed from fuzz.ratio to normalized Levenshtein; threshold=0.5; 2,500 held-out S1 entities, including 152 singletons. | 2026-09-26 |
| Random Forest (feature set v2) | 95.4174% | 0.947164 | 0.959219 | 0.931150 | Similarity metric changed from fuzz.ratio to normalized Levenshtein; threshold=0.5; 2,500 held-out S1 entities, including 152 singletons. | 2026-09-26 |
| XGBoost (feature set v2) | 95.4174% | 0.945500 | 0.955095 | 0.936193 | Similarity metric changed from fuzz.ratio to normalized Levenshtein; threshold=0.5; 2,500 held-out S1 entities, including 152 singletons. | 2026-09-26 |
| Logistic Regression v1 threshold sweep (0.95 grid maximum) | 95.4174% | 0.922110 | 0.939610 | 0.899198 | 2,500 S1 validation sample including 152 singletons; same-model threshold 0.50 baseline F0.5=0.821195, precision=0.813734, recall=0.918306. LR-only sweep; superseded for selected model by RF v2 sweep. | 2026-09-26 |
| Random Forest v2 threshold sweep (0.75 maximum) | 95.4174% | 0.951503 | 0.971190 | 0.913899 | 2,500 S1 validation sample including 152 singletons; 22 thresholds from 0.30–0.99. Threshold 0.50 baseline F0.5=0.947164, precision=0.959219, recall=0.931150; near-plateau from 0.65–0.75. Inference default remains 0.5 pending Phase 4 threshold selection. | 2026-09-26 |
| Random Forest v2 threshold locked at 0.70 | 95.4174% | 0.951156 | 0.968562 | 0.919385 | Chosen from 0.65–0.75 plateau: within 0.000347 F0.5 of 0.75 maximum while retaining more recall. Phase 4 baseline. | 2026-09-26 |
| Phase 4 error-analysis baseline, canonical candidate_pairs_val.tsv | 95.4174% | 0.951156 | 0.968562 | 0.919385 | RF v2, threshold 0.70, 2,500 S1s including 152 singletons; reused saved canonical candidates (no blocking rerun). 173 FP, 297 model FN, 398 blocking-miss pairs. | 2026-09-26 |
| Archived matched-only candidate sample (noncanonical) | 95.2958% | 0.951749 | 0.971997 | 0.915962 | RF v2, threshold 0.70, 2,500 matched S1s and zero singletons; 9,162 truth edges. Archived at `output/archive/candidate_pairs_val_matched_only_legacy.tsv`; do not use for validation comparisons. | 2026-09-26 |
| RF v2 feature trial: numeric overlap despite weak name similarity | 95.4174% | 0.950745 | 0.968385 | 0.918572 | Rejected and reverted: F0.5 change -0.000411 versus baseline 0.951156. Same canonical 2,500-S1 validation set (152 singletons), seed-42 training sample, and threshold 0.70. | 2026-09-27 |

## Comparison against the pre-correction run

The historical rows below used RapidFuzz `fuzz.ratio` (Indel similarity) and a matched-only validation sample. Corrected rows use normalized Levenshtein and include 152 singletons in the 2,500-entity validation sample. The matched-only corrected metrics are shown as an intermediate reference; that run used the prior 2,500 matched-entity validation sample. Consequently, the historical-to-final delta combines the feature correction and validation-sample/singleton scope change; it does not isolate a singleton-only effect.

| model | old F0.5 (pre-correction) | corrected F0.5 (matched-only) | corrected F0.5 (including singletons) | old-to-final delta |
|---|---:|---:|---:|---:|
| Logistic Regression v1 | 0.863669 | 0.833041 | 0.821195 | -0.042474 |
| Logistic Regression v2 | 0.881332 | 0.864555 | 0.851902 | -0.029430 |
| Random Forest v2 | 0.950106 | 0.949578 | 0.947164 | -0.002942 |
| XGBoost v2 | 0.949921 | 0.9493 | 0.945500 | -0.004421 |

All four corrected comparison scores used the same fixed threshold, 0.5. Threshold sweeping remains a separate Phase 3 step.
