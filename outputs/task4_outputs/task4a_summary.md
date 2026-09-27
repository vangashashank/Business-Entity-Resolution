# Task 4A - Baseline Pairwise Matching Models

## Scope

Exactly two model families were trained on the existing Task 3 entity split: a scaled, class-balanced regularized logistic regression and one conservative, class-weighted LightGBM classifier. Candidate generation, pair features, and the split were not changed. No threshold tuning, calibration, test inference, or submission work was performed.

## Data

| Split | S1 entities | Pairs | Positives | Negatives | Positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 8,000 | 1,999,962 | 25,798 | 1,974,164 | 1.289925% |
| Validation | 2,000 | 499,985 | 6,454 | 493,531 | 1.290839% |

All 66 audited features were used. The model matrices contain no identifier, target, split, match-group, NaN, or infinite input. Validation has 239 additional positive links missed by candidate generation; no classifier can rank those links.

## Model comparison

| Metric | Logistic Regression | LightGBM |
| --- | ---: | ---: |
| PR-AUC | 0.929393 | 0.984601 |
| ROC-AUC | 0.998447 | 0.999529 |
| Log loss | 0.048406 | 0.007394 |
| Recall@1, retrieved | 0.288968 | 0.303068 |
| Recall@1, end-to-end | 0.278649 | 0.292246 |
| Recall@3, retrieved | 0.655717 | 0.677099 |
| Recall@3, end-to-end | 0.632302 | 0.652921 |
| Recall@5, retrieved | 0.866439 | 0.891695 |
| Recall@5, end-to-end | 0.835500 | 0.859854 |
| Recall@10, retrieved | 0.988689 | 0.996281 |
| Recall@10, end-to-end | 0.953384 | 0.960705 |
| Recall@20, retrieved | 0.996746 | 0.998915 |
| Recall@20, end-to-end | 0.961153 | 0.963245 |
| Training runtime (s) | 2.374 | 105.694 |
| Inference runtime (s) | 0.178 | 3.420 |

LightGBM changed PR-AUC by +0.055208 versus logistic regression. Logistic convergence status: **converged** after 27 iterations. LightGBM best iteration: **1489**.

## Probability distributions

| Model | Label subset | Mean | Median | P10 | P90 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | positive | 0.975751 | 0.999854 | 0.967457 | 0.999994 | 0.999998 |
| logistic_regression | negative | 0.023687 | 0.000286 | 0.000007 | 0.023766 | 0.715710 |
| lightgbm | positive | 0.966923 | 0.999945 | 0.981647 | 0.999999 | 1.000000 |
| lightgbm | negative | 0.002960 | 0.000000 | 0.000000 | 0.000034 | 0.020703 |

The linear baseline leaves a substantially heavier high-probability negative tail. LightGBM improves separation and log loss, but its hardest false matches can still receive probabilities close to one. These scores are ranking outputs only; no threshold was selected.

## Source analysis

LightGBM ranking uses each candidate's global rank within its S1 candidate set.

| Source | Retrieved positives | Retrieval misses | PR-AUC | Recall@5 | Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 | 3129 | 85 | 0.983980 | 0.896772 | 0.998402 |
| S3 | 3325 | 154 | 0.985166 | 0.886917 | 0.999398 |

## Match-group analysis

| Match group | Retrieved positives | Retrieval misses | PR-AUC | Recall@5 | Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 481 | 19 | 0.961453 | 0.987526 | 1.000000 |
| 2 | 967 | 33 | 0.975864 | 0.989659 | 0.995863 |
| 3-5 | 1900 | 58 | 0.985492 | 0.981053 | 0.999474 |
| 6+ | 3106 | 129 | 0.992803 | 0.791693 | 0.999356 |

## LightGBM feature importance

Gain importance is descriptive and not causal.

| Rank | Feature | Family | Gain share | Split count |
| --- | ---: | ---: | ---: | ---: |
| 1 | `retrieval_best_rank` | retrieval_evidence | 72.97% | 1017 |
| 2 | `name_address_similarity_product` | baseline_name_similarity | 6.82% | 730 |
| 3 | `address_token_set` | address_similarity | 4.59% | 1016 |
| 4 | `address_jaro_winkler` | address_similarity | 2.83% | 1054 |
| 5 | `independent_strong_signal_count` | interpretable_interaction | 2.12% | 331 |
| 6 | `retrieval_rrf_score_top100` | retrieval_evidence | 0.82% | 1339 |
| 7 | `transliteration_levenshtein_gain` | transliterated_name_similarity | 0.80% | 1041 |
| 8 | `suffix_name_token_set` | legal_suffix_name_similarity | 0.79% | 1058 |
| 9 | `transliterated_name_levenshtein` | transliterated_name_similarity | 0.79% | 1638 |
| 10 | `address_number_jaccard` | address_number_structure | 0.77% | 564 |
| 11 | `name_char3_jaccard` | baseline_name_similarity | 0.68% | 1252 |
| 12 | `address_conflicting_number_count` | address_number_structure | 0.66% | 669 |
| 13 | `suffix_name_levenshtein` | legal_suffix_name_similarity | 0.48% | 1198 |
| 14 | `candidate_address_missing` | missingness | 0.41% | 8 |
| 15 | `name_length_ratio` | baseline_name_similarity | 0.39% | 2075 |
| 16 | `name_jaro_winkler` | baseline_name_similarity | 0.36% | 1677 |
| 17 | `name_first_token_exact` | baseline_name_similarity | 0.36% | 250 |
| 18 | `name_token_jaccard` | baseline_name_similarity | 0.31% | 1144 |
| 19 | `address_token_jaccard` | address_similarity | 0.25% | 1176 |
| 20 | `name_token_set` | baseline_name_similarity | 0.19% | 777 |

6 of the top 20 features belong to a Task 3.5 high-correlation pair. All correlated-pair features together account for 2.91% of LightGBM gain. They were retained as required; feature ablation remains future work.

The learned importance broadly agrees with Task 3 descriptive evidence: frozen retrieval rank is dominant, combined name-address and address similarities are strong, and numeric, transliteration, and legal-suffix features add smaller but measurable signal. The 72.97% gain share on `retrieval_best_rank` also means Task 4B should explicitly test robustness to retrieval-position dependence.

## Hard cases

The 20 hardest validation positives contain these overlapping patterns: `{"conflicting_address_numbers": 9, "cross_script_proxy": 9, "high_address_similarity": 6, "high_name_similarity": 7, "legal_suffix_gain_positive": 4, "match_group_6_plus": 11, "missing_candidate_address": 6, "multiple_retrieval_signals": 0, "s3": 6}`.

The 20 hardest validation negatives contain these overlapping patterns: `{"conflicting_address_numbers": 5, "cross_script_proxy": 4, "high_address_similarity": 15, "high_name_similarity": 10, "legal_suffix_gain_positive": 5, "match_group_6_plus": 3, "missing_candidate_address": 0, "multiple_retrieval_signals": 10, "s3": 7}`.

The detailed samples include raw training names/addresses, probabilities, within-entity ranks, text similarities, address-number evidence, and frozen retrieval evidence.

## Sanity and readiness

All 12/12 Task 4A sanity checks passed. Validation labels were used only for LightGBM early stopping and evaluation, never for preprocessing, class weighting, or gradient updates. The candidate and pair-feature input hashes were unchanged after the run.

The results support proceeding to Task 4B decision-rule analysis.

Total runtime was 2.26 minutes with approximately 3.25 GiB peak process RSS.
