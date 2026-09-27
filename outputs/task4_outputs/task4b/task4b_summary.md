# Task 4B - Rank Robustness and Entity-Level Decision Policy

## Scope

Task 4B reused the frozen Task 4A LightGBM validation predictions, trained three controlled LightGBM ablations at the same fixed 1,489 iterations, inspected ranking/classification errors, and selected one validation-only entity decision policy. No candidate, pair feature, entity split, Task 4A artifact, test record, or submission was changed or generated.

## Retrieval-rank dependence

| Variant | Features | PR-AUC | Log loss | Retrieved Recall@5 | End-to-end Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_66_existing | 66 | 0.984601 | 0.007394 | 89.1695% | 96.3245% |
| remove_retrieval_best_rank | 65 | 0.984466 | 0.007438 | 89.2160% | 96.3395% |
| remove_direct_position_features | 52 | 0.980114 | 0.009459 | 89.0920% | 96.3544% |
| remove_all_retrieval_evidence | 44 | 0.977563 | 0.011620 | 88.9836% | 96.3395% |

Removing all retrieval evidence changes PR-AUC by -0.007037 relative to the full model. The detailed artifact includes ROC-AUC, all Recall@K values, and S2/S3 and match-group breakdowns for every variant. This quantifies rank dependence directly rather than inferring it from gain importance.

## Selected decision policy

- Rule: **global_p_ge_0.95**
- Family: `global_probability_threshold`
- Parameters: `{"probability_threshold": 0.95}`
- Predicted links: **6,244**
- Average predicted matches per S1: **3.1220**
- Entities with zero predictions: **54**
- Precision: **95.7880%**
- Retrieved-only recall / F1: **92.6712% / 94.2038%**
- End-to-end recall / F1: **89.3620% / 92.4635%**
- Retrieved-only complete entity rate: **71.3500%**
- End-to-end complete entity rate: **65.2000%**

There are **239** unreachable validation positives. They reduce end-to-end recall and complete-set recovery but are not classifier failures.

Selection method: Among rules within 1.0 percentage point of the best end-to-end complete entity rate, maximize end-to-end link F1, then prefer fewer false positives and lower rule complexity. This avoids forcing low-score top candidates for a marginal complete-set gain.

The maximum-complete comparator `relative_gap_le_0.05` reaches 66.1500% complete recovery versus 65.2000% for the selected rule, but adds 534 false positives and lowers end-to-end F1 from 92.4635% to 89.0370%. The source/group CSVs retain both policies for direct inspection.

## Source behavior

| Source | Precision | Retrieved recall | End-to-end recall | End-to-end F1 |
| --- | ---: | ---: | ---: | ---: |
| S2 | 95.3801% | 93.0329% | 90.5725% | 92.9141% |
| S3 | 96.1779% | 92.3308% | 88.2437% | 92.0402% |

## Match-group behavior

| Match group | Precision | Retrieved recall | End-to-end recall | End-to-end complete rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 83.9319% | 92.3077% | 88.8000% | 76.0000% |
| 2 | 92.8793% | 93.0714% | 90.0000% | 71.8000% |
| 3-5 | 97.0653% | 92.2632% | 89.5301% | 63.4000% |
| 6+ | 98.0952% | 92.8525% | 89.1499% | 49.6000% |

## Dominant errors

| Error type | Count | Denominator | Rate |
| --- | ---: | ---: | ---: |
| candidate_generation_miss | 239 | 6693 | 3.5709% |
| retrieved_positive_rank_gt_5 | 699 | 6454 | 10.8305% |
| retrieved_positive_rank_gt_10 | 24 | 6454 | 0.3719% |
| retrieved_positive_rank_gt_20 | 7 | 6454 | 0.1085% |
| retrieved_positive_probability_lt_0.1 | 86 | 6454 | 1.3325% |
| retrieved_positive_probability_lt_0.5 | 184 | 6454 | 2.8509% |
| negative_probability_ge_0.5 | 1144 | 493531 | 0.2318% |
| negative_probability_ge_0.9 | 399 | 493531 | 0.0808% |
| negative_probability_ge_0.99 | 72 | 493531 | 0.0146% |
| selected_policy_false_positive | 263 | 493531 | 0.0533% |
| selected_policy_false_negative_retrieved | 473 | 6454 | 7.3288% |

The detailed error artifacts separate candidate-generation misses, poorly ranked retrieved positives, high-scoring negatives, feature-pattern risks, and errors introduced by the selected policy.

## Score reliability

LightGBM expected calibration error is **0.002507** and Brier score is **0.002032** on validation. No calibration model was added. The selected policy treats scores as validation-ranked evidence rather than claiming literal probability calibration.

## Integrity and readiness

All **15/15** integrity checks passed. The selected decision rule uses only the LightGBM pair score. It does not use match-count group, labels, identifiers as predictors, or test information.

Task 4 can be frozen on validation and the project is ready to plan full-scale inference.

Runtime was **3.98 minutes** with approximately **2.63 GiB** peak process RSS.
