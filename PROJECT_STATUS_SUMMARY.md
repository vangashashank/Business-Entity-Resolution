# Amazon ML Challenge 2026 - Project Status

Status date: 2026-09-26

## Current status

Tasks 1, 2, 2.5, 3, the Task 3.5 pre-model audit, Task 4A baseline modeling, Task 4B validation decision analysis, Task 5 inference architecture validation, and Task 6 production-state persistence are complete on the training data. The candidate generator remains frozen at 250 candidates, the 66-feature LightGBM remains the primary pairwise model, and the decision policy remains a global LightGBM score threshold of 0.95. Task 6 persisted and hash-verified the full retrieval bundle and a selective disk-backed candidate text store, then reproduced the Task 5 smoke pipeline from reloaded state. Full-scale inference, test prediction, and submission generation have not started.

## Project objective

For every Source 1 business entity, retrieve all Source 2 and Source 3 records representing the same underlying business. This is a multi-match entity-resolution problem.

Training data used:

| Dataset | Records | Columns |
| --- | ---: | --- |
| Source 1 | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` |
| Source 2 | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` |
| Source 3 | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` |
| Ground truth | 2,206,821 | `source1_entity_id`, `matched_entity_ids` |

## Work completed before Codex

The pre-existing Colab notebook export performed initial exploratory work:

- Loaded S1, S2, S3, and ground truth from `/content`.
- Printed shapes, sample rows, schema information, null counts, and country counts.
- Added a ground-truth match count and inspected its distribution.
- Counted S2 and S3 IDs in the ground truth.
- Sampled 25 S1 entities using `random_state=42` across 1-match, 2-match, 3-5-match, and 6+-match groups.
- Retrieved one sampled S1 record and its corresponding S2/S3 records for manual inspection.

This established the dataset structure and the multi-match nature of the problem. It did not yet produce a complete pairwise inspection table, systematic normalized agreement statistics, full-ground-truth country validation, scalable candidate retrieval, Recall@K measurements, or retrieval failure analysis.

## Work completed by Codex

### Task 1: Positive ground-truth inspection

A reproducible, training-only Task 1 analysis was created without overwriting the original notebook.

Completed work:

- Confirmed the actual schema and full training shapes.
- Sampled 25 S1 entities across the required match-count groups.
- Retrieved every positive S2/S3 record for those entities.
- Built a pairwise inspection table containing 86 positive links.
- Added conservative normalized exact-match indicators for name, address, and country.
- Produced source-level agreement statistics and pattern summaries.
- Confirmed zero lookup issues and zero flagged country inconsistencies in the sample.

Task 1 findings:

| Measure | Positive-link agreement |
| --- | ---: |
| Exact normalized name | 18.60% |
| Exact normalized address | 10.47% |
| Same normalized country | 100.00% |

S2 exact agreement was higher than S3 for both names and addresses. The sample showed spelling differences, legal-suffix changes, punctuation and formatting differences, reordered or altered addresses, and missing addresses. Exact name or address is therefore unsuitable as a hard blocking rule.

### Task 2: Candidate generation and Recall@K

A memory-bounded candidate-generation pipeline was implemented and run against all 10,320,219 S2/S3 candidate records.

Completed work:

- Validated country consistency against all 7,638,365 positive training links.
- Created a reproducible 10,000-entity evaluation set with 2,500 entities in each match-count group.
- Evaluated 33,285 positive links in that sample.
- Implemented conservative Unicode-aware name, address, and country normalization.
- Measured exact-name, exact-address, and exact-union baselines.
- Implemented character 3-5 gram TF-IDF retrieval with sparse random projection and compressed FAISS IVFPQ indexes.
- Evaluated name and address retrieval independently and in combined strategies at K=5, 10, 20, 50, and 100.
- Reported link recall, complete recall, candidate-count statistics, S2/S3 recall, and recall by match-count group.
- Produced a 50-link retrieval failure sample with similarities, available ranks, and failure categories.
- Recorded runtime and approximate memory use.

Full country validation:

| Result | Links |
| --- | ---: |
| Total positive links | 7,638,365 |
| Same-country | 7,638,365 |
| Different-country | 0 |
| Missing-country | 0 |
| Missing candidate lookup | 0 |

Country is therefore a lossless hard block on the available training ground truth.

Key retrieval results:

| Strategy | Link recall | Complete recall | Average candidates | Maximum candidates |
| --- | ---: | ---: | ---: | ---: |
| Exact union | 28.34% | 9.99% | 9.5 | 458 |
| All signals ranked, K=100 | 91.41% | 80.28% | 100.0 | 100 |
| Name + address union, K=100 | 93.53% | 83.79% | 197.8 | 200 |
| All signals union, K=100 | 93.59% | 83.86% | 199.9 | 589 |

The raw-recall winner is the all-signals union at K=100. The name-plus-address union at K=100 is the cleaner operational choice because it gives nearly identical recall while keeping every candidate set at or below 200 records.

Additional findings:

- Best-configuration S2 link recall: 95.34%.
- Best-configuration S3 link recall: 91.95%.
- Best-configuration complete recall for 6+-match entities: 72.48%.
- Address retrieval was stronger than name retrieval at each tested K.
- Frequent failure patterns were address-number differences, script/transliteration differences, very different names, missing addresses, and true matches ranked beyond the candidate budget.
- End-to-end runtime was 13.95 minutes with approximately 2.37 GB peak process memory.

### Task 2.5: Candidate generator selection

The exact saved Task 2 evaluation set was reused for a focused ablation of transliteration-aware names, address-number normalization, missing-address handling, and conservative legal-suffix normalization.

Key results:

| Configuration | Link recall | Complete recall | Average candidates | Maximum candidates |
| --- | ---: | ---: | ---: | ---: |
| Task 2 name + address union, K=100 | 93.53% | 83.79% | 197.8 | 200 |
| Baseline + transliteration | 94.79% | 86.82% | 244.1 | 300 |
| Baseline + address numbers | 96.23% | 89.77% | 272.5 | 300 |
| Baseline + legal suffixes | 94.45% | 85.92% | 249.7 | 300 |
| Combined ranked, cap=200 | 96.39% | 90.78% | 200.0 | 200 |
| **Combined baseline-preserving, cap=250** | **96.90%** | **91.59%** | **250.0** | **250** |

The frozen configuration preserves the original name-plus-address union first, then fills to 250 using reciprocal-rank fusion over original name, original address, transliterated name, number-aware address, and suffix-normalized name signals. It recovered 1,122 of 2,155 baseline-missed positive links without losing any baseline hits. S2/S3 recall reached 97.44%/96.38%, and 6+-match complete recall improved from 72.40% to 84.72%.

The standalone missing-address policy had no measurable effect because none of the 10,000 S1 evaluation queries had a missing address. Among 1,435 positive candidates with missing addresses, the frozen configuration improved link recall from 75.33% to 83.07% through its additional name signals.

Task 2.5 ran in 32.61 minutes with approximately 2.60 GB peak process RSS. It used training data only and performed no classifier, Task 3, test-prediction, or submission work.

### Task 3: Pairwise feature engineering and analysis

The frozen Task 2.5 candidates were reconstructed exactly from the saved checkpoint before any feature work. All 10,000 evaluation labels were also checked directly against `train_ground_truth.tsv`.

Pair dataset:

| Measure | Result |
| --- | ---: |
| Pair rows | 2,499,947 |
| Positive pairs | 32,252 |
| Negative pairs | 2,467,695 |
| Positive rate | 1.2901% |
| Negatives per positive | 76.51 |
| Retrieval-missed positives, excluded from pair labels | 1,033 |

The typed Parquet dataset contains 98 total columns, including metadata and audit columns. Sixty-six compact, inference-safe features are recommended for the first Task 4 experiment. They cover baseline, transliterated, and suffix-normalized names; address text and numeric structure; missingness; candidate source; frozen retrieval ranks/scores; and a small set of interpretable interactions.

Strong descriptive separators included retrieval best rank, reciprocal-rank-fusion evidence, independent strong-signal count, address token/character overlap, number-aware address evidence, and combined name-address agreement. These are descriptive findings, not fitted feature importance.

S3 positives had weaker address similarities and slightly weaker retrieval consensus than S2 positives, despite slightly stronger mean name similarities. Match-group similarity distributions were broadly stable, but retrieval best rank worsened from 2.34 for one-match entities to 3.75 for 6+ entities. Task 2.5-only recoveries were disproportionately S3, had much weaker raw name/address similarities, showed positive transliteration gain, and retained strong address-number agreement.

The reproducible split contains 8,000 train S1 entities and 2,000 validation S1 entities, with exactly 2,000/500 entities from each match-count group. There is no S1 overlap. Task 3 completed in 2.03 active minutes, used approximately 2.17 GB peak RSS, and produced a 147.6 MB pair-feature Parquet file.

### Task 3.5: Pre-model audit

A lightweight audit batch-scanned the 66 recommended features across all 2,499,947 existing pair rows without regenerating candidates or features. Every recommended field exists in Parquet, is inference-available, and excludes identifiers, labels, split metadata, and ground-truth-derived analysis fields. The scan found zero null, NaN, infinite, or constant recommended features and zero pair-row split mismatches. `address_exact` was the only near-constant feature under a 0.5% minority-prevalence rule, and seven high-correlation feature pairs were flagged for Task 4 review rather than removed. Eighteen difficult positives and eighteen hard negatives were inspected. The audit result is PASS: Task 3 artifacts are technically ready for Task 4 baseline modeling.

### Task 4A: Baseline pairwise matching models

Exactly two baseline families were trained using all 66 audited features and the existing 8,000/2,000 entity split. The scaled, class-balanced logistic regression converged in 27 iterations. A conservative class-weighted LightGBM model used training-derived `scale_pos_weight`, validation PR-AUC early stopping, and reached its best iteration at 1,489 of 1,500 rounds.

| Metric | Logistic regression | LightGBM |
| --- | ---: | ---: |
| PR-AUC | 0.929393 | 0.984601 |
| ROC-AUC | 0.998447 | 0.999529 |
| Log loss | 0.048406 | 0.007394 |
| Retrieved-candidate Recall@1 | 28.90% | 30.31% |
| Retrieved-candidate Recall@5 | 86.64% | 89.17% |
| Retrieved-candidate Recall@10 | 98.87% | 99.63% |
| Retrieved-candidate Recall@20 | 99.67% | 99.89% |

Validation contains 6,454 retrieved positive links and 239 additional links missed by candidate generation. LightGBM end-to-end Recall@20 is therefore 96.32%, despite 99.89% ranking recall among retrieved candidates. Its strongest gain features were retrieval best rank, combined name-address similarity, address token similarity, address Jaro-Winkler similarity, independent strong-signal count, RRF evidence, transliteration gain, suffix-normalized name similarity, and address-number agreement.

LightGBM materially outperformed the linear baseline. The main difficult positives involved weak or cross-script names, missing candidate addresses, numeric conflicts, S3 records, and 6+-match entities. The hardest negatives were frequently near-identical businesses with strong name/address evidence and convincing multi-signal retrieval support. Task 4A completed in 2.26 minutes at approximately 3.25 GiB peak RSS. No threshold, calibration, test, submission, or Task 4B work was performed.

### Task 4B: Rank robustness and entity decision policy

Task 4B reused the saved Task 4A LightGBM validation predictions and trained three controlled ablations at the same fixed 1,489-tree budget. Removing only `retrieval_best_rank` barely changed PR-AUC, from 0.984601 to 0.984466. Removing all 14 direct rank/position features reduced PR-AUC to 0.980114, while removing the full 22-feature retrieval-evidence family reduced it to 0.977563. The full model is therefore not dependent on one rank feature, but the broader retrieval evidence provides useful signal and remains justified.

The selected inference-safe decision policy is:

`predict a candidate link when LightGBM score >= 0.95`

Validation results:

| Measure | Result |
| --- | ---: |
| Predicted links | 6,244 |
| Precision | 95.79% |
| Retrieved-only recall / F1 | 92.67% / 94.20% |
| End-to-end recall / F1 | 89.36% / 92.46% |
| End-to-end complete entity rate | 65.20% |
| Average predicted links per S1 | 3.122 |
| S1 entities with zero predictions | 54 |

The maximum-complete comparator, a relative score-gap rule, reached 66.15% complete recovery but added 534 false positives and reduced end-to-end F1 to 89.04%. It did not improve complete recovery for the `3-5` or `6+` groups; its gain came from singleton/doubleton cases. The simpler global threshold was therefore frozen as the more stable operating point.

At the selected threshold, S2/S3 end-to-end F1 was 92.91%/92.04%. End-to-end complete recovery was 76.0%, 71.8%, 63.4%, and 49.6% for the `1`, `2`, `3-5`, and `6+` groups. The 239 candidate-generation misses remain unreachable. Other difficult cases include 473 retrieved positives rejected by the threshold, 263 selected false positives, cross-script and missing-address positives, weak retrieval consensus, numeric conflicts, and near-identical hard negatives.

LightGBM score reliability was inspected using fixed bins: expected calibration error was 0.002507 and Brier score was 0.002032. No calibration model was added. All 15 Task 4B integrity checks passed, including frozen-input hashes, entity-split checks, leakage checks, source/group reconciliation, and confirmation that no test data was accessed. Runtime was 3.98 minutes with approximately 2.63 GiB peak RSS.

### Task 5: Full-scale inference architecture and smoke test

Task 5 froze a machine-readable inference contract covering the exact Task 2.5 candidate configuration, five retrieval signals, normalization implementations, candidate cap, ordered 66-feature schema, missing-value conventions, saved LightGBM hash, threshold, batch schemas, and checkpoint/resume behavior.

The raw replay rebuilt the frozen retrieval path from S1/S2/S3 training records without ground truth. Across 1,000 validation S1 entities, every top-250 ID and retrieval score for all five signals matched the saved Task 2.5 checkpoint. Frozen candidate counts, IDs, and ordering also matched exactly. For 25 parity entities and 6,250 pairs, all 412,500 values in the ordered 66-feature matrix matched exactly. Fresh LightGBM scores differed from the saved float32 Task 4A scores by at most 2.98e-08, with zero decision differences at the 0.95 threshold.

Smoke-test results:

| Measure | Result |
| --- | ---: |
| S1 entities | 1,000 |
| Candidate/pair rows | 249,992 |
| Average / maximum candidates per S1 | 249.992 / 250 |
| Predicted links at 0.95 | 3,062 |
| Average predicted links per S1 | 3.062 |
| Zero-prediction entities | 25 |
| Atomic checkpoint batches | 10 |
| Checkpoint output size | 4.14 MiB |
| Batch feature/model/write runtime | 14.95 seconds |
| Candidate-generation runtime | 1,948.91 seconds |
| Peak RSS | 2.62 GiB |

The checkpoint test wrote one batch, resumed to write the remaining nine while skipping the completed batch, then performed a no-op rerun that skipped all ten. No duplicate or omitted S1-candidate pairs were found.

Using the 2,206,821-record training S1 table as a planning proxy, the measured projection is approximately 551.7 million candidate pairs, 11.34 hours total runtime, and 8.92 GiB of compressed prediction checkpoints. These are projections, not full-run measurements. Retrieval index construction is a fixed startup cost, and production should persist the retrieval bundle plus a bucketed normalized candidate-text store before launching the batch loop. The exact competition submission schema is not yet established and was not invented.

All 17 Task 5 parity checks and all 17 integrity checks passed. No test data, ground truth, model training, threshold tuning, full-scale predictions, or submission work was performed.

### Task 6: Persisted production retrieval bundle and candidate text store

Task 6 persisted the exact frozen retrieval state as ten fitted TF-IDF/random-projection transform files and ten country-partitioned FAISS IVFPQ indexes, plus bundle metadata. The resulting 21-artifact retrieval bundle is 3.45 GiB. It also created a 64-shard SQLite candidate text store containing all 10,320,219 S2/S3 records and every raw or normalized representation required by Task 3. The store is 2.90 GiB and supports selective lookup by encoded candidate ID without loading the full corpus into memory.

Reload validation used the same 1,000 Task 5 S1 entities. Both independent persisted-bundle reloads produced exact top-250 IDs and scores for all five signals. Frozen candidate counts, IDs, and ordering matched exactly for all 249,992 pairs. A representative 25-entity sample performed selective lookup for 6,250 unique candidates; every identity, source-derived field, raw text, normalized text, transliteration, suffix-normalized name, numeric-address sequence, country, and script indicator matched a direct raw S2/S3 scan.

Downstream parity also passed: all 412,500 ordered feature values matched Task 3, LightGBM scores differed by at most 2.98e-08 from the saved float32 reference, and all `score >= 0.95` decisions were exact. The reload-only production smoke produced the same 3,062 predicted links, 25 zero-prediction entities, ten checkpoint batches, and 4.14 MiB output as Task 5.

| Measure | Result |
| --- | ---: |
| Retrieval bundle artifacts / size | 21 / 3.45 GiB |
| Candidate text records / shards / size | 10,320,219 / 64 / 2.90 GiB |
| Retrieval parity checks | 22 / 22 passed |
| Text, feature, score, and decision parity checks | 15 / 15 passed |
| Integrity checks | 25 / 25 passed |
| One-time bundle and store build runtime | 59.91 minutes |
| Reload-only 1,000-S1 production smoke runtime | 32.65 seconds |
| Peak process RSS | 1.85 GiB |

The updated linear full-scale planning projection is approximately 551.7 million candidate pairs, 20.02 hours of reload-only inference, and 8.92 GiB of compressed prediction checkpoints. Persistent production state is 6.35 GiB, giving an estimated 15.28 GiB combined requirement with projected checkpoints. The longer runtime projection includes hash validation, selective disk text lookup, and the complete persisted-state workflow. It remains a projection rather than a full-scale execution measurement.

No Task 7 work, retrieval retuning, classifier training, threshold tuning, test-data access, ground-truth use, full-scale inference, test predictions, or submission generation was performed.

## Verification completed

- Confirmed that only training files were used.
- Reconciled global recall numerators and denominators to both source-level and match-group outputs.
- Confirmed all recall values are within valid bounds.
- Confirmed recall increases monotonically with K for each retrieval family.
- Confirmed the 10,000-entity sample contains exactly 2,500 entities per match-count group.
- Confirmed the Task 2 notebook review cells execute against the saved outputs.
- Confirmed exact Task 2 baseline parity in Task 2.5.
- Confirmed all Task 2.5 global totals reconcile to source and match-count outputs.
- Confirmed the Task 2.5 retrieval checkpoint contains five 10,000 by 250 ID/score pools.
- Confirmed exact Task 2.5 parity before Task 3 feature generation.
- Confirmed all 2,499,947 pair rows belong to frozen candidate sets with no duplicate pairs.
- Confirmed pair labels reconcile to 32,252 retrieved positives plus 1,033 retrieval misses.
- Confirmed no NaN/inf values in numeric features and no train/validation S1 overlap.
- Confirmed the leakage audit and all ten Task 3 consistency checks pass.
- Confirmed all 66 Task 4 features pass the Task 3.5 inference, numeric-integrity, and entity-split audit.
- Confirmed both Task 4A models use exactly 66 features and produce finite validation probabilities.
- Confirmed all 499,985 validation predictions are unique S1-candidate pairs with valid within-entity ranks.
- Confirmed all twelve Task 4A scope and leakage checks pass and the Task 2.5/Task 3 input hashes are unchanged.
- Confirmed Task 4B reproduced the 6,454 retrieved validation positives and reconciled all 239 unreachable links.
- Confirmed the selected Task 4B rule uses only the LightGBM pair score and does not use labels, match group, or ground-truth match count.
- Confirmed all fifteen Task 4B integrity checks pass and all frozen Task 2.5, Task 3, and Task 4A input hashes are unchanged.
- Confirmed exact Task 5 raw replay for all five top-250 retrieval ID/score arrays across 1,000 S1 entities.
- Confirmed exact Task 5 candidate counts/order and all 412,500 parity feature values across 25 entities.
- Confirmed Task 5 score differences are limited to saved float32 precision and produce zero threshold-decision differences.
- Confirmed Task 5 checkpoint/resume writes no duplicate or omitted pairs and no-ops after all batches complete.
- Confirmed all seventeen Task 5 integrity checks pass and frozen Task 2.5, Task 3, Task 4A, and Task 4B hashes are unchanged.
- Confirmed every Task 6 persisted transform, FAISS index, metadata file, and SQLite shard matches its manifest byte count and SHA-256 before use.
- Confirmed two independent Task 6 bundle reloads produce exact top-250 IDs and scores for all five signals.
- Confirmed Task 6 selective text lookup matches direct raw-source normalization and reproduces all 66 features, LightGBM scores, and threshold decisions.
- Confirmed all twenty-five Task 6 integrity checks pass and all frozen upstream hashes are unchanged.
- Confirmed all eight Python pipelines compile and all available review notebooks execute successfully.

## Current artifacts

### Original exploratory work

- `Notebooks/1_eda─_02_task1_match_analysis_──_03_candidate_generation.py`

### Reproducible code and notebooks

- `src/AmazonMLChallenge2026_Task1_Positive_Matches.py`
- `src/task2_candidate_generation.py`
- `src/task2_5_candidate_selection.py`
- `src/task3_pairwise_features.py`
- `src/task4a_baseline_models.py`
- `src/task4b_decision_analysis.py`
- `src/task5_full_inference_pipeline.py`
- `src/task6_persist_production_retrieval.py`
- `requirements-task4a.txt`
- `Notebooks/02_Task2_Candidate_Generation.ipynb`
- `Notebooks/03_Task2_5_Candidate_Selection.ipynb`
- `Notebooks/04_Task3_Pairwise_Features.ipynb`

### Results

- `outputs/task1_outputs/`: Task 1 schema, sample, pairwise inspection, agreement, patterns, flags, and summary.
- `outputs/task2_outputs/`: country validation, evaluation entities, strategy comparison, recall breakdowns, failure analysis, runtime, and summary.
- `outputs/task2_5_outputs/`: ablation metrics, source/group/address breakdowns, recovered links, remaining failures, runtime, frozen decision, manifest, and top-250 retrieval checkpoint.
- `outputs/task3_outputs/`: frozen candidate reconstruction, typed pair features, missed-positive features, schema, class balance, source/group/recovery analyses, hard cases, entity split, leakage audit, validation, Task 3.5 pre-model audit, runtime, manifest, and summary.
- `outputs/task4_outputs/`: logistic and LightGBM models, validation probabilities and ranks, pairwise/ranking metrics, source and match-group analyses, feature importance, hard cases, manifest, and Task 4A summary.
- `outputs/task4_outputs/task4b/`: rank ablations, validation decision-rule comparison, selected policy, source/group diagnostics, score reliability, aggregate and representative errors, ablation models, manifest, and Task 4B summary.
- `outputs/task5_outputs/`: frozen inference contract, raw-replay parity evidence, smoke checkpoints, runtime and scale estimates, integrity report, run manifest, and Task 5 summary.
- `outputs/task6_outputs/`: persisted retrieval transforms and FAISS indexes, 64-shard candidate text store, artifact manifests and hashes, reload parity evidence, production smoke checkpoints, runtime and scale projection, integrity checks, run manifest, and Task 6 summary.

## Not completed

The following stages remain intentionally unstarted:

- Full-scale candidate generation and batched inference for every S1 entity.
- Test-set prediction and submission-file generation.

## Next recommendation

Keep `Combined baseline-preserving augmentation, cap=250`, the persisted Task 6 retrieval bundle and candidate text store, the ordered 66-feature contract, the saved Task 4A LightGBM, and the global `score >= 0.95` rule frozen. Task 7A reconciled the stale 18.65-hour value to the canonical 20.02-hour Task 6 estimate, which includes pre-use hash validation and the explicit retrieval, lookup, feature, model, and checkpoint-write phases. Before any separately authorized full run, confirm at least 12.74 GiB free disk, fix the launch input fingerprint in the strengthened checkpoint contract, and retain batch size 100 with one native worker initially. Do not generate test predictions or a submission until the external submission schema is explicitly reviewed.
