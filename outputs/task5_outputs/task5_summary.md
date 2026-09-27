# Task 5 - Full-Scale Inference Architecture and Smoke Test

## Result

**PASS**

Task 5 rebuilt the frozen retrieval path from raw training S1/S2/S3 records, replayed the exact Task 2.5 candidate and Task 3 feature contracts, scored with the saved Task 4A LightGBM, applied the frozen Task 4B threshold, and exercised atomic batch checkpoint/resume behavior. No ground truth, test data, model training, threshold tuning, full-scale prediction, or submission generation was performed.

## Parity

- Parity checks passed: **17/17**
- Maximum 66-feature absolute difference: **0**
- Maximum LightGBM score absolute difference: **2.98e-08**
- Candidate IDs, counts, order, feature order, feature values, scores, and `score >= 0.95` decisions were required to meet the documented exact/strict tolerances.

## Smoke test

- S1 entities: **1,000**
- Candidate/pair rows: **249,992**
- Average / maximum candidates per S1: **249.992 / 250**
- Predicted links at 0.95: **3,062**
- Average predicted links per S1: **3.0620**
- Zero-prediction entities: **25**
- Checkpoint batches: **10**
- Checkpoint output size: **4.14 MiB**
- Peak process RSS: **2.62 GiB**

Checkpoint/resume test: **passed**. The first invocation wrote one batch, the resumed invocation completed the remaining batches while skipping the completed checkpoint, and a final rerun wrote zero new batches.

## Scale projection

The training-scale planning proxy contains **2,206,821** S1 entities and approximately **551,687,595** candidate pairs. The measured projection is **11.34 hours**, approximately **8.92 GiB** of compressed prediction checkpoints, and **6.29 MiB** for the raw float32 feature matrix at the initial batch size. These are projections, not measured full-scale results. Retrieval index/model construction is treated as a fixed startup cost; larger-query throughput, candidate-store construction, and compression can be nonlinear.

## Integrity

All **17/17** integrity checks passed. The model contract is exactly 66 ordered features, the candidate cap is 250, the threshold is 0.95, frozen hashes remained unchanged, checkpoint resume reconciled without duplicates or omissions, and no forbidden inference field was used.

## Readiness

The batch architecture and frozen inference contract are technically ready for a separately reviewed full-scale run. Task 5 stops here. The future run should first persist the retrieval bundle and bucketed normalized candidate-text store, then execute the deterministic batch loop under the frozen contract. No full-scale inference was launched.
