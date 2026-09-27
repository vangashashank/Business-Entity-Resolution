# Task 7A Full-Scale Inference Readiness Audit

Status: **PASS**

## Decision

The frozen production pipeline is operationally ready for a separately authorized Task 7B full-inference run. Task 7A did not launch that run, access test data, create test predictions, or generate a submission.

## Scale reconciliation

- Canonical estimate: **20.02 hours** for 551,687,595 pairs.
- Canonical source: `task6_scale_estimate.json`, derived from 32.655 seconds for 1,000 S1 entities.
- The stale 18.65-hour estimate was a linear extrapolation from an earlier 30.421-second Task 6 total that excluded artifact hash validation and used the earlier aggregate downstream timing.
- Current Task 6 evidence includes pre-use hash validation and the explicit feature, model, and checkpoint-write phases. The final recommendation in `PROJECT_STATUS_SUMMARY.md` was stale and was corrected to 20.02 hours.

## Storage readiness

- Retrieval bundle: 3.45 GiB.
- Candidate text store: 2.90 GiB.
- Projected checkpoints: 8.92 GiB.
- Combined expected footprint: 15.28 GiB.
- Recommended safety margin: 3.82 GiB (25%).
- Minimum free space before launch with Task 6 state already present: 12.74 GiB.
- Fresh-deployment capacity including margin: 19.10 GiB.
- Current free space: 314.92 GiB; disk preflight PASS.

## Measured 100-S1 smoke

- Candidate pairs: 25,000; output: 0.414 MiB.
- Total readiness-smoke runtime: 9.863 seconds.
- Hash validation: 2.635s; bundle load: 0.795s; retrieval query: 1.093s.
- Candidate assembly: 0.057s; selective text lookup: 3.249s.
- Feature generation: 0.631s; LightGBM: 0.863s; checkpoint write: 0.017s.
- Peak RSS: 920.1 MB.
- Retrieval, candidate order, all 66 features, LightGBM scores, threshold decisions, and the final Parquet batch matched the frozen pipeline.

## Runtime scaling

- Fixed startup: source/input fingerprint validation, persisted-artifact hash validation, and retrieval-bundle loading.
- Primarily S1-count scaling: five-signal retrieval queries and candidate assembly.
- Primarily candidate-pair scaling: selective text lookup, 66-feature generation, LightGBM inference, and checkpoint writing.
- In the Task 6 1,000-S1 benchmark, selective lookup (11.58s) and LightGBM inference (9.01s) were the largest measured repeatable stages, followed by feature generation (5.79s).
- Operational constraint: Task 7A reproduced a native mixed-OpenMP exit with eight FAISS workers. The audited smoke therefore pins one native worker and remains byte-identical. Task 7B should retain one worker unless a separate preflight proves a higher count stable.

## Checkpoint and input contracts

- Checkpoint audit: PASS. The Task 7A contract explicitly binds model, feature schema, Task 5 contract, Task 6 retrieval/store manifests, threshold, source input fingerprint, deterministic batch boundaries, and output schema.
- Completed batches are committed by temporary Parquet plus `os.replace`, followed by an atomic manifest update.
- Restart verified one completed batch, skipped it, completed the remaining batch, and a completed rerun became a no-op.
- A deliberately corrupted completed batch was rejected by SHA-256/size verification; an invalid uncommitted temporary file was not treated as complete.
- Input contract: PASS on 100 training S1 records. Missing column, duplicate ID, unknown country, and both-text-missing negative tests all failed safely.

## Output boundary

Internal checkpoints contain: `source1_entity_id`, `candidate_entity_id`, `candidate_source`, `lightgbm_score`, `predicted_match`, `candidate_position`, and `batch_id`.

The official competition submission schema, required column naming/order, grouping rules, empty-match representation, and file-format constraints are not established in repository artifacts. This remains an external contract and must be handled only by a later submission-transformation task. It does not block internal inference readiness.

## Recommendation

Task 7B may be authorized only as a separate request after confirming at least 12.74 GiB free space, fixing the launch input fingerprint in the run manifest, retaining batch size 100 and one native worker initially, and using the strengthened checkpoint contract. No unresolved blocker prevents inference; the submission format remains deliberately unresolved.
