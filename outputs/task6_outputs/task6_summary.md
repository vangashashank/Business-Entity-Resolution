# Task 6 Summary

Status: **PASS**

## Persisted production assets

- Frozen candidate generator: `Combined baseline-preserving augmentation, cap=250`, cap `250`.
- Retrieval bundle: 21 artifacts, 3.45 GiB.
- Candidate text store: 10,320,219 S2/S3 records in 64 SQLite shards, 2.90 GiB.
- Candidate lookup is selective by encoded ID. It does not load the full text corpus into memory.

## Reload parity

- Retrieval checks passed: 22/22.
- Candidate text, feature, score, and decision checks passed: 15/15.
- Integrity checks passed: 25/25.
- Top-250 signal IDs, ordering, candidate counts, and assembled candidates are exact.
- Retrieval scores satisfy absolute tolerance `1e-06`.
- All 66 ordered features satisfy Task 5 dtype-specific tolerances; LightGBM scores satisfy `5e-08` and score >= 0.95 decisions are exact.

## Production smoke

- S1 entities: 1,000.
- Pair rows: 249,992; average 249.992; max 250.
- Predicted links: 3,062; zero-prediction entities: 25.
- Checkpoint files: 10; bytes: 4,342,181.
- One-time build runtime: 59.91 minutes.
- Reload-only production smoke runtime: 32.65 seconds.
- Bundle load / query: 0.70 / 2.32 seconds.
- Store initialization / candidate assembly / text lookup: 0.0000 / 0.55 / 11.58 seconds.
- Feature generation / LightGBM / checkpoint write: 5.79 / 9.01 / 0.07 seconds.
- Process peak RSS: 1896.4 MB.

## Full-scale projection

- Target S1 entities: 2,206,821.
- Projected candidate pairs: 551,687,595.
- Linear projected runtime: 20.02 hours.
- Projected checkpoint bytes: 8.92 GiB.
- Persistent plus projected checkpoint storage: 15.28 GiB.
- Recommended initial S1 batch size: 100.
- This is a projection only. No full-scale run was performed.

## Boundaries

No Task 7 work, classifier training, threshold tuning, retrieval retuning, test-data access, ground-truth use, full-scale inference, test predictions, or submission generation was performed.
