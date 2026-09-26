# PRD.md — Business Entity Resolution (Amazon ML Challenge 2026)

## 1. One-liner
Given noisy business records from 3 sources, predict which Source-2/Source-3 records
refer to the same real-world business as each Source-1 (reference) entity — optimizing
for macro F0.5 (precision-weighted).

## 2. Problem Summary
- Source 1 = deduplicated reference set. For every S1 entity, find 0, 1, or many
  matching records in Source 2 and Source 3.
- Fields per record: `entity_id` (prefix = source), `business_name`,
  `business_address`, `country`.
- Train countries: US, India. **Test adds France** — the pipeline must treat `country`
  as an open set, never a hardcoded `{US, India}` filter.
- Noise: legal-suffix variants, abbreviations, typos, transliteration, word-order
  swaps, landmark addresses, missing components.

## 3. Solo Scope Decision
The teammate's plan splits work into 3 people-tracks (Blocking / Features / Scoring)
running in parallel over 48 hours. **This project is being built solo, vibecoding with
an AI pair-programmer.** So:
- No parallel tracks. Work runs as one sequential pipeline, one AI conversation/file at
  a time (see `Phases.md`).
- Total effort budget stays similar to the 3-person plan (~48–60 hrs of actual work),
  just spread across more calendar time instead of 3 people at once.
- `memory.md` replaces the "daily sync" — it's how you (and the AI) stay oriented
  across sessions instead of a team standup.

## 4. Success Metrics
- **Primary:** macro-averaged F0.5 on the private leaderboard.
  `F0.5 = 1.25·P·R / (0.25·P + R)` per S1 entity, averaged over all S1 entities.
  Singletons scored too (empty prediction on a true singleton = 1.0).
- **Secondary (internal):** candidate recall out of blocking (target ≥95% before
  investing more time in modeling — a model can't recover a match blocking never
  generated).
- **Gate metric:** `utils/validate_submission.py` returns `PASS` before every upload.

## 5. Functional Requirements
- Pipeline produces exactly two TSVs: `output/matching_results.tsv` (scored) and
  `output/candidate_pairs.tsv` (blocking set fed to the model, audit-only).
- Every test S1 entity appears exactly once in both files.
- `matched_entity_ids` ⊆ `candidate_entity_ids` for every S1 entity.
- No self-matches to S1, no duplicate IDs, no IDs outside the test set.
- Handles France (and any future unseen country) without crashing or degrading
  silently.

## 6. Non-Functional Requirements
- **Licensing:** final matching model must be MIT/Apache-2.0 licensed and ≤8B
  parameters. Applies to any pretrained component used (e.g. an embedding model).
- **Fair play:** zero external API/database/geocoding lookups. Everything derived
  from the provided train/test TSVs only.
- **Reproducibility:** submission zip's `code/business_entity_resolution/` must let a
  stranger regenerate both output files from raw data using only what's in the folder.
- **Hardware:** local machine has no meaningful GPU. Anything embedding- or
  neural-model-heavy runs on Google Colab (see `Architecture.md` and `Phases.md` for
  exactly which steps).

## 7. Deliverables (final zip)
```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   (filled in)
```

## 8. Out of Scope
- Multi-person workflow tooling (branch-per-person, team experiment sheet as a
  literal Google Sheet — `memory.md` + `experiments.md` cover this solo).
- Any deep-learning fine-tuning beyond what fits comfortably in a Colab free-tier
  session, unless recall/precision plateaus force it.

## 9. Open Questions (fill in once known)
- Actual challenge start/end time (if it's a timed live hackathon vs. a rolling
  submission window) — affects `Phases.md` pacing.
- Dataset size (row counts) — determines whether TF-IDF/NearestNeighbors blocking is
  enough locally or whether Colab is needed even for blocking, not just embeddings.
