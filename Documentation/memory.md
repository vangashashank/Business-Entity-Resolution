# memory.md

> **Update this file at the end of every work session, before closing the AI chat.**
> This is the single source of truth the AI reads first in every new session — if it's
> stale, the AI will guess wrong about what's done. Paste (or have the AI read) this
> file at the start of every new session.

---

## Last Updated
`YYYY-MM-DD HH:MM` — (update every session)

## Current Phase
Phase 0 — Setup & Shared EDA *(update as you move through Phases.md)*

## Currently Being Worked On
- File: *(none yet — not started)*
- What's half-done in it, if anything: —

## Completed ✅
- [ ] Repo scaffolding created
- [ ] `requirements.txt` frozen
- [ ] TSVs loaded, schema confirmed
- [ ] Noise-pattern table built
- [ ] Stratified val split created + committed (`dataset/val_split_ids.txt`)
- [ ] Normalization pass (`preprocessing.py`)
- [ ] Blocking v1 (TF-IDF)
- [ ] Blocking v2 (token/address key)
- [ ] Blocking v3 (phonetic)
- [ ] Candidate recall measured: ___%
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
- `YYYY-MM-DD`: decision — reason

## Known Issues / TODO
- *(nothing yet)*

## Next Immediate Step
Start Phase 0: scaffold the repo per `Architecture.md` and load the 4 TSVs.

---

## Reference Numbers (fill in as you get them — don't let the AI guess these)
- Candidate recall (blocking): ___
- Best validation F0.5: ___ (model: ___, threshold: ___)
- Singleton rate in train ground truth: ___%
- Public leaderboard F0.5: ___
