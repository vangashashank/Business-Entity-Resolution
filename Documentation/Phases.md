# Phases.md

Adapted from the teammate's 3-person/48-hour plan into a **sequential solo build**.
Same total scope, no parallel tracks. Day labels assume a flexible multi-day window —
compress into hours if this turns out to be a locked hackathon clock.

Every phase ends with: commit + update `memory.md`.

---

## Phase 0 — Setup & Shared EDA (Day 1, morning) — 💻 Local
- [ ] Create repo with the folder structure from `Architecture.md`
- [ ] `pip install` the stack from `RULES.md`, freeze into `requirements.txt`
- [ ] Load all 4 TSVs with `sep="\t"`, check shape/dtypes/nulls
- [ ] `country.value_counts()` on train — confirm France is test-only
- [ ] Compute 0/1/2/3+ match-count distribution on `train_ground_truth.tsv`
      (this tells you the singleton rate — drives threshold strategy later)
- [ ] Pull 30–50 real matched pairs, read them manually → build a noise-pattern table
      (suffixes, abbreviations, typos, reordering, missing fields)
- [ ] **Stratified train/validation split** by match-count bucket (0/1/2/3+). Save
      entity IDs to `dataset/val_split_ids.txt` and commit — never regenerate this.
      *(Skipping this is the single biggest way to fool yourself: an
      "always-predict-nothing" model looks great on a singleton-skewed validation
      set.)*

## Phase 1 — Normalization + Blocking (Day 1 afternoon – Day 2) — 💻 Local
- [ ] Normalization pass: lowercase, strip punctuation, collapse whitespace;
      legal-suffix map; address-abbreviation map. Keep raw + normalized columns.
- [ ] Blocking strategy 1: TF-IDF char n-grams (`analyzer='char_wb'`,
      `ngram_range=(2,4)`) + `NearestNeighbors` cosine, top-k=20 per S1
- [ ] Blocking strategy 2: token/address blocking key (sig. address tokens minus
      stopwords, or coarse geo key + name token overlap ≥1)
- [ ] Blocking strategy 3: phonetic (`jellyfish.metaphone`/soundex on first name
      token)
- [ ] Union + dedupe all three → first `candidate_pairs.tsv`
- [ ] **Measure candidate recall** on validation split — this is the single KPI that
      matters most right now. If < 85%, this is priority #1 before touching modeling.
- [ ] **Decision point:** if recall is still < 95% after tuning the three classical
      strategies → this is when you move to Phase 1b (Colab embeddings). Otherwise
      skip straight to Phase 2.

## Phase 1b (optional) — Embedding-Based Blocking — 🖥️ Colab GPU
Only if Phase 1's classical blocking plateaus below ~95% recall.
- [ ] Upload normalized `business_name` + `business_address` columns (not the whole
      repo) to Colab as a `.parquet`
- [ ] Generate embeddings with a small MIT/Apache model (`all-MiniLM-L6-v2` or
      `bge-small-en-v1.5`), cosine-similarity top-k as a 4th candidate source
- [ ] Save `embeddings.parquet` to Drive → download to `artifacts/` locally
- [ ] Re-measure candidate recall locally with the 4th source unioned in
- **Download from Colab:** `embeddings.parquet`
- **Back to local** for everything else — don't split the rest of the pipeline across
  environments.

## Phase 2 — Features + Baseline Model (Day 2–3) — 💻 Local
- [ ] Feature set v1: name/address levenshtein ratio, jaccard, tfidf cosine,
      country_match, name_length_ratio, token_overlap_count
- [ ] Baseline: Logistic Regression, `class_weight='balanced'`
- [ ] First end-to-end run: candidates → features → LR → threshold 0.5 →
      `matching_results.tsv` → `validate_submission.py` — **confirms plumbing works
      before optimizing anything**
- [ ] Feature set v2: name_first_token_match, numeric_token_match (PIN/building
      numbers), abbreviation_normalized_match, address_component_count_diff
- [ ] Model iteration: Random Forest, then XGBoost/LightGBM — compare validation F0.5
      against the LR baseline (via `scoring.py` from Phase 3, build that first if not
      done yet)

## Phase 3 — Scorer, Threshold, Singletons (interleave with Phase 2) — 💻 Local
Build this early — everything else is judged by it.
- [ ] Implement exact macro F0.5 scorer per the formula in `PRD.md` §4, handling the
      three singleton edge cases explicitly
- [ ] Hand-verify on a 2–3 entity toy example you compute yourself — **do this before
      trusting any other number in this project**
- [ ] Wrap as `score(predictions_df, ground_truth_df) -> float` + per-entity breakdown
- [ ] Threshold sweep utility (0.05 steps, 0.3–0.95), plot val F0.5 vs. threshold, pick
      the max
- [ ] Singleton/abstention logic: below threshold → empty list. Consider a two-tier
      threshold (candidate-worth-considering vs. commit-to-match)

## Phase 4 — Integration / Error-Analysis Loop (repeat 3–4 times) — 💻 Local
1. **Merge** latest blocking + model + threshold into one pipeline run
2. **Score** on validation, log to `experiments.md` (change, recall, F0.5, P, R, notes,
   date)
3. **Error analysis:** pull 15–20 false positives and 15–20 false negatives.
   - FPs usually: over-aggressive suffix stripping, generic city names dominating
     address similarity, partial-name substring matches
   - FNs: check whether dropped at blocking (recall problem), scored too low
     (model problem), or filtered by threshold
   - Categorize each against the Phase 0 noise-pattern table
4. **One concrete next-change**, based on the analysis
5. Repeat. Suggested focus per pass:
   | Pass | Focus |
   |---|---|
   | 1st | systematic false-positive source (generic terms/addresses) |
   | 2nd | recover missed true matches (blocking or features) |
   | 3rd | threshold refinement, singleton edge cases |
   | 4th | France/open-set sanity (see Phase 5) if not already solid |
- [ ] **Checkpoint after each loop:** did validation F0.5 improve? If a change didn't
      move the number, revert it — don't keep it on vibes.

## Phase 5 — France / Open-Set Sanity Check — 💻 Local
- [ ] Search codebase for any hardcoded `country in ["US", "India"]` — remove
- [ ] Construct a few France-like synthetic addresses (different postal formats,
      "SARL"/"SA" suffixes) and manually confirm blocking + features don't
      divide-by-zero or silently no-op on unfamiliar formats
- [ ] Confirm the pipeline handles a never-seen `country` value without crashing

## Phase 6 — Freeze & Leaderboard Submission — 💻 Local
- [ ] Freeze `features.py`/`train.py` — no more changes past this point unless a
      validator failure is found
- [ ] Run full test-set inference → `candidate_pairs.tsv`, `matching_results.tsv`
- [ ] Run `validate_submission.py` — fix every issue, don't upload with warnings
- [ ] Sanity checks: one row per test S1 entity, no self-matches, no out-of-set IDs,
      no dupes, matched IDs ⊆ candidate IDs
- [ ] Upload to leaderboard, confirm `SCORED` status with an F0.5 number

## Phase 7 — Packaging & Documentation — 💻 Local
- [ ] `code/business_entity_resolution/src/` — clean, runnable, dead experiments
      removed
- [ ] `README.md` — exact reproduction steps: env → data path → blocking → features →
      model → threshold → both output files
- [ ] `requirements.txt` — pinned, filtered `pip freeze`
- [ ] Fill `Documentation_template.md` directly from `experiments.md` — the
      with/without-X ablation deltas are what pay off most here
- [ ] License/param check: confirm any pretrained component is MIT/Apache-2.0, ≤8B
      params
- [ ] Final zip structure check against the exact required layout

## Phase 8 — Buffer
Reserve explicitly, don't schedule real work into it:
- Re-run the validator one final time on the exact files going into the zip
- Fix anything the Phase 7 review caught
- If nothing's broken: one more error-analysis pass on the highest-impact remaining
  failure mode

---

## Risk Register (unchanged from the team plan — still applies solo)
| Risk | Mitigation |
|---|---|
| Scorer bug invalidates all decisions | Hand-verify on toy example before Phase 4 |
| Blocking recall too low, model can't compensate | Track recall as a first-class KPI from Phase 1 onward |
| Submission format rejected | Run `validate_submission.py` after every pipeline change |
| Country-specific hack breaks on France | Explicit Phase 5 check + code search for hardcoded country strings |
| Colab session times out / disconnects mid-run | Save embeddings/model to Drive incrementally, don't rely on one long uninterrupted cell |
