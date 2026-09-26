# Plan.md — Prompts to Vibecode Each Phase

How to use this file: at the start of each phase, open a fresh AI coding session
(Claude Code or similar), have it read `PRD.md`, `Architecture.md`, `RULES.md`, and
`memory.md` first, then paste the phase's prompt. After the phase, update `memory.md`
yourself (or ask the AI to draft the update, then review it).

**Standing context to give the AI at the start of every session:**
> "Read PRD.md, Architecture.md, RULES.md, and memory.md before doing anything. Follow
> RULES.md strictly — no external API/data lookups, no hardcoding country to
> {US, India}, always sep='\t', run utils/validate_submission.py after any change to
> output-generating code. Update memory.md's 'Currently Being Worked On' and
> 'Completed' sections as you go, not just at the end."

---

## Phase 0 — Setup & EDA
```
I have already set up the repo structure but i didn't put up the structure similar to Architecture.md, just dont delete any files which are not there in Architecture.md just add on top to replicate the repo structure as much as you can from Architecture.md exactly, and update the Architecture.md with the right repo structure. Then write
notebooks/01_eda.ipynb that:
1. Loads all 4 TSVs with sep="\t", prints shape/dtypes/null counts for each
2. Prints country.value_counts() on train and test separately, confirm France
   only appears in test
3. Computes the 0/1/2/3+ match-count distribution from train_ground_truth.tsv
   and plots it (use the color palette from design.md)
4. Joins ground truth -> source1 -> source2/source3 and prints 30-50 real
   matched pairs (name + address side by side) so I can read them and build a
   noise-pattern table by hand
5. Creates a stratified train/validation split by match-count bucket
   (0/1/2/3+), saves the S1 entity IDs to dataset/val_split_ids.txt, and NEVER
   regenerates this file once it exists - check if it exists first and load it
   instead of resplitting.
Do not write any modeling code yet. Stop after EDA and wait for me to review
the noise-pattern table before continuing.
```

## Phase 1 — Normalization + Blocking
```
Implement src/preprocessing.py: normalization (lowercase, strip punctuation,
collapse whitespace, legal-suffix map, address-abbreviation map). Keep both
raw and normalized name/address columns in the output.

Then implement src/blocking.py with three candidate-generation strategies:
1. TF-IDF char n-grams (analyzer='char_wb', ngram_range=(2,4)) on business_name,
   fit on combined S1+S2+S3 vocabulary, NearestNeighbors cosine, top_k=20
2. Token/address blocking: sorted significant address tokens (drop stopwords
   like "road", "street") as a blocking key, or coarse geo key + name token
   overlap >= 1
3. Phonetic: jellyfish.metaphone on the first significant name token

Union and dedupe the three candidate sets per S1 entity into
candidate_pairs.tsv (schema from README.md). Then write a small script that
loads dataset/val_split_ids.txt and measures candidate recall: what % of true
ground-truth matches for validation S1 entities appear in the candidate set.
Print that number clearly - this is the metric I care about most right now.
```

## Phase 1b — Colab Embeddings (only if Phase 1 recall < 95%)
```
[Run this in a Colab notebook, GPU runtime enabled]

Load business_name and business_address columns for train S1/S2/S3 (upload
just these columns as a parquet, not the full dataset). Generate embeddings
with sentence-transformers using "all-MiniLM-L6-v2" (MIT licensed, well under
8B params) on the concatenation of normalized name + address. Compute cosine
similarity top-k=20 per S1 entity against S2/S3 embeddings. Save the raw
embeddings to embeddings.parquet in Google Drive, and save the top-k candidate
pairs to a separate candidates_embedding.tsv. Batch the encoding with a
progress bar and checkpoint every N batches to Drive in case the Colab session
disconnects.
```
**After this runs:** download `embeddings.parquet` (or just
`candidates_embedding.tsv` if you don't need the raw vectors elsewhere) into
`artifacts/`, then back to local for everything else.

## Phase 2 — Features + Baseline Model
```
Implement src/features.py with this feature set v1: name_levenshtein_ratio,
name_jaccard (token set), name_tfidf_cosine, address_levenshtein_ratio,
address_jaccard, address_tfidf_cosine, country_match (binary),
name_length_ratio, token_overlap_count. Use rapidfuzz for string similarity.

Implement src/train.py with a Logistic Regression baseline
(class_weight='balanced'). Implement src/inference.py that runs candidates ->
features -> model -> threshold 0.5 (hardcoded for now) -> matching_results.tsv.

Run the full pipeline end-to-end on a small subset first to confirm it works,
then run utils/validate_submission.py on the output. Fix anything it flags
before doing anything else.
```

```
Now add feature set v2 to features.py: name_first_token_match,
numeric_token_match (compare digit sequences - catches PIN codes and building
numbers), abbreviation_normalized_match (compare names after suffix
stripping), address_component_count_diff. Retrain and compare Random Forest
and XGBoost against the Logistic Regression baseline using src/scoring.py
(build that first if it doesn't exist - see Phase 3 prompt). Log each result
to experiments.md with columns: change, candidate_recall, val_F0.5, precision,
recall, notes, date.
```

## Phase 3 — Scorer & Threshold
```
Implement src/scoring.py: an exact macro F0.5 scorer matching the formula in
PRD.md section 4. Handle singleton edge cases explicitly: 0 predicted & 0 true
-> F0.5 = 1.0; 0 predicted & >0 true -> F0.5 = 0; >0 predicted & 0 true -> F0.5
= 0. Expose it as score(predictions_df, ground_truth_df) -> float, plus a
per-entity breakdown for error analysis.

Before using this anywhere else, write a toy example with 2-3 hand-picked
entities where I can compute precision/recall/F0.5 by hand, run the scorer on
it, and show me both the hand-computed and scorer-computed numbers side by
side so I can verify they match.
```

```
Implement src/threshold.py: given match-probability scores from the trained
model, sweep thresholds from 0.3 to 0.95 in 0.05 steps, compute validation
F0.5 at each via scoring.py, and plot F0.5 vs threshold using the design.md
color palette with a vertical dashed line at the chosen maximum. Print the
best threshold and its F0.5/precision/recall.
```

## Phase 4 — Error Analysis Loop (repeat this prompt each pass)
```
Run the current full pipeline (latest blocking + model + threshold) on the
validation split and score it. Then pull 15-20 false positives and 15-20
false negatives (entity ID, names, addresses, model score, which feature
values look off). For each one, tell me whether it looks like a blocking
problem (true match never appeared as a candidate), a modeling problem
(candidate was there but scored too low/high), or a threshold problem. Group
them by pattern against the noise-pattern table from Phase 0. Suggest ONE
concrete next change, not five - I want to test one hypothesis at a time and
log the before/after in experiments.md.
```

## Phase 5 — France / Open-Set Sanity
```
Search the entire src/ directory for any hardcoded reference to "US" or
"India" as country values (including in list literals, equality checks, or
one-hot encoding that assumes a fixed vocabulary). Flag every instance. Then
construct 5-10 synthetic France-like records (SARL/SA suffixes, French postal
code format, French street naming) and run them through preprocessing,
blocking, and feature extraction. Confirm nothing crashes, divides by zero, or
silently produces empty/NaN features.
```

## Phase 6 — Freeze & Submit
```
Freeze features.py and train.py - no further changes past this point. Run
full inference on the actual test set (dataset/test/), generate both
candidate_pairs.tsv and matching_results.tsv in output/. Run
utils/validate_submission.py and show me the full output. Then run the manual
sanity checks: exactly one row per test S1 entity, no self-matches to S1, no
IDs outside the test set, no duplicate IDs within any list, and confirm every
matched ID in matching_results.tsv also appears in candidate_pairs.tsv for the
same entity.
```

## Phase 7 — Packaging
```
Clean up code/business_entity_resolution/src/ - remove dead experiments and
unused code paths, keep only what the final pipeline actually uses. Write
code/business_entity_resolution/README.md with exact reproduction steps: env
setup, data path expectations, and the command to run each stage (blocking ->
features -> train -> inference -> both output files). Generate
requirements.txt from a filtered pip freeze (pinned versions, only actual
deps). Then help me fill Documentation_template.md section by section, pulling
the ablation numbers directly from experiments.md so the "with vs without X"
deltas are accurate, not summarized from memory.
```
