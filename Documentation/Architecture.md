# Architecture.md

## 1. Pipeline Flow

```
train/test *.tsv (sep="\t")
        │
        ▼
[1] Preprocessing / Normalization
    - lowercase, strip punctuation, collapse whitespace
    - legal-suffix map (Ltd/Limited, Pvt/Private, Corp/Corporation, & vs "and")
    - address-abbrev map (Rd/Road, St/Street)
    - keep BOTH raw and normalized columns (A/B test which helps blocking)
        │
        ▼
[2] Blocking / Candidate Generation  → candidate_pairs.tsv
    - TF-IDF char n-gram cosine (name) + NearestNeighbors, top-k per S1
    - token/address blocking key (sig. tokens, drop stopwords like "road")
    - phonetic blocking (jellyfish.metaphone / soundex on first name token)
    - [optional, Colab] sentence-embedding cosine similarity as a 4th blocking
      signal if the above plateaus below ~95% recall
    - union + dedupe all candidate sources per S1 entity
        │
        ▼
[3] Feature Engineering
    - name: levenshtein_ratio, jaccard (token set), tfidf_cosine, first_token_match,
      length_ratio
    - address: levenshtein_ratio, jaccard, tfidf_cosine, numeric_token_match
      (PIN/building numbers), component_count_diff
    - country_match (binary), abbreviation_normalized_match
        │
        ▼
[4] Matching Model (train + inference)
    - baseline: Logistic Regression, class_weight="balanced"
    - iterate: Random Forest / XGBoost / LightGBM (CPU is fine at this scale —
      no GPU needed for classical gradient boosting)
    - [optional, Colab GPU] only if a neural/embedding-based matcher is added
        │
        ▼
[5] Threshold & Singleton Logic
    - sweep thresholds against local F0.5 scorer, pick argmax (not 0.5 default)
    - below-threshold → empty match list, not a weak guess
        │
        ▼
[6] Output Generation  → matching_results.tsv
        │
        ▼
[7] Validation (utils/validate_submission.py) → PASS/FAIL before every upload
```

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Data handling | pandas, numpy | standard, no GPU needed |
| String similarity | rapidfuzz (preferred over python-Levenshtein, faster + MIT) | Jaccard/Levenshtein features |
| Phonetic | jellyfish | metaphone/soundex for typo/transliteration blocking |
| Blocking vectors | scikit-learn `TfidfVectorizer` + `NearestNeighbors` | runs fine on CPU |
| Model | scikit-learn (LogReg, RF), xgboost or lightgbm | CPU-trainable at this data scale, MIT/Apache licensed |
| Embeddings (optional) | `sentence-transformers` with a small MIT/Apache model (e.g. `all-MiniLM-L6-v2`, `bge-small-en-v1.5`) | only if TF-IDF+phonetic blocking recall is insufficient; check license + param count (both well under 8B) |
| Plotting | matplotlib, seaborn | EDA + error-analysis charts (see `design.md`) |
| Env | Python 3.10+, `requirements.txt` pinned via `pip freeze` | reproducibility requirement |
| Heavy compute | Google Colab (free/Pro tier) | see §4 below |
| Versioning | git, single `main` branch (solo — no branch-per-person needed) | keep `main` always validator-passing |

## 3. Folder Structure (solo, simplified from the 3-person layout)

```
amazon-ml-challenge-2026/
├── dataset/
│   ├── train/  (train_source1.tsv, train_source2.tsv, train_source3.tsv,
│   │            train_ground_truth.tsv)
│   ├── test/   (test_source1.tsv, test_source2.tsv, test_source3.tsv)
│   └── val_split_ids.txt          # committed once, never regenerated
├── notebooks/
│   └── 01_eda.ipynb
├── colab_notebooks/                # anything that runs on Colab lives here
│   └── 02_embeddings.ipynb         # only if embeddings are needed
├── src/
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
│   ├── inference.py
│   ├── scoring.py                  # local F0.5 metric — build this FIRST
│   └── threshold.py
├── artifacts/                      # things exported from Colab land here
│   ├── embeddings.parquet          # (if used)
│   └── model.pkl / model.json
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── utils/
│   └── validate_submission.py      # provided by organizers
├── experiments.md                  # change, recall, F0.5, P, R, notes, date
├── memory.md
├── requirements.txt
└── README.md
```

## 4. Local vs. Colab Split

Everything in this pipeline runs comfortably on CPU **except** dense embedding
generation over the full name+address corpus with a transformer model, which is slow
without a GPU. Rule of thumb:

| Step | Where | Why |
|---|---|---|
| Preprocessing, blocking (TF-IDF/phonetic/token), feature engineering, LogReg/RF/XGBoost training, scoring, thresholding, output generation, validation | **Local** | classical ML — CPU is fine, no reason to add Colab friction |
| Sentence-embedding generation (only if added as a 4th blocking signal or feature) | **Colab (GPU runtime)** | transformer forward passes over tens of thousands of rows are slow on CPU |
| Any neural/Siamese matcher, if you go that route later | **Colab (GPU runtime)** | training needs a GPU to iterate in reasonable time |

**Handoff protocol (Colab ⇄ local):**
1. Upload only the minimal input the Colab step needs (e.g. normalized
   name+address columns as a `.parquet`, not the whole repo).
2. Colab notebook writes its output (embeddings `.parquet`, model weights
   `.pkl`/`.pt`) to Google Drive.
3. Download that single artifact file back into `artifacts/` locally, commit it to
   git (or `.gitignore` it and just keep it local + backed up in Drive — decide once
   file size is known), and resume the rest of the pipeline locally.
4. Never keep the "current pipeline" split across both environments at once — Colab
   is a side-trip that produces one artifact file, then you're back to local.
