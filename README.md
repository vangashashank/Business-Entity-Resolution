# Amazon ML Challenge 2026 — Business Entity Resolution

This repository implements an end-to-end Entity Resolution (ER) pipeline matching Source 2 and Source 3 business records to reference Source 1 entities, optimizing for macro-averaged F0.5.

## Project Structure
- `student_resource/dataset/`: Organizer-provided training/test TSV files and persistent validation split IDs.
- `notebooks/`: Exploratory Data Analysis (`01_eda.ipynb`).
- `colab_notebooks/`: GPU-dependent steps (e.g. dense sentence embeddings, if needed).
- `src/`: Core pipeline modules (`preprocessing`, `blocking`, `features`, `train`, `inference`, `scoring`, `threshold`).
- `artifacts/`: Exported embeddings, model weights, and intermediate serializations.
- `output/`: Generated submission files (`matching_results.tsv` and `candidate_pairs.tsv`).
- `student_resource/utils/`: Submission validator (`validate_submission.py`).
- `Documentation/`: Full specifications, PRD, architecture, rules, design, and session memory.
- `code/business_entity_resolution/`: Created once from `src/` during Phase 7 packaging.
