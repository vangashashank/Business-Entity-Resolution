"""Generate submission candidates and scored match predictions."""

from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.blocking import run_blocking_pipeline  # noqa: E402
from src.preprocessing import preprocess_dataframe  # noqa: E402

TEST_DIR = os.path.join(ROOT, "student_resource", "dataset", "test")


def _read_test(name: str) -> pd.DataFrame:
    path = os.path.join(TEST_DIR, name)
    frame = pd.read_csv(
        path, sep="\t",
        dtype={"entity_id": str, "business_name": str,
               "business_address": str, "country": str},
    )
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if frame.entity_id.isna().any():
        raise ValueError(f"{path} has null entity_id values")
    return frame


def run_inference(model_path: str = "artifacts/random_forest_v2.joblib",
                  candidate_path: str = "output/candidate_pairs.tsv",
                  matching_path: str = "output/matching_results.tsv",
                  sample_size: int = 0, pool_sample_size: int = 0,
                  row_batch_size: int = 1000, threshold: float = 0.70):
    started = time.time()
    s1 = _read_test("test_source1.tsv")
    s2 = _read_test("test_source2.tsv")
    s3 = _read_test("test_source3.tsv")
    if sample_size and sample_size < len(s1):
        s1 = s1.sample(n=sample_size, random_state=42).copy()
    if pool_sample_size:
        s2 = s2.sample(n=min(pool_sample_size, len(s2)), random_state=42).copy()
        s3 = s3.sample(n=min(pool_sample_size, len(s3)), random_state=42).copy()

    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    candidate_path_abs = os.path.join(ROOT, candidate_path)
    matching_path_abs = os.path.join(ROOT, matching_path)
    os.makedirs(os.path.dirname(candidate_path_abs), exist_ok=True)
    os.makedirs(os.path.dirname(matching_path_abs), exist_ok=True)

    print(f"Generating candidates for {len(s1):,} S1 records...")
    run_blocking_pipeline(s1, s2, s3, output_path=candidate_path_abs)
    artifact = joblib.load(os.path.join(ROOT, model_path))
    extractor, model = artifact["feature_extractor"], artifact["model"]
    s1_index = s1.drop_duplicates("entity_id").set_index("entity_id")
    pool = pd.concat([s2, s3], ignore_index=True).drop_duplicates("entity_id")

    threshold_value = float(threshold)
    print(f"Using match threshold {threshold_value:.4f}")
    rows_written = 0
    with open(matching_path_abs, "w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for chunk in pd.read_csv(
            candidate_path_abs, sep="\t", dtype=str, chunksize=row_batch_size,
            keep_default_na=False,
        ):
            pair_sids, pair_mids = [], []
            chunk_sids = chunk.source1_entity_id.tolist()
            chunk_matches = {sid: [] for sid in chunk_sids}
            for sid, raw_ids in zip(chunk_sids, chunk.candidate_entity_ids):
                for mid in raw_ids.split(",") if raw_ids else ():
                    pair_sids.append(sid)
                    pair_mids.append(mid)

            if pair_sids:
                pairs = pd.DataFrame({
                    "source1_entity_id": pair_sids,
                    "candidate_entity_id": pair_mids,
                })
                features = extractor.transform(s1, pool, pairs)
                probabilities = model.predict_proba(features)[:, 1]
                for sid, mid, probability in zip(pair_sids, pair_mids, probabilities):
                    if probability >= threshold_value:
                        chunk_matches[sid].append(mid)

            for sid in chunk_sids:
                # Candidate IDs are already unique and stable from the blocking union.
                matches = sorted(set(chunk_matches[sid]))
                out.write(f"{sid}\t{','.join(matches)}\n")
                rows_written += 1
            if rows_written and rows_written % 100000 == 0:
                print(f"Scored {rows_written:,} S1 entities...")

    print(f"Wrote {rows_written:,} matching rows to {os.path.relpath(matching_path_abs, ROOT)}")
    print(f"Candidate file: {os.path.relpath(candidate_path_abs, ROOT)}")
    print(f"Inference elapsed: {time.time() - started:.1f}s")
    return matching_path_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/random_forest_v2.joblib")
    parser.add_argument("--candidate-output", default="output/candidate_pairs.tsv")
    parser.add_argument("--matching-output", default="output/matching_results.tsv")
    parser.add_argument("--sample", type=int, default=0,
                        help="Small-subset S1 smoke run; 0 means the full test set")
    parser.add_argument("--pool-sample", type=int, default=0,
                        help="Optional per-source target pool sample size (0 means full pool)")
    parser.add_argument("--row-batch-size", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    run_inference(args.model, args.candidate_output, args.matching_output,
                  args.sample, args.pool_sample, args.row_batch_size, args.threshold)
