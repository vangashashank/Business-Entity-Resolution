"""
ML Challenge 2026 — Business Entity Resolution
Script: evaluate_candidate_recall.py

Evaluates the candidate recall of the blocking pipeline on the validation split:
1. Loads dataset/val_split_ids.txt.
2. Filters train_ground_truth.tsv for validation S1 entities.
3. Runs the 3-strategy blocking pipeline (TF-IDF char n-grams, Address/token, Phonetic).
4. Measures Candidate Recall:
   Recall = (Recovered true matches in candidate set) / (Total true ground-truth matches)
5. Prints the metric clearly along with candidate set statistics.
"""

import sys
import os
import time
import argparse
from typing import Dict, Set, List
import pandas as pd
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.preprocessing import preprocess_dataframe
from src.blocking import (
    generate_tfidf_candidates,
    generate_address_candidates,
    generate_phonetic_candidates,
    union_and_dedupe_candidates,
    save_candidate_pairs,
)


def evaluate_candidate_recall(
    sample_size: int = 5000,
    random_state: int = 42,
    output_path: str = "output/candidate_pairs_val.tsv",
    top_k_tfidf: int = 20,
) -> float:
    t_start = time.time()
    print("=" * 70)
    print("CANDIDATE RECALL EVALUATION ON VALIDATION SPLIT")
    print("=" * 70)

    # 1. Load validation IDs
    val_split_path = "dataset/val_split_ids.txt"
    if not os.path.exists(val_split_path):
        raise FileNotFoundError(f"Missing {val_split_path}! Run EDA first.")

    with open(val_split_path, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]
    val_ids_set = set(val_ids)
    print(f"Total locked validation S1 entities: {len(val_ids):,}")

    # 2. Load ground truth
    gt_path = "dataset/train/train_ground_truth.tsv"
    print(f"Loading {gt_path} (sep='\\t')...")
    gt = pd.read_csv(gt_path, sep="\t")

    # Filter to validation S1 entities
    gt_val = gt[gt["source1_entity_id"].isin(val_ids_set)].copy()
    gt_val_with_matches = gt_val.dropna(subset=["matched_entity_ids"]).copy()

    # Sample S1 entities if requested
    if sample_size and sample_size < len(gt_val_with_matches):
        print(f"Selecting representative sample of {sample_size:,} validation S1 entities...")
        eval_gt = gt_val_with_matches.sample(n=sample_size, random_state=random_state)
    else:
        print(f"Evaluating on all {len(gt_val_with_matches):,} validation S1 entities with matches...")
        eval_gt = gt_val_with_matches

    eval_s1_ids = set(eval_gt["source1_entity_id"])
    print(f"Entities in evaluation set: {len(eval_s1_ids):,}")

    # Build target ground truth mapping
    ground_truth_map: Dict[str, Set[str]] = {}
    all_needed_matches: Set[str] = set()
    for _, row in eval_gt.iterrows():
        sid = row["source1_entity_id"]
        mids = set(x.strip() for x in str(row["matched_entity_ids"]).split(",") if x.strip())
        ground_truth_map[sid] = mids
        all_needed_matches.update(mids)

    total_true_matches = sum(len(v) for v in ground_truth_map.values())
    print(f"Total true ground-truth match pairs to retrieve: {total_true_matches:,}")

    # 3. Load S1 data
    print("\nLoading Source 1 records...")
    s1_all = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s1_eval = s1_all[s1_all["entity_id"].isin(eval_s1_ids)].copy()
    s1_eval = preprocess_dataframe(s1_eval)

    # 4. Load S2 and S3 candidate pool
    print("\nLoading Source 2 and Source 3 candidate pool...")
    s2_all = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3_all = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")

    # The pool must contain all true matches + background negatives
    s2_targets = s2_all[s2_all["entity_id"].isin(all_needed_matches)]
    s3_targets = s3_all[s3_all["entity_id"].isin(all_needed_matches)]

    # Sample realistic background noise (50,000 from each source to form a large realistic candidate pool)
    n_neg = min(100000, len(s2_all))
    s2_neg = s2_all.sample(n=n_neg, random_state=random_state)
    s3_neg = s3_all.sample(n=n_neg, random_state=random_state)

    pool_df = pd.concat([s2_targets, s3_targets, s2_neg, s3_neg]).drop_duplicates(subset=["entity_id"]).copy()
    print(f"Total candidate pool records loaded: {len(pool_df):,} (includes all {len(all_needed_matches):,} true matches)")
    pool_df = preprocess_dataframe(pool_df)

    # 5. Run Candidate Generation Partitioned by Country
    all_candidates: Dict[str, List[str]] = {}
    strategy_recalls = {"tfidf": 0, "address": 0, "phonetic": 0}

    unique_countries = s1_eval["country"].unique()
    for country in unique_countries:
        sub_s1 = s1_eval[s1_eval["country"] == country]
        sub_pool = pool_df[pool_df["country"] == country]

        s1_ids = list(sub_s1["entity_id"].values)
        s1_names = list(sub_s1["business_name_norm"].values)
        s1_addrs = list(sub_s1["business_address_norm"].values)

        pool_ids = list(sub_pool["entity_id"].values)
        pool_names = list(sub_pool["business_name_norm"].values)
        pool_addrs = list(sub_pool["business_address_norm"].values)

        print(f"\n--- Running Blocking for Country: {country} ---")
        print(f"S1 Queries: {len(s1_ids):,} | Pool Records: {len(pool_ids):,}")

        # Strategy 1: TF-IDF char n-grams
        t0 = time.time()
        tfidf_cands = generate_tfidf_candidates(s1_names, s1_ids, pool_names, pool_ids, top_k=top_k_tfidf)
        t_tfidf = time.time() - t0
        hits_tf = sum(len(ground_truth_map[sid] & tfidf_cands[sid]) for sid in s1_ids if sid in ground_truth_map)
        strategy_recalls["tfidf"] += hits_tf
        print(f"  -> TF-IDF Recall ({country}): {hits_tf} matches recovered in {t_tfidf:.2f}s")

        # Strategy 2: Address / Token
        t0 = time.time()
        addr_cands = generate_address_candidates(s1_addrs, s1_ids, pool_addrs, pool_ids)
        t_addr = time.time() - t0
        hits_ad = sum(len(ground_truth_map[sid] & addr_cands[sid]) for sid in s1_ids if sid in ground_truth_map)
        strategy_recalls["address"] += hits_ad
        print(f"  -> Address Recall ({country}): {hits_ad} matches recovered in {t_addr:.2f}s")

        # Strategy 3: Phonetic
        t0 = time.time()
        phonetic_cands = generate_phonetic_candidates(s1_names, s1_ids, pool_names, pool_ids)
        t_ph = time.time() - t0
        hits_ph = sum(len(ground_truth_map[sid] & phonetic_cands[sid]) for sid in s1_ids if sid in ground_truth_map)
        strategy_recalls["phonetic"] += hits_ph
        print(f"  -> Phonetic Recall ({country}): {hits_ph} matches recovered in {t_ph:.2f}s")

        # Combine
        combined_country = union_and_dedupe_candidates(s1_ids, tfidf_cands, addr_cands, phonetic_cands)
        all_candidates.update(combined_country)

    # 6. Save Candidate Pairs
    if output_path:
        save_candidate_pairs(all_candidates, output_path, s1_id_order=list(s1_eval["entity_id"].values))

    # 7. Final Recall Calculation
    recovered_matches = sum(
        len(ground_truth_map[sid] & set(all_candidates.get(sid, [])))
        for sid in eval_s1_ids
        if sid in ground_truth_map
    )

    candidate_recall_pct = (recovered_matches / total_true_matches) * 100.0
    total_candidates_generated = sum(len(v) for v in all_candidates.values())
    avg_candidates_per_entity = total_candidates_generated / max(len(eval_s1_ids), 1)

    print("\n" + "=" * 70)
    print("FINAL CANDIDATE GENERATION RECALL RESULTS")
    print("=" * 70)
    print(f"Validation S1 Entities Evaluated:      {len(eval_s1_ids):,}")
    print(f"Total True Ground-Truth Match Pairs:   {total_true_matches:,}")
    print(f"True Matches Recovered by Blocking:    {recovered_matches:,}")
    print(f"\n>>> CANDIDATE RECALL:                  {candidate_recall_pct:.2f}% <<<\n")
    print(f"Individual Strategy Recoveries:")
    print(f"  - TF-IDF Char n-grams (top-{top_k_tfidf}):     {strategy_recalls['tfidf'] / total_true_matches * 100:.2f}% ({strategy_recalls['tfidf']:,}/{total_true_matches:,})")
    print(f"  - Address / Token Blocking:          {strategy_recalls['address'] / total_true_matches * 100:.2f}% ({strategy_recalls['address']:,}/{total_true_matches:,})")
    print(f"  - Phonetic Blocking (Metaphone):     {strategy_recalls['phonetic'] / total_true_matches * 100:.2f}% ({strategy_recalls['phonetic']:,}/{total_true_matches:,})")
    print(f"\nTotal Candidates Generated:            {total_candidates_generated:,}")
    print(f"Average Candidates per S1 Entity:      {avg_candidates_per_entity:.1f}")
    print(f"Output File:                           {output_path}")
    print(f"Total Evaluation Time:                 {time.time()-t_start:.2f}s")
    print("=" * 70)

    return candidate_recall_pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Candidate Recall on Validation Split")
    parser.add_argument("--sample", type=int, default=2500, help="Number of validation S1 entities to evaluate (default: 2500)")
    parser.add_argument("--output", type=str, default="output/candidate_pairs_val.tsv", help="Path to save candidate pairs")
    parser.add_argument("--top_k_tfidf", type=int, default=20, help="Top-k for TF-IDF char n-grams (default: 20)")
    args = parser.parse_args()

    evaluate_candidate_recall(
        sample_size=args.sample,
        output_path=args.output,
        top_k_tfidf=args.top_k_tfidf,
    )
