"""Train the v1 balanced Logistic Regression candidate matcher.

By default the training run samples 2,500 matched S1 records outside the locked
validation split, then generates blocking candidates against a seeded pool with
all sampled true matches plus 100,000 background records from each target source.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Set

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.blocking import (  # noqa: E402
    generate_address_candidates,
    generate_phonetic_candidates,
    generate_tfidf_candidates,
    union_and_dedupe_candidates,
)
from src.features import PairFeatureExtractor  # noqa: E402
from src.preprocessing import preprocess_dataframe  # noqa: E402

DATA_DIR = os.path.join(ROOT, "student_resource", "dataset", "train")


def _read(name: str, usecols=None) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_DIR, name), sep="\t", usecols=usecols,
        dtype={"entity_id": str, "source1_entity_id": str,
               "matched_entity_ids": str, "country": str},
    )


def build_training_pairs(sample_size: int = 2500, random_state: int = 42,
                         background_per_source: int = 100000,
                         top_k_tfidf: int = 20):
    gt = _read("train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    val_path = os.path.join(ROOT, "student_resource", "dataset", "val_split_ids.txt")
    with open(val_path, encoding="utf-8") as f:
        validation_ids = {line.strip() for line in f if line.strip()}
    gt_train = gt[~gt.source1_entity_id.isin(validation_ids)].dropna(subset=["matched_entity_ids"])
    if sample_size and sample_size < len(gt_train):
        gt_sample = gt_train.sample(n=sample_size, random_state=random_state)
    else:
        gt_sample = gt_train

    truth: Dict[str, Set[str]] = {}
    needed = set()
    for sid, value in zip(gt_sample.source1_entity_id, gt_sample.matched_entity_ids):
        mids = {item.strip() for item in str(value).split(",") if item.strip()}
        truth[sid] = mids
        needed.update(mids)

    s1_all = _read("train_source1.tsv")
    s1 = preprocess_dataframe(s1_all[s1_all.entity_id.isin(truth)].copy())
    s2_all = _read("train_source2.tsv")
    s3_all = _read("train_source3.tsv")
    s2_targets = s2_all[s2_all.entity_id.isin(needed)]
    s3_targets = s3_all[s3_all.entity_id.isin(needed)]
    n2 = min(background_per_source, len(s2_all))
    n3 = min(background_per_source, len(s3_all))
    pool = pd.concat([
        s2_targets, s3_targets,
        s2_all.sample(n=n2, random_state=random_state),
        s3_all.sample(n=n3, random_state=random_state),
    ]).drop_duplicates(subset=["entity_id"]).copy()
    pool = preprocess_dataframe(pool)

    candidates: Dict[str, list[str]] = {}
    for country in s1.country.drop_duplicates():
        q = s1[s1.country == country]
        p = pool[pool.country == country]
        qids, pids = q.entity_id.tolist(), p.entity_id.tolist()
        tfidf = generate_tfidf_candidates(
            q.business_name_norm.tolist(), qids,
            p.business_name_norm.tolist(), pids, top_k=top_k_tfidf,
        )
        address = generate_address_candidates(
            q.business_address_norm.tolist(), qids,
            p.business_address_norm.tolist(), pids,
        )
        phonetic = generate_phonetic_candidates(
            q.business_name_norm.tolist(), qids,
            p.business_name_norm.tolist(), pids,
        )
        candidates.update(union_and_dedupe_candidates(qids, tfidf, address, phonetic))

    pair_rows = [
        (sid, mid, int(mid in truth.get(sid, set())))
        for sid, mids in candidates.items() for mid in mids
    ]
    pairs = pd.DataFrame(pair_rows, columns=["source1_entity_id", "candidate_entity_id", "label"])
    if pairs.empty or pairs.label.nunique() < 2:
        raise ValueError("Training sample must produce both positive and negative candidate pairs")
    return s1, pool, pairs


def train_model(sample_size: int = 2500, random_state: int = 42,
                background_per_source: int = 100000,
                model_path: str = "artifacts/baseline_logreg.joblib"):
    s1, pool, pairs = build_training_pairs(sample_size, random_state, background_per_source)
    print(f"Training candidate pairs: {len(pairs):,}")
    print(f"Positive pairs: {int(pairs.label.sum()):,}; negative pairs: {int((pairs.label == 0).sum()):,}")

    extractor = PairFeatureExtractor().fit(s1, pool, pairs)
    features = extractor.transform(s1, pool, pairs)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
    )
    classifier.fit(features, pairs.label.to_numpy())

    artifact = os.path.join(ROOT, model_path)
    os.makedirs(os.path.dirname(artifact), exist_ok=True)
    joblib.dump({"feature_extractor": extractor, "model": classifier}, artifact)
    print(f"Saved baseline model to {os.path.relpath(artifact, ROOT)}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=2500,
                        help="Matched S1 training entities; 0 means all non-validation matched entities")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-per-source", type=int, default=100000)
    parser.add_argument("--model", default="artifacts/baseline_logreg.joblib")
    args = parser.parse_args()
    train_model(args.sample, args.seed, args.background_per_source, args.model)

