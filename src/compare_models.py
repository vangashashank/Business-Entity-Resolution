"""Train v2 models and compare them with the saved v1 Logistic Regression."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import PairFeatureExtractor  # noqa: E402
from src.scoring import macro_metrics  # noqa: E402
from src.train import build_training_pairs  # noqa: E402

TRAIN_DIR = os.path.join(ROOT, "student_resource", "dataset", "train")


def _read(name, usecols=None):
    return pd.read_csv(
        os.path.join(TRAIN_DIR, name), sep="\t", usecols=usecols,
        dtype={"entity_id": str, "source1_entity_id": str, "matched_entity_ids": str,
               "candidate_entity_ids": str, "country": str},
    )


def load_validation_sample(candidate_file: str):
    candidate_path = os.path.join(ROOT, candidate_file)
    candidate_rows = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    sids = set(candidate_rows.source1_entity_id)
    needed = {mid for value in candidate_rows.candidate_entity_ids
              for mid in value.split(",") if mid}
    gt = _read("train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt.source1_entity_id.isin(sids)].copy()
    if set(gt.source1_entity_id) != sids:
        raise ValueError("Validation candidate rows and ground truth S1 IDs do not match")

    s1_all = _read("train_source1.tsv")
    s1 = s1_all[s1_all.entity_id.isin(sids)].copy()
    s2 = _read("train_source2.tsv")
    s3 = _read("train_source3.tsv")
    pool = pd.concat([
        s2[s2.entity_id.isin(needed)],
        s3[s3.entity_id.isin(needed)],
    ]).drop_duplicates("entity_id").copy()
    from src.preprocessing import preprocess_dataframe
    return (preprocess_dataframe(s1), preprocess_dataframe(pool),
            candidate_rows, gt)


def _prediction_frame(candidate_rows, pairs, probabilities, threshold):
    matched = {sid: [] for sid in candidate_rows.source1_entity_id}
    for sid, mid, probability in zip(
        pairs.source1_entity_id, pairs.candidate_entity_id, probabilities
    ):
        if probability >= threshold:
            matched[sid].append(mid)
    return pd.DataFrame({
        "source1_entity_id": list(matched),
        "matched_entity_ids": [",".join(sorted(set(matched[sid]))) for sid in matched],
    })


def _candidate_recall(candidate_rows, ground_truth):
    candidate_map = {
        row.source1_entity_id: {x for x in row.candidate_entity_ids.split(",") if x}
        for row in candidate_rows.itertuples(index=False)
    }
    hits = total = 0
    for row in ground_truth.itertuples(index=False):
        truth = (set() if pd.isna(row.matched_entity_ids) else
                 {x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()})
        total += len(truth)
        hits += len(truth & candidate_map.get(row.source1_entity_id, set()))
    return hits / total if total else 0.0


def _evaluate(name, model, extractor, s1, pool, pairs, candidates, truth, recall, threshold):
    x = extractor.transform(s1, pool, pairs)
    probabilities = model.predict_proba(x)[:, 1]
    predictions = _prediction_frame(candidates, pairs, probabilities, threshold)
    metrics = macro_metrics(predictions, truth)
    result = {
        "change": name,
        "candidate_recall": recall,
        "val_f0_5": metrics["val_f0_5"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "notes": (f"Threshold={threshold}; {len(candidates):,} held-out S1 entities, "
                  f"including {int(truth.matched_entity_ids.isna().sum())} singletons."),
        "date": date.today().isoformat(),
    }
    print("{change}: candidate recall={candidate_recall:.4%}, val F0.5={val_f0_5:.4f}, "
          "precision={precision:.4f}, recall={recall:.4f}".format(**result))
    return result


def _write_experiments(results, path="experiments.md"):
    full_path = os.path.join(ROOT, path)
    if os.path.isfile(full_path):
        with open(full_path, encoding="utf-8") as f:
            old = f.read()
    else:
        old = "# Experiments\n\n"
    header = "| change | candidate_recall | val_F0.5 | precision | recall | notes | date |\n"
    separator = "|---|---:|---:|---:|---:|---|---|\n"
    lines = []
    for r in results:
        notes = str(r["notes"]).replace("|", "\\|")
        lines.append(
            f"| {r['change']} | {r['candidate_recall']:.4%} | {r['val_f0_5']:.6f} | "
            f"{r['precision']:.6f} | {r['recall']:.6f} | {notes} | {r['date']} |\n"
        )
    # Preserve the setup note while normalizing the table to the requested columns.
    prefix = old.split("\n| change |", 1)[0]
    if prefix.strip() == "# experiments.md":
        prefix = "# Experiments\n\nLog every change and experiment here. Track metrics rigorously to guide pipeline decisions."
    initial = (
        "| Setup / EDA | - | - | - | - | Repository scaffolding and stratified split setup. | 2026-09-26 |\n"
    )
    with open(full_path, "w", encoding="utf-8", newline="") as f:
        f.write(prefix.rstrip() + "\n\n")
        f.write(header)
        f.write(separator)
        f.write(initial)
        f.writelines(lines)


def compare(candidate_file="output/candidate_pairs_val.tsv",
            sample_size=2500, random_state=42, threshold=0.5):
    # Rebuild the same non-validation training sample and v1 training pairs.
    train_s1, train_pool, train_pairs = build_training_pairs(
        sample_size=sample_size, random_state=random_state
    )
    print(f"Training pairs: {len(train_pairs):,}; positives={int(train_pairs.label.sum()):,}")

    val_s1, val_pool, val_candidates, val_truth = load_validation_sample(candidate_file)
    val_pairs = pd.DataFrame(
        [(row.source1_entity_id, mid)
         for row in val_candidates.itertuples(index=False)
         for mid in row.candidate_entity_ids.split(",") if mid],
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    val_recall = _candidate_recall(val_candidates, val_truth)
    print(f"Validation S1: {len(val_candidates):,}; candidate recall={val_recall:.4%}")

    extractor_v1 = PairFeatureExtractor(feature_set="v1").fit(train_s1, train_pool, train_pairs)
    x_train_v1 = extractor_v1.transform(train_s1, train_pool, train_pairs)
    y_train = train_pairs.label.to_numpy()
    baseline_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
    ).fit(x_train_v1, y_train)
    joblib.dump({"feature_extractor": extractor_v1, "model": baseline_model},
                os.path.join(ROOT, "artifacts", "baseline_logreg.joblib"))
    results = [_evaluate(
        "Logistic Regression (feature set v1)", baseline_model, extractor_v1,
        val_s1, val_pool, val_pairs, val_candidates, val_truth, val_recall, threshold,
    )]
    _write_experiments(results)

    extractor_v2 = PairFeatureExtractor(feature_set="v2").fit(train_s1, train_pool, train_pairs)
    x_train = extractor_v2.transform(train_s1, train_pool, train_pairs)
    x_val = extractor_v2.transform(val_s1, val_pool, val_pairs)

    logistic_v2 = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
    ).fit(x_train, y_train)
    results.append(_evaluate(
        "Logistic Regression (feature set v2)", logistic_v2, extractor_v2,
        val_s1, val_pool, val_pairs, val_candidates, val_truth, val_recall, threshold,
    ))
    joblib.dump({"feature_extractor": extractor_v2, "model": logistic_v2},
                os.path.join(ROOT, "artifacts", "logistic_regression_v2.joblib"))
    _write_experiments(results)

    rf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=1, random_state=random_state,
    ).fit(x_train, y_train)
    joblib.dump({"feature_extractor": extractor_v2, "model": rf},
                os.path.join(ROOT, "artifacts", "random_forest_v2.joblib"))
    results.append(_evaluate(
        "Random Forest (feature set v2)", rf, extractor_v2,
        val_s1, val_pool, val_pairs, val_candidates, val_truth, val_recall, threshold,
    ))
    _write_experiments(results)

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is not installed; cannot complete requested comparison") from exc
    xgb = XGBClassifier(
        n_estimators=250, max_depth=8, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_jobs=1, random_state=random_state,
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    ).fit(x_train, y_train)
    joblib.dump({"feature_extractor": extractor_v2, "model": xgb},
                os.path.join(ROOT, "artifacts", "xgboost_v2.joblib"))
    results.append(_evaluate(
        "XGBoost (feature set v2)", xgb, extractor_v2,
        val_s1, val_pool, val_pairs, val_candidates, val_truth, val_recall, threshold,
    ))

    _write_experiments(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", default="output/candidate_pairs_val.tsv")
    parser.add_argument("--sample", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    compare(args.candidate_file, args.sample, args.seed, args.threshold)
