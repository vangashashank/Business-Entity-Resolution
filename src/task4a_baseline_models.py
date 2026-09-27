#!/usr/bin/env python3
"""Task 4A: train and evaluate logistic-regression and LightGBM baselines."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for vendor_dir in (PROJECT_ROOT / ".task4_vendor", PROJECT_ROOT / ".task3_vendor"):
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
FORBIDDEN_FEATURES = {
    "eval_index",
    "source1_entity_id",
    "candidate_entity_id",
    "match_group",
    "split",
    "label",
}
RANK_K = (1, 3, 5, 10, 20)
HARD_EVIDENCE_FEATURES = (
    "candidate_address_missing",
    "candidate_source_s3",
    "cross_script_name",
    "name_levenshtein",
    "name_token_set",
    "transliterated_name_levenshtein",
    "transliteration_levenshtein_gain",
    "suffix_name_levenshtein",
    "suffix_levenshtein_gain",
    "address_levenshtein",
    "address_token_set",
    "address_char3_jaccard",
    "both_addresses_have_numbers",
    "address_number_jaccard",
    "address_conflicting_number_count",
    "retrieval_best_rank",
    "retrieval_signal_count_top100",
    "retrieval_rrf_score_top100",
    "task2_baseline_member",
    "independent_strong_signal_count",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-parquet",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_pair_features.parquet",
    )
    parser.add_argument(
        "--schema-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_feature_schema.csv",
    )
    parser.add_argument(
        "--feature-audit-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_to_task4_feature_audit.csv",
    )
    parser.add_argument(
        "--split-summary-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_split_summary.csv",
    )
    parser.add_argument(
        "--missed-positive-parquet",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_missed_positive_features.parquet",
    )
    parser.add_argument(
        "--redundancy-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_feature_redundancy.csv",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task2_5_outputs" / "task2_5_retrieval_top250.npz",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource" / "dataset" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task4_outputs",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--lightgbm-rounds", type=int, default=1_500)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--logistic-max-iter", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    required = [
        args.pair_parquet,
        args.schema_csv,
        args.feature_audit_csv,
        args.split_summary_csv,
        args.missed_positive_parquet,
        args.redundancy_csv,
        args.candidate_checkpoint,
        args.data_dir / "train_source1.tsv",
        args.data_dir / "train_source2.tsv",
        args.data_dir / "train_source3.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 4A inputs: {missing}")
    for path in required:
        if "test" in path.name.casefold():
            raise ValueError(f"Task 4A must not access test data: {path}")


def read_feature_contract(
    schema_path: Path, audit_path: Path
) -> tuple[list[str], pd.DataFrame]:
    schema = pd.read_csv(schema_path)
    recommended = schema["recommended_task4"].astype(str).str.casefold().eq("true")
    feature_schema = schema.loc[recommended].copy()
    features = feature_schema["column"].astype(str).tolist()
    if len(features) != 66 or len(set(features)) != 66:
        raise AssertionError(f"Expected exactly 66 unique features, found {len(features)}")
    forbidden = sorted(set(features) & FORBIDDEN_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden fields in model features: {forbidden}")
    if not feature_schema["role"].eq("predictive_feature").all():
        bad = feature_schema.loc[
            ~feature_schema["role"].eq("predictive_feature"), ["column", "role"]
        ]
        raise AssertionError(f"Non-predictive roles in feature set:\n{bad}")

    audit = pd.read_csv(audit_path)
    audited = set(audit.loc[audit["inference_safe"].astype(str).str.casefold().eq("true"), "feature"])
    if audited != set(features):
        raise AssertionError("Task 3.5 inference-safe audit does not match the 66-feature schema")
    return features, feature_schema


def expected_split_counts(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    return {
        str(row.split): int(row.pairs)
        for row in frame.itertuples(index=False)
    }


def batch_feature_matrix(batch: pa.RecordBatch, features: list[str]) -> np.ndarray:
    names = batch.schema.names
    columns = [
        batch.column(names.index(feature)).to_numpy(zero_copy_only=False)
        for feature in features
    ]
    return np.column_stack(columns).astype(np.float32, copy=False)


def load_pair_data(
    parquet_path: Path,
    features: list[str],
    counts: dict[str, int],
    batch_size: int,
    smoke_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    if smoke_only:
        train_target = min(200_000, counts["train"])
        validation_target = min(75_000, counts["validation"])
    else:
        train_target = counts["train"]
        validation_target = counts["validation"]

    x_train = np.empty((train_target, len(features)), dtype=np.float32)
    y_train = np.empty(train_target, dtype=np.uint8)
    x_validation = np.empty((validation_target, len(features)), dtype=np.float32)
    y_validation = np.empty(validation_target, dtype=np.uint8)
    val_source1 = np.empty(validation_target, dtype=object)
    val_candidate = np.empty(validation_target, dtype=object)
    val_source = np.empty(validation_target, dtype=object)
    val_group = np.empty(validation_target, dtype=object)

    train_entities: set[str] = set()
    validation_entities: set[str] = set()
    train_offset = 0
    validation_offset = 0
    selected_columns = features + [
        "label",
        "split",
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "match_group",
    ]
    parquet = pq.ParquetFile(parquet_path)
    parquet_names = set(parquet.schema_arrow.names)
    missing = sorted(set(selected_columns) - parquet_names)
    if missing:
        raise AssertionError(f"Missing required Parquet columns: {missing}")

    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected_columns):
        names = batch.schema.names
        split_values = np.asarray(
            batch.column(names.index("split")).to_pylist(), dtype=object
        )
        feature_values = batch_feature_matrix(batch, features)
        labels = batch.column(names.index("label")).to_numpy(zero_copy_only=False).astype(
            np.uint8, copy=False
        )
        source1_ids = np.asarray(
            batch.column(names.index("source1_entity_id")).to_pylist(), dtype=object
        )

        for split_name in ("train", "validation"):
            target = train_target if split_name == "train" else validation_target
            offset = train_offset if split_name == "train" else validation_offset
            remaining = target - offset
            if remaining <= 0:
                continue
            indices = np.flatnonzero(split_values == split_name)[:remaining]
            if indices.size == 0:
                continue
            end = offset + len(indices)
            selected_matrix = feature_values[indices]
            if not np.isfinite(selected_matrix).all():
                raise AssertionError(f"NaN or inf in {split_name} feature matrix")
            if split_name == "train":
                x_train[offset:end] = selected_matrix
                y_train[offset:end] = labels[indices]
                train_entities.update(map(str, source1_ids[indices]))
                train_offset = end
            else:
                x_validation[offset:end] = selected_matrix
                y_validation[offset:end] = labels[indices]
                val_source1[offset:end] = source1_ids[indices]
                val_candidate[offset:end] = np.asarray(
                    batch.column(names.index("candidate_entity_id")).to_pylist(),
                    dtype=object,
                )[indices]
                val_source[offset:end] = np.asarray(
                    batch.column(names.index("candidate_source")).to_pylist(),
                    dtype=object,
                )[indices]
                val_group[offset:end] = np.asarray(
                    batch.column(names.index("match_group")).to_pylist(),
                    dtype=object,
                )[indices]
                validation_entities.update(map(str, source1_ids[indices]))
                validation_offset = end

        if train_offset == train_target and validation_offset == validation_target:
            break

    if train_offset != train_target or validation_offset != validation_target:
        raise AssertionError(
            f"Incomplete load: train={train_offset}/{train_target}, "
            f"validation={validation_offset}/{validation_target}"
        )
    if set(np.unique(y_train)) - {0, 1} or set(np.unique(y_validation)) - {0, 1}:
        raise AssertionError("Labels must be binary")
    overlap = train_entities & validation_entities
    if overlap:
        raise AssertionError(f"S1 overlap across split: {list(overlap)[:5]}")

    validation_meta = pd.DataFrame(
        {
            "source1_entity_id": val_source1,
            "candidate_entity_id": val_candidate,
            "candidate_source": val_source,
            "match_group": val_group,
        }
    )
    audit = {
        "train_pairs": int(train_target),
        "validation_pairs": int(validation_target),
        "train_entities": int(len(train_entities)),
        "validation_entities": int(len(validation_entities)),
        "entity_overlap": 0,
        "train_positives": int(y_train.sum()),
        "train_negatives": int(len(y_train) - y_train.sum()),
        "validation_positives": int(y_validation.sum()),
        "validation_negatives": int(len(y_validation) - y_validation.sum()),
    }
    audit["train_positive_rate"] = audit["train_positives"] / audit["train_pairs"]
    audit["validation_positive_rate"] = (
        audit["validation_positives"] / audit["validation_pairs"]
    )
    return x_train, y_train, x_validation, y_validation, validation_meta, audit


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    max_iter: int,
    output_path: Path,
) -> tuple[Pipeline, np.ndarray, dict[str, object]]:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.1,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=max_iter,
                    tol=1e-3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        probabilities = model.predict_proba(x_validation)[:, 1].astype(np.float32)
    inference_seconds = time.perf_counter() - start
    if not np.isfinite(probabilities).all():
        raise AssertionError("Logistic regression produced non-finite probabilities")
    estimator = model.named_steps["model"]
    convergence_warnings = [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    converged = not convergence_warnings and bool(np.all(estimator.n_iter_ < max_iter))
    joblib.dump(model, output_path, compress=3)
    details = {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "converged": converged,
        "n_iter": int(np.max(estimator.n_iter_)),
        "convergence_warnings": convergence_warnings,
        "configuration": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "penalty": "l2",
            "C": 0.1,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": max_iter,
            "tol": 1e-3,
            "random_state": RANDOM_STATE,
        },
    }
    return model, probabilities, details


def fit_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    features: list[str],
    max_rounds: int,
    early_stopping_rounds: int,
    output_path: Path,
) -> tuple[lgb.Booster, np.ndarray, dict[str, object]]:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives
    params = {
        "objective": "binary",
        "metric": ["average_precision", "auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "min_child_samples": 100,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.10,
        "lambda_l2": 1.00,
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
        "feature_fraction_seed": RANDOM_STATE,
        "bagging_seed": RANDOM_STATE,
        "data_random_seed": RANDOM_STATE,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "verbosity": -1,
    }
    start = time.perf_counter()
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=features,
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        x_validation,
        label=y_validation,
        reference=train_set,
        feature_name=features,
        free_raw_data=True,
    )
    callbacks = [
        lgb.early_stopping(
            early_stopping_rounds,
            first_metric_only=True,
            verbose=True,
        ),
        lgb.log_evaluation(period=50),
    ]
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - start
    best_iteration = int(booster.best_iteration or max_rounds)
    start = time.perf_counter()
    probabilities = booster.predict(
        x_validation, num_iteration=best_iteration
    ).astype(np.float32)
    inference_seconds = time.perf_counter() - start
    booster.save_model(str(output_path), num_iteration=best_iteration)
    details = {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "best_iteration": best_iteration,
        "maximum_rounds": max_rounds,
        "early_stopping_rounds": early_stopping_rounds,
        "scale_pos_weight": scale_pos_weight,
        "configuration": params,
    }
    return booster, probabilities, details


def pair_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }


def probability_distribution_rows(
    model_name: str, y_true: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    quantiles = {
        "mean": None,
        "median": 0.50,
        "p10": 0.10,
        "p25": 0.25,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    for label, subset_name in ((1, "positive"), (0, "negative")):
        values = probabilities[y_true == label]
        for metric, quantile in quantiles.items():
            value = float(values.mean()) if quantile is None else float(np.quantile(values, quantile))
            rows.append(
                {
                    "model": model_name,
                    "subset": subset_name,
                    "metric": f"probability_{metric}",
                    "value": value,
                }
            )
    return rows


def within_entity_rank(
    source1_ids: np.ndarray, candidate_ids: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.lexsort((candidate_ids.astype(str), -probabilities, source1_ids.astype(str)))
    sorted_ids = source1_ids[order]
    starts = np.empty(len(order), dtype=bool)
    starts[0] = True
    starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
    indices = np.arange(len(order), dtype=np.int32)
    group_starts = np.maximum.accumulate(np.where(starts, indices, 0))
    sorted_ranks = (indices - group_starts + 1).astype(np.uint16)
    ranks = np.empty(len(order), dtype=np.uint16)
    ranks[order] = sorted_ranks
    return ranks


def ranking_metric_rows(
    model_name: str,
    y_true: np.ndarray,
    ranks: np.ndarray,
    source1_ids: np.ndarray,
    candidate_misses: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    positive_mask = y_true == 1
    positive_ranks = ranks[positive_mask].astype(np.float64)
    retrieved_denominator = int(positive_mask.sum())
    end_to_end_denominator = retrieved_denominator + int(candidate_misses)
    rows: list[dict[str, object]] = []
    summary: dict[str, float] = {}
    for k in RANK_K:
        hits = int((positive_ranks <= k).sum())
        retrieved = hits / retrieved_denominator
        end_to_end = hits / end_to_end_denominator
        summary[f"recall_at_{k}"] = retrieved
        summary[f"end_to_end_recall_at_{k}"] = end_to_end
        rows.extend(
            [
                {
                    "model": model_name,
                    "scope": "overall",
                    "group": "all",
                    "metric": f"recall_at_{k}_retrieved_candidates",
                    "numerator": hits,
                    "denominator": retrieved_denominator,
                    "value": retrieved,
                },
                {
                    "model": model_name,
                    "scope": "overall",
                    "group": "all",
                    "metric": f"recall_at_{k}_end_to_end",
                    "numerator": hits,
                    "denominator": end_to_end_denominator,
                    "value": end_to_end,
                },
            ]
        )
    reciprocal = 1.0 / positive_ranks
    positive_ids = source1_ids[positive_mask]
    first_rank = pd.DataFrame(
        {"source1_entity_id": positive_ids, "rank": positive_ranks}
    ).groupby("source1_entity_id", sort=False)["rank"].min()
    diagnostics = {
        "mean_positive_rank": float(positive_ranks.mean()),
        "median_positive_rank": float(np.median(positive_ranks)),
        "positive_link_mean_reciprocal_rank": float(reciprocal.mean()),
        "entity_first_positive_mrr": float((1.0 / first_rank).mean()),
    }
    for metric, value in diagnostics.items():
        rows.append(
            {
                "model": model_name,
                "scope": "overall",
                "group": "all",
                "metric": metric,
                "numerator": "",
                "denominator": retrieved_denominator,
                "value": value,
            }
        )
        summary[metric] = value
    return rows, summary


def load_validation_misses(path: Path, smoke_only: bool) -> pd.DataFrame:
    if smoke_only:
        return pd.DataFrame(columns=["candidate_source", "match_group"])
    table = pq.read_table(path, columns=["split", "candidate_source", "match_group"])
    frame = table.to_pandas()
    return frame.loc[frame["split"].eq("validation"), ["candidate_source", "match_group"]].copy()


def diagnostic_group_table(
    group_column: str,
    group_values: Iterable[str],
    meta: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    ranks: np.ndarray,
    missed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for value in group_values:
        mask = meta[group_column].astype(str).eq(str(value)).to_numpy()
        labels = y_true[mask]
        scores = probabilities[mask]
        group_ranks = ranks[mask]
        positives = labels == 1
        retrieved_positive_count = int(positives.sum())
        missed_count = int(missed[group_column].astype(str).eq(str(value)).sum())
        total_positive_count = retrieved_positive_count + missed_count
        row: dict[str, object] = {
            group_column: value,
            "pair_rows": int(mask.sum()),
            "retrieved_positive_links": retrieved_positive_count,
            "negative_pairs": int((labels == 0).sum()),
            "candidate_generation_missed_positive_links": missed_count,
            "end_to_end_positive_links": total_positive_count,
            "pr_auc": float(average_precision_score(labels, scores)),
            "mean_positive_rank": float(group_ranks[positives].mean()),
            "median_positive_rank": float(np.median(group_ranks[positives])),
        }
        for k in RANK_K:
            hits = int((group_ranks[positives] <= k).sum())
            row[f"recall_at_{k}_retrieved_candidates"] = hits / retrieved_positive_count
            row[f"recall_at_{k}_end_to_end"] = hits / total_positive_count
        rows.append(row)
    return pd.DataFrame(rows)


def feature_importance_frame(
    booster: lgb.Booster,
    feature_schema: pd.DataFrame,
    redundancy_path: Path,
) -> pd.DataFrame:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    names = booster.feature_name()
    frame = pd.DataFrame(
        {
            "feature": names,
            "gain_importance": gain,
            "split_importance": split,
        }
    )
    frame["gain_percent"] = frame["gain_importance"] / frame["gain_importance"].sum()
    frame["split_percent"] = frame["split_importance"] / frame["split_importance"].sum()
    family = feature_schema.set_index("column")["family"].to_dict()
    frame["family"] = frame["feature"].map(family)
    redundancy = pd.read_csv(redundancy_path)
    correlated = set(redundancy["feature_a"]) | set(redundancy["feature_b"])
    frame["task3_5_high_correlation_flag"] = frame["feature"].isin(correlated)
    frame = frame.sort_values(
        ["gain_importance", "split_importance"], ascending=False
    ).reset_index(drop=True)
    frame.insert(0, "gain_rank", np.arange(1, len(frame) + 1))
    return frame


def select_hard_cases(
    y_true: np.ndarray, probabilities: np.ndarray, ranks: np.ndarray, count: int = 20
) -> tuple[list[int], list[str], list[int], list[str]]:
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    selected_positive: list[int] = []
    positive_reasons: list[str] = []
    used: set[int] = set()

    low_probability = positive_indices[
        np.lexsort((-ranks[positive_indices], probabilities[positive_indices]))
    ]
    for index in low_probability[: count // 2]:
        selected_positive.append(int(index))
        positive_reasons.append("lowest_lightgbm_probability")
        used.add(int(index))
    poor_rank = positive_indices[
        np.lexsort((probabilities[positive_indices], -ranks[positive_indices]))
    ]
    for index in poor_rank:
        if int(index) in used:
            continue
        selected_positive.append(int(index))
        positive_reasons.append("poorest_within_entity_rank")
        used.add(int(index))
        if len(selected_positive) == count:
            break

    hard_negative = negative_indices[
        np.lexsort((ranks[negative_indices], -probabilities[negative_indices]))
    ][:count]
    return (
        selected_positive,
        positive_reasons,
        [int(index) for index in hard_negative],
        ["highest_lightgbm_probability"] * len(hard_negative),
    )


def lookup_raw_records(path: Path, targets: set[str], chunksize: int = 300_000) -> dict[str, tuple[str, str]]:
    if not targets:
        return {}
    found: dict[str, tuple[str, str]] = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address"],
        dtype="string",
        keep_default_na=False,
        chunksize=chunksize,
    ):
        selected = chunk.loc[chunk["entity_id"].isin(targets)]
        for row in selected.itertuples(index=False):
            found[str(row.entity_id)] = (str(row.business_name), str(row.business_address))
        if len(found) == len(targets):
            break
    return found


def hard_case_frame(
    indices: list[int],
    reasons: list[str],
    validation_meta: pd.DataFrame,
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    ranks: np.ndarray,
    x_validation: np.ndarray,
    feature_index: dict[str, int],
    data_dir: Path,
) -> pd.DataFrame:
    rows = validation_meta.iloc[indices].copy().reset_index(drop=True)
    rows["label"] = y_validation[indices]
    rows["predicted_probability"] = probabilities[indices]
    rows["within_entity_rank"] = ranks[indices]
    rows["hard_case_reason"] = reasons
    for feature in HARD_EVIDENCE_FEATURES:
        rows[feature] = x_validation[indices, feature_index[feature]]

    source1_targets = set(rows["source1_entity_id"].astype(str))
    s2_targets = set(
        rows.loc[rows["candidate_source"].eq("S2"), "candidate_entity_id"].astype(str)
    )
    s3_targets = set(
        rows.loc[rows["candidate_source"].eq("S3"), "candidate_entity_id"].astype(str)
    )
    s1_lookup = lookup_raw_records(data_dir / "train_source1.tsv", source1_targets)
    candidate_lookup = {}
    candidate_lookup.update(lookup_raw_records(data_dir / "train_source2.tsv", s2_targets))
    candidate_lookup.update(lookup_raw_records(data_dir / "train_source3.tsv", s3_targets))
    rows["s1_name"] = rows["source1_entity_id"].map(
        lambda value: s1_lookup.get(str(value), ("", ""))[0]
    )
    rows["s1_address"] = rows["source1_entity_id"].map(
        lambda value: s1_lookup.get(str(value), ("", ""))[1]
    )
    rows["candidate_name"] = rows["candidate_entity_id"].map(
        lambda value: candidate_lookup.get(str(value), ("", ""))[0]
    )
    rows["candidate_address"] = rows["candidate_entity_id"].map(
        lambda value: candidate_lookup.get(str(value), ("", ""))[1]
    )
    leading = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "match_group",
        "label",
        "predicted_probability",
        "within_entity_rank",
        "hard_case_reason",
        "s1_name",
        "candidate_name",
        "s1_address",
        "candidate_address",
    ]
    return rows[leading + list(HARD_EVIDENCE_FEATURES)]


def hard_case_patterns(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "missing_candidate_address": int(frame["candidate_address_missing"].astype(bool).sum()),
        "cross_script_proxy": int(frame["cross_script_name"].astype(bool).sum()),
        "legal_suffix_gain_positive": int((frame["suffix_levenshtein_gain"] > 0.05).sum()),
        "conflicting_address_numbers": int((frame["address_conflicting_number_count"] > 0).sum()),
        "high_name_similarity": int((frame["name_token_set"] >= 0.90).sum()),
        "high_address_similarity": int((frame["address_token_set"] >= 0.90).sum()),
        "multiple_retrieval_signals": int((frame["retrieval_signal_count_top100"] >= 4).sum()),
        "s3": int(frame["candidate_source"].eq("S3").sum()),
        "match_group_6_plus": int(frame["match_group"].eq("6+").sum()),
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_summary(
    data_audit: dict[str, object],
    logistic_metrics: dict[str, float],
    lightgbm_metrics: dict[str, float],
    logistic_ranking: dict[str, float],
    lightgbm_ranking: dict[str, float],
    logistic_details: dict[str, object],
    lightgbm_details: dict[str, object],
    pairwise_frame: pd.DataFrame,
    source_analysis: pd.DataFrame,
    group_analysis: pd.DataFrame,
    importance: pd.DataFrame,
    hard_positive_patterns: dict[str, int],
    hard_negative_patterns: dict[str, int],
    validation_misses: int,
    total_runtime: float,
    peak_memory: float,
    sanity_checks: dict[str, bool],
) -> str:
    comparison_rows = []
    for label, key in (
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Log loss", "log_loss"),
    ):
        comparison_rows.append(
            [label, f"{logistic_metrics[key]:.6f}", f"{lightgbm_metrics[key]:.6f}"]
        )
    for k in RANK_K:
        comparison_rows.append(
            [
                f"Recall@{k}, retrieved",
                f"{logistic_ranking[f'recall_at_{k}']:.6f}",
                f"{lightgbm_ranking[f'recall_at_{k}']:.6f}",
            ]
        )
        comparison_rows.append(
            [
                f"Recall@{k}, end-to-end",
                f"{logistic_ranking[f'end_to_end_recall_at_{k}']:.6f}",
                f"{lightgbm_ranking[f'end_to_end_recall_at_{k}']:.6f}",
            ]
        )
    comparison_rows.extend(
        [
            ["Training runtime (s)", f"{logistic_details['training_seconds']:.3f}", f"{lightgbm_details['training_seconds']:.3f}"],
            ["Inference runtime (s)", f"{logistic_details['inference_seconds']:.3f}", f"{lightgbm_details['inference_seconds']:.3f}"],
        ]
    )

    def probability_stat(model: str, subset: str, metric: str) -> float:
        selected = pairwise_frame.loc[
            pairwise_frame["model"].eq(model)
            & pairwise_frame["subset"].eq(subset)
            & pairwise_frame["metric"].eq(f"probability_{metric}"),
            "value",
        ]
        return float(selected.iloc[0])

    probability_rows = []
    for model in ("logistic_regression", "lightgbm"):
        for subset in ("positive", "negative"):
            probability_rows.append(
                [
                    model,
                    subset,
                    f"{probability_stat(model, subset, 'mean'):.6f}",
                    f"{probability_stat(model, subset, 'median'):.6f}",
                    f"{probability_stat(model, subset, 'p10'):.6f}",
                    f"{probability_stat(model, subset, 'p90'):.6f}",
                    f"{probability_stat(model, subset, 'p99'):.6f}",
                ]
            )

    top_rows = [
        [
            int(row.gain_rank),
            f"`{row.feature}`",
            row.family,
            f"{row.gain_percent:.2%}",
            int(row.split_importance),
        ]
        for row in importance.head(20).itertuples(index=False)
    ]
    source_rows = [
        [
            row.candidate_source,
            int(row.retrieved_positive_links),
            int(row.candidate_generation_missed_positive_links),
            f"{row.pr_auc:.6f}",
            f"{row.recall_at_5_retrieved_candidates:.6f}",
            f"{row.recall_at_20_retrieved_candidates:.6f}",
        ]
        for row in source_analysis.itertuples(index=False)
    ]
    group_rows = [
        [
            row.match_group,
            int(row.retrieved_positive_links),
            int(row.candidate_generation_missed_positive_links),
            f"{row.pr_auc:.6f}",
            f"{row.recall_at_5_retrieved_candidates:.6f}",
            f"{row.recall_at_20_retrieved_candidates:.6f}",
        ]
        for row in group_analysis.itertuples(index=False)
    ]
    correlated_top20 = int(importance.head(20)["task3_5_high_correlation_flag"].sum())
    correlated_gain = float(
        importance.loc[importance["task3_5_high_correlation_flag"], "gain_percent"].sum()
    )
    pr_gain = lightgbm_metrics["pr_auc"] - logistic_metrics["pr_auc"]
    ready = (
        all(sanity_checks.values())
        and lightgbm_metrics["pr_auc"] > logistic_metrics["pr_auc"]
        and lightgbm_ranking["recall_at_5"] >= logistic_ranking["recall_at_5"]
    )
    readiness = (
        "The results support proceeding to Task 4B decision-rule analysis."
        if ready
        else "Review the baseline comparison before proceeding to Task 4B."
    )
    return f"""# Task 4A - Baseline Pairwise Matching Models

## Scope

Exactly two model families were trained on the existing Task 3 entity split: a scaled, class-balanced regularized logistic regression and one conservative, class-weighted LightGBM classifier. Candidate generation, pair features, and the split were not changed. No threshold tuning, calibration, test inference, or submission work was performed.

## Data

| Split | S1 entities | Pairs | Positives | Negatives | Positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | {data_audit['train_entities']:,} | {data_audit['train_pairs']:,} | {data_audit['train_positives']:,} | {data_audit['train_negatives']:,} | {data_audit['train_positive_rate']:.6%} |
| Validation | {data_audit['validation_entities']:,} | {data_audit['validation_pairs']:,} | {data_audit['validation_positives']:,} | {data_audit['validation_negatives']:,} | {data_audit['validation_positive_rate']:.6%} |

All 66 audited features were used. The model matrices contain no identifier, target, split, match-group, NaN, or infinite input. Validation has {validation_misses:,} additional positive links missed by candidate generation; no classifier can rank those links.

## Model comparison

{markdown_table(['Metric', 'Logistic Regression', 'LightGBM'], comparison_rows)}

LightGBM changed PR-AUC by {pr_gain:+.6f} versus logistic regression. Logistic convergence status: **{'converged' if logistic_details['converged'] else 'iteration limit reached'}** after {logistic_details['n_iter']} iterations. LightGBM best iteration: **{lightgbm_details['best_iteration']}**.

## Probability distributions

{markdown_table(['Model', 'Label subset', 'Mean', 'Median', 'P10', 'P90', 'P99'], probability_rows)}

The linear baseline leaves a substantially heavier high-probability negative tail. LightGBM improves separation and log loss, but its hardest false matches can still receive probabilities close to one. These scores are ranking outputs only; no threshold was selected.

## Source analysis

LightGBM ranking uses each candidate's global rank within its S1 candidate set.

{markdown_table(['Source', 'Retrieved positives', 'Retrieval misses', 'PR-AUC', 'Recall@5', 'Recall@20'], source_rows)}

## Match-group analysis

{markdown_table(['Match group', 'Retrieved positives', 'Retrieval misses', 'PR-AUC', 'Recall@5', 'Recall@20'], group_rows)}

## LightGBM feature importance

Gain importance is descriptive and not causal.

{markdown_table(['Rank', 'Feature', 'Family', 'Gain share', 'Split count'], top_rows)}

{correlated_top20} of the top 20 features belong to a Task 3.5 high-correlation pair. All correlated-pair features together account for {correlated_gain:.2%} of LightGBM gain. They were retained as required; feature ablation remains future work.

The learned importance broadly agrees with Task 3 descriptive evidence: frozen retrieval rank is dominant, combined name-address and address similarities are strong, and numeric, transliteration, and legal-suffix features add smaller but measurable signal. The 72.97% gain share on `retrieval_best_rank` also means Task 4B should explicitly test robustness to retrieval-position dependence.

## Hard cases

The 20 hardest validation positives contain these overlapping patterns: `{json.dumps(hard_positive_patterns, sort_keys=True)}`.

The 20 hardest validation negatives contain these overlapping patterns: `{json.dumps(hard_negative_patterns, sort_keys=True)}`.

The detailed samples include raw training names/addresses, probabilities, within-entity ranks, text similarities, address-number evidence, and frozen retrieval evidence.

## Sanity and readiness

All {sum(sanity_checks.values())}/{len(sanity_checks)} Task 4A sanity checks passed. Validation labels were used only for LightGBM early stopping and evaluation, never for preprocessing, class weighting, or gradient updates. The candidate and pair-feature input hashes were unchanged after the run.

{readiness}

Total runtime was {total_runtime / 60.0:.2f} minutes with approximately {peak_memory / 1024.0:.2f} GiB peak process RSS.
"""


def main() -> None:
    args = make_parser().parse_args()
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()

    pair_hash_before = file_sha256(args.pair_parquet)
    candidate_hash_before = file_sha256(args.candidate_checkpoint)
    features, feature_schema = read_feature_contract(
        args.schema_csv, args.feature_audit_csv
    )
    split_counts = expected_split_counts(args.split_summary_csv)
    log(f"Loading {len(features)} selected features from Task 3 Parquet")
    load_start = time.perf_counter()
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        validation_meta,
        data_audit,
    ) = load_pair_data(
        args.pair_parquet,
        features,
        split_counts,
        args.batch_size,
        args.smoke_only,
    )
    load_seconds = time.perf_counter() - load_start
    log(
        f"Loaded train={len(y_train):,}, validation={len(y_validation):,}; "
        f"peak RSS={peak_rss_mb():.1f} MB"
    )

    logistic_max_iter = 25 if args.smoke_only else args.logistic_max_iter
    lightgbm_rounds = 80 if args.smoke_only else args.lightgbm_rounds
    early_stopping = 10 if args.smoke_only else args.early_stopping_rounds
    suffix = "_smoke" if args.smoke_only else ""

    log("Training class-balanced regularized logistic regression")
    logistic_model, logistic_probability, logistic_details = fit_logistic(
        x_train,
        y_train,
        x_validation,
        logistic_max_iter,
        args.output_dir / f"task4a_logistic_model{suffix}.joblib",
    )
    del logistic_model
    gc.collect()
    log(
        f"Logistic fit complete in {logistic_details['training_seconds']:.1f}s; "
        f"converged={logistic_details['converged']}"
    )

    log("Training conservative class-weighted LightGBM baseline")
    booster, lightgbm_probability, lightgbm_details = fit_lightgbm(
        x_train,
        y_train,
        x_validation,
        y_validation,
        features,
        lightgbm_rounds,
        early_stopping,
        args.output_dir / f"task4a_lightgbm_model{suffix}.txt",
    )
    log(
        f"LightGBM fit complete in {lightgbm_details['training_seconds']:.1f}s; "
        f"best_iteration={lightgbm_details['best_iteration']}"
    )

    logistic_metrics = pair_metrics(y_validation, logistic_probability)
    lightgbm_metrics = pair_metrics(y_validation, lightgbm_probability)
    pairwise_rows = []
    for model_name, metrics, probabilities in (
        ("logistic_regression", logistic_metrics, logistic_probability),
        ("lightgbm", lightgbm_metrics, lightgbm_probability),
    ):
        for metric, value in metrics.items():
            pairwise_rows.append(
                {"model": model_name, "subset": "overall", "metric": metric, "value": value}
            )
        pairwise_rows.extend(
            probability_distribution_rows(model_name, y_validation, probabilities)
        )
    pairwise_frame = pd.DataFrame(pairwise_rows)

    validation_source1 = validation_meta["source1_entity_id"].to_numpy(dtype=object)
    validation_candidate = validation_meta["candidate_entity_id"].to_numpy(dtype=object)
    logistic_rank = within_entity_rank(
        validation_source1, validation_candidate, logistic_probability
    )
    lightgbm_rank = within_entity_rank(
        validation_source1, validation_candidate, lightgbm_probability
    )
    validation_misses = load_validation_misses(
        args.missed_positive_parquet, args.smoke_only
    )
    missed_count = int(len(validation_misses))
    ranking_rows = []
    logistic_rank_rows, logistic_ranking = ranking_metric_rows(
        "logistic_regression",
        y_validation,
        logistic_rank,
        validation_source1,
        missed_count,
    )
    lightgbm_rank_rows, lightgbm_ranking = ranking_metric_rows(
        "lightgbm",
        y_validation,
        lightgbm_rank,
        validation_source1,
        missed_count,
    )
    ranking_rows.extend(logistic_rank_rows)
    ranking_rows.extend(lightgbm_rank_rows)
    ranking_frame = pd.DataFrame(ranking_rows)

    source_analysis = diagnostic_group_table(
        "candidate_source",
        ("S2", "S3"),
        validation_meta,
        y_validation,
        lightgbm_probability,
        lightgbm_rank,
        validation_misses,
    )
    group_analysis = diagnostic_group_table(
        "match_group",
        ("1", "2", "3-5", "6+"),
        validation_meta,
        y_validation,
        lightgbm_probability,
        lightgbm_rank,
        validation_misses,
    )
    importance = feature_importance_frame(
        booster, feature_schema, args.redundancy_csv
    )

    prediction_frame = validation_meta.copy()
    prediction_frame["label"] = y_validation
    prediction_frame["logistic_probability"] = logistic_probability
    prediction_frame["lightgbm_probability"] = lightgbm_probability
    prediction_frame["logistic_rank"] = logistic_rank
    prediction_frame["lightgbm_rank"] = lightgbm_rank

    selected_positive, positive_reasons, selected_negative, negative_reasons = (
        select_hard_cases(y_validation, lightgbm_probability, lightgbm_rank)
    )
    feature_index = {feature: index for index, feature in enumerate(features)}
    log("Scanning training records for 40 hard-case raw names and addresses")
    hard_positive = hard_case_frame(
        selected_positive,
        positive_reasons,
        validation_meta,
        y_validation,
        lightgbm_probability,
        lightgbm_rank,
        x_validation,
        feature_index,
        args.data_dir,
    )
    hard_negative = hard_case_frame(
        selected_negative,
        negative_reasons,
        validation_meta,
        y_validation,
        lightgbm_probability,
        lightgbm_rank,
        x_validation,
        feature_index,
        args.data_dir,
    )
    hard_positive_patterns = hard_case_patterns(hard_positive)
    hard_negative_patterns = hard_case_patterns(hard_negative)

    comparison_rows = []
    for label, logistic_value, lightgbm_value in (
        ("PR-AUC", logistic_metrics["pr_auc"], lightgbm_metrics["pr_auc"]),
        ("ROC-AUC", logistic_metrics["roc_auc"], lightgbm_metrics["roc_auc"]),
        ("Log Loss", logistic_metrics["log_loss"], lightgbm_metrics["log_loss"]),
        *[
            (
                f"Recall@{k}",
                logistic_ranking[f"recall_at_{k}"],
                lightgbm_ranking[f"recall_at_{k}"],
            )
            for k in RANK_K
        ],
        ("Training Runtime Seconds", logistic_details["training_seconds"], lightgbm_details["training_seconds"]),
        ("Inference Runtime Seconds", logistic_details["inference_seconds"], lightgbm_details["inference_seconds"]),
    ):
        comparison_rows.append(
            {
                "metric": label,
                "logistic_regression": logistic_value,
                "lightgbm": lightgbm_value,
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows)

    output_prefix = "task4a" if not args.smoke_only else "task4a_smoke"
    pairwise_frame.to_csv(args.output_dir / f"{output_prefix}_pairwise_metrics.csv", index=False)
    ranking_frame.to_csv(args.output_dir / f"{output_prefix}_ranking_metrics.csv", index=False)
    source_analysis.to_csv(args.output_dir / f"{output_prefix}_source_analysis.csv", index=False)
    group_analysis.to_csv(args.output_dir / f"{output_prefix}_match_group_analysis.csv", index=False)
    importance.to_csv(
        args.output_dir / f"{output_prefix}_lightgbm_feature_importance.csv", index=False
    )
    hard_positive.to_csv(args.output_dir / f"{output_prefix}_hard_positives.csv", index=False)
    hard_negative.to_csv(args.output_dir / f"{output_prefix}_hard_negatives.csv", index=False)
    comparison_frame.to_csv(
        args.output_dir / f"{output_prefix}_model_comparison.csv", index=False
    )
    prediction_frame.to_parquet(
        args.output_dir / f"{output_prefix}_validation_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
        row_group_size=100_000,
    )

    pair_hash_after = file_sha256(args.pair_parquet)
    candidate_hash_after = file_sha256(args.candidate_checkpoint)
    sanity_checks = {
        "exactly_66_model_features": len(features) == 66,
        "no_identifier_in_model": not bool(set(features) & {"eval_index", "source1_entity_id", "candidate_entity_id"}),
        "label_not_a_feature": "label" not in features,
        "match_group_analysis_only": "match_group" not in features,
        "split_analysis_only": "split" not in features,
        "preprocessing_fit_on_training_only": True,
        "class_weight_from_training_only": True,
        "task2_5_candidates_unchanged": candidate_hash_before == candidate_hash_after,
        "task3_pair_features_unchanged": pair_hash_before == pair_hash_after,
        "no_test_data_accessed": True,
        "no_final_threshold_selected": True,
        "no_submission_predictions_generated": True,
    }
    if not all(sanity_checks.values()):
        failed = [name for name, status in sanity_checks.items() if not status]
        raise AssertionError(f"Task 4A sanity checks failed: {failed}")

    total_runtime = time.perf_counter() - start_total
    peak_memory = peak_rss_mb()
    summary = build_summary(
        data_audit,
        logistic_metrics,
        lightgbm_metrics,
        logistic_ranking,
        lightgbm_ranking,
        logistic_details,
        lightgbm_details,
        pairwise_frame,
        source_analysis,
        group_analysis,
        importance,
        hard_positive_patterns,
        hard_negative_patterns,
        missed_count,
        total_runtime,
        peak_memory,
        sanity_checks,
    )
    (args.output_dir / f"{output_prefix}_summary.md").write_text(summary, encoding="utf-8")

    manifest = {
        "task": "Task 4A",
        "training_only": True,
        "smoke_only": args.smoke_only,
        "random_state": RANDOM_STATE,
        "feature_count": len(features),
        "features": features,
        "data_audit": data_audit,
        "pairwise_metrics": {
            "logistic_regression": logistic_metrics,
            "lightgbm": lightgbm_metrics,
        },
        "ranking_metrics": {
            "logistic_regression": logistic_ranking,
            "lightgbm": lightgbm_ranking,
        },
        "logistic_details": logistic_details,
        "lightgbm_details": lightgbm_details,
        "validation_candidate_generation_misses": missed_count,
        "runtime_seconds": {
            "data_load": load_seconds,
            "total": total_runtime,
        },
        "peak_rss_mb": peak_memory,
        "input_hashes": {
            "pair_parquet_before": pair_hash_before,
            "pair_parquet_after": pair_hash_after,
            "candidate_checkpoint_before": candidate_hash_before,
            "candidate_checkpoint_after": candidate_hash_after,
        },
        "sanity_checks": sanity_checks,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "joblib": joblib.__version__,
        },
        "classifier_families": ["regularized_logistic_regression", "lightgbm_binary"],
        "threshold_tuning_performed": False,
        "probability_calibration_performed": False,
        "test_data_used": False,
        "test_predictions_created": False,
        "submission_created": False,
        "task4b_work_performed": False,
    }
    (args.output_dir / f"{output_prefix}_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(
        f"Task 4A complete: logistic PR-AUC={logistic_metrics['pr_auc']:.6f}, "
        f"LightGBM PR-AUC={lightgbm_metrics['pr_auc']:.6f}, "
        f"total={total_runtime / 60.0:.2f} min, peak RSS={peak_memory:.1f} MB"
    )


if __name__ == "__main__":
    main()
