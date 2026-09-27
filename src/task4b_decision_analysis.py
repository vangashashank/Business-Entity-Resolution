#!/usr/bin/env python3
"""Task 4B: LightGBM rank ablations, error analysis, and decision rules."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for vendor_dir in (PROJECT_ROOT / ".task4_vendor", PROJECT_ROOT / ".task3_vendor"):
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

import task4a_baseline_models as t4a


RANDOM_STATE = 42
DIRECT_POSITION_FEATURES = (
    "baseline_name_rank",
    "baseline_name_rank_missing",
    "baseline_address_rank",
    "baseline_address_rank_missing",
    "transliterated_name_rank",
    "transliterated_name_rank_missing",
    "number_address_rank",
    "number_address_rank_missing",
    "suffix_name_rank",
    "suffix_name_rank_missing",
    "retrieval_best_rank",
    "retrieval_mean_rank_present",
    "retrieval_rrf_score_top100",
    "frozen_candidate_position",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task3-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs",
    )
    parser.add_argument(
        "--task4-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task4_outputs",
    )
    parser.add_argument(
        "--task2-5-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task2_5_outputs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource" / "dataset" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task4_outputs" / "task4b",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "pairs": args.task3_dir / "task3_pair_features.parquet",
        "schema": args.task3_dir / "task3_feature_schema.csv",
        "feature_audit": args.task3_dir / "task3_to_task4_feature_audit.csv",
        "split_summary": args.task3_dir / "task3_split_summary.csv",
        "entity_split": args.task3_dir / "task3_entity_split.csv",
        "missed": args.task3_dir / "task3_missed_positive_features.parquet",
        "task4a_predictions": args.task4_dir / "task4a_validation_predictions.parquet",
        "task4a_model": args.task4_dir / "task4a_lightgbm_model.txt",
        "task4a_manifest": args.task4_dir / "task4a_run_manifest.json",
        "candidate_checkpoint": args.task2_5_dir / "task2_5_retrieval_top250.npz",
        "s1": args.data_dir / "train_source1.tsv",
        "s2": args.data_dir / "train_source2.tsv",
        "s3": args.data_dir / "train_source3.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 4B inputs: {missing}")
    for path in paths.values():
        if "test" in path.name.casefold():
            raise ValueError(f"Task 4B cannot access test data: {path}")
    return paths


def hash_inputs(paths: dict[str, Path]) -> dict[str, str]:
    important = (
        "pairs",
        "schema",
        "entity_split",
        "task4a_predictions",
        "task4a_model",
        "task4a_manifest",
        "candidate_checkpoint",
    )
    return {name: t4a.file_sha256(paths[name]) for name in important}


def read_validation_misses(path: Path) -> pd.DataFrame:
    requested = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "match_group",
        "split",
        "label",
        *t4a.HARD_EVIDENCE_FEATURES,
    ]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [column for column in requested if column in available]
    frame = pq.read_table(path, columns=columns).to_pandas()
    frame = frame.loc[frame["split"].eq("validation")].copy()
    if len(frame) != 239 or not frame["label"].eq(1).all():
        raise AssertionError(
            f"Expected 239 validation candidate-generation misses, found {len(frame)}"
        )
    return frame


def validate_prediction_alignment(
    predictions: pd.DataFrame,
    validation_meta: pd.DataFrame,
    y_validation: np.ndarray,
) -> None:
    if len(predictions) != len(validation_meta):
        raise AssertionError("Task 4A predictions and Task 3 validation rows differ")
    for column in ("source1_entity_id", "candidate_entity_id", "candidate_source", "match_group"):
        left = predictions[column].astype(str).to_numpy()
        right = validation_meta[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise AssertionError(f"Task 4A prediction alignment failed for {column}")
    if not np.array_equal(predictions["label"].to_numpy(dtype=np.uint8), y_validation):
        raise AssertionError("Task 4A labels do not align with Task 3 validation rows")


def entity_context(
    validation_meta: pd.DataFrame,
    probabilities: np.ndarray,
    misses: pd.DataFrame,
) -> dict[str, object]:
    entity_ids, entity_codes = np.unique(
        validation_meta["source1_entity_id"].astype(str).to_numpy(),
        return_inverse=True,
    )
    entity_count = len(entity_ids)
    best = np.full(entity_count, -np.inf, dtype=np.float32)
    np.maximum.at(best, entity_codes, probabilities)
    missed_counts = (
        misses.groupby("source1_entity_id").size().reindex(entity_ids, fill_value=0).to_numpy(dtype=np.int32)
    )
    return {
        "entity_ids": entity_ids,
        "entity_codes": entity_codes.astype(np.int32),
        "entity_count": entity_count,
        "best_scores": best,
        "best_score_per_row": best[entity_codes],
        "missed_counts": missed_counts,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_rule(
    predicted: np.ndarray,
    labels: np.ndarray,
    entity_codes: np.ndarray,
    entity_count: int,
    missed_counts: np.ndarray,
) -> dict[str, object]:
    predicted = predicted.astype(bool, copy=False)
    positive = labels == 1
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn_retrieved = int((~predicted & positive).sum())
    missed = int(missed_counts.sum())
    precision, recall_retrieved, f1_retrieved = precision_recall_f1(
        tp, fp, fn_retrieved
    )
    _, recall_end_to_end, f1_end_to_end = precision_recall_f1(
        tp, fp, fn_retrieved + missed
    )

    predicted_count = np.bincount(
        entity_codes, weights=predicted.astype(np.int16), minlength=entity_count
    ).astype(np.int32)
    true_count = np.bincount(
        entity_codes, weights=positive.astype(np.int16), minlength=entity_count
    ).astype(np.int32)
    tp_count = np.bincount(
        entity_codes, weights=(predicted & positive).astype(np.int16), minlength=entity_count
    ).astype(np.int32)
    exact_retrieved = (predicted_count == true_count) & (tp_count == true_count)
    exact_end_to_end = exact_retrieved & (missed_counts == 0)
    return {
        "true_positive_links": tp,
        "false_positive_links": fp,
        "false_negative_retrieved_links": fn_retrieved,
        "candidate_generation_missed_links": missed,
        "false_negative_end_to_end_links": fn_retrieved + missed,
        "predicted_links": int(predicted.sum()),
        "precision": precision,
        "recall_retrieved": recall_retrieved,
        "f1_retrieved": f1_retrieved,
        "recall_end_to_end": recall_end_to_end,
        "f1_end_to_end": f1_end_to_end,
        "average_predicted_matches_per_entity": float(predicted_count.mean()),
        "entities_with_zero_predictions": int((predicted_count == 0).sum()),
        "complete_entity_matches_retrieved": int(exact_retrieved.sum()),
        "complete_entity_match_rate_retrieved": float(exact_retrieved.mean()),
        "complete_entity_matches_end_to_end": int(exact_end_to_end.sum()),
        "complete_entity_match_rate_end_to_end": float(exact_end_to_end.mean()),
    }


def rule_candidates(
    probabilities: np.ndarray,
    ranks: np.ndarray,
    best_score_per_row: np.ndarray,
) -> Iterable[tuple[str, str, dict[str, object], np.ndarray, int]]:
    absolute_thresholds = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
        0.995,
    )
    for threshold in absolute_thresholds:
        yield (
            f"global_p_ge_{threshold:g}",
            "global_probability_threshold",
            {"probability_threshold": threshold},
            probabilities >= threshold,
            1,
        )

    for top_k in (5, 10, 20):
        for threshold in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95):
            yield (
                f"top_{top_k}_and_p_ge_{threshold:g}",
                "top_k_and_probability_threshold",
                {"top_k": top_k, "probability_threshold": threshold},
                (ranks <= top_k) & (probabilities >= threshold),
                2,
            )

    safe_best = np.maximum(best_score_per_row, np.float32(1e-12))
    score_ratio = probabilities / safe_best
    for ratio in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90):
        yield (
            f"relative_ratio_ge_{ratio:g}",
            "relative_to_best_ratio",
            {"minimum_best_score_ratio": ratio},
            score_ratio >= ratio,
            2,
        )
    score_gap = best_score_per_row - probabilities
    for gap in (0.01, 0.05, 0.10, 0.20):
        yield (
            f"relative_gap_le_{gap:g}",
            "relative_to_best_gap",
            {"maximum_best_score_gap": gap},
            score_gap <= gap,
            2,
        )

    for threshold in (0.10, 0.30, 0.50, 0.70, 0.90):
        for ratio in (0.05, 0.10, 0.20, 0.50):
            yield (
                f"hybrid_p_ge_{threshold:g}_ratio_ge_{ratio:g}",
                "absolute_and_relative_hybrid",
                {
                    "probability_threshold": threshold,
                    "minimum_best_score_ratio": ratio,
                },
                (probabilities >= threshold) & (score_ratio >= ratio),
                3,
            )


def decision_rule_search(
    probabilities: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    context: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, dict[str, np.ndarray]]:
    rows = []
    masks: dict[str, np.ndarray] = {}
    for rule_name, family, params, predicted, complexity in rule_candidates(
        probabilities,
        ranks,
        np.asarray(context["best_score_per_row"]),
    ):
        metrics = evaluate_rule(
            predicted,
            labels,
            np.asarray(context["entity_codes"]),
            int(context["entity_count"]),
            np.asarray(context["missed_counts"]),
        )
        rows.append(
            {
                "rule_name": rule_name,
                "rule_family": family,
                "parameters": json.dumps(params, sort_keys=True),
                "complexity_order": complexity,
                **metrics,
            }
        )
        masks[rule_name] = predicted
    frame = pd.DataFrame(rows)
    best_complete = float(frame["complete_entity_match_rate_end_to_end"].max())
    shortlist = frame.loc[
        frame["complete_entity_match_rate_end_to_end"] >= best_complete - 0.01
    ].copy()
    shortlist = shortlist.sort_values(
        [
            "f1_end_to_end",
            "complete_entity_match_rate_end_to_end",
            "false_positive_links",
            "complexity_order",
            "rule_name",
        ],
        ascending=[False, False, True, True, True],
    )
    selected_row = shortlist.iloc[0].to_dict()
    selected_name = str(selected_row["rule_name"])
    frame["selected"] = frame["rule_name"].eq(selected_name)
    selected_row["selection_method"] = (
        "Among rules within 1.0 percentage point of the best end-to-end complete "
        "entity rate, maximize end-to-end link F1, then prefer fewer false "
        "positives and lower rule complexity. This avoids forcing low-score top "
        "candidates for a marginal complete-set gain."
    )
    return frame, selected_row, masks[selected_name], masks


def subgroup_policy_metrics(
    group_column: str,
    values: Iterable[str],
    validation_meta: pd.DataFrame,
    labels: np.ndarray,
    predicted: np.ndarray,
    misses: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for value in values:
        row_mask = validation_meta[group_column].astype(str).eq(str(value)).to_numpy()
        subset_meta = validation_meta.loc[row_mask]
        subset_labels = labels[row_mask]
        subset_predicted = predicted[row_mask]
        entity_ids, entity_codes = np.unique(
            subset_meta["source1_entity_id"].astype(str).to_numpy(), return_inverse=True
        )
        missed_subset = misses.loc[misses[group_column].astype(str).eq(str(value))]
        missed_counts = (
            missed_subset.groupby("source1_entity_id")
            .size()
            .reindex(entity_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        metrics = evaluate_rule(
            subset_predicted,
            subset_labels,
            entity_codes.astype(np.int32),
            len(entity_ids),
            missed_counts,
        )
        rows.append({group_column: value, "entities": len(entity_ids), **metrics})
    return pd.DataFrame(rows)


def train_fixed_ablation(
    variant_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    feature_names: list[str],
    params: dict[str, object],
    iterations: int,
    output_path: Path,
) -> tuple[np.ndarray, float, float]:
    train_start = time.perf_counter()
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=feature_names,
        free_raw_data=True,
    )
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=iterations,
        callbacks=[lgb.log_evaluation(period=250)],
    )
    training_seconds = time.perf_counter() - train_start
    inference_start = time.perf_counter()
    probabilities = booster.predict(x_validation, num_iteration=iterations).astype(np.float32)
    inference_seconds = time.perf_counter() - inference_start
    if not np.isfinite(probabilities).all():
        raise AssertionError(f"Non-finite predictions for ablation {variant_name}")
    booster.save_model(str(output_path), num_iteration=iterations)
    return probabilities, training_seconds, inference_seconds


def ablation_rows(
    variant: str,
    feature_count: int,
    removed: list[str],
    iterations: int,
    training_seconds: float,
    inference_seconds: float,
    probabilities: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    validation_meta: pd.DataFrame,
    misses: pd.DataFrame,
) -> list[dict[str, object]]:
    contexts: list[tuple[str, str, np.ndarray, int]] = [
        ("overall", "all", np.ones(len(labels), dtype=bool), len(misses))
    ]
    for source in ("S2", "S3"):
        mask = validation_meta["candidate_source"].eq(source).to_numpy()
        miss_count = int(misses["candidate_source"].eq(source).sum())
        contexts.append(("candidate_source", source, mask, miss_count))
    for group in ("1", "2", "3-5", "6+"):
        mask = validation_meta["match_group"].astype(str).eq(group).to_numpy()
        miss_count = int(misses["match_group"].astype(str).eq(group).sum())
        contexts.append(("match_group", group, mask, miss_count))

    rows = []
    for scope, group, mask, miss_count in contexts:
        subset_labels = labels[mask]
        subset_scores = probabilities[mask]
        subset_ranks = ranks[mask]
        positive = subset_labels == 1
        retrieved_positives = int(positive.sum())
        metrics = t4a.pair_metrics(subset_labels, subset_scores)
        row: dict[str, object] = {
            "variant": variant,
            "scope": scope,
            "group": group,
            "feature_count": feature_count,
            "removed_feature_count": len(removed),
            "removed_features": ";".join(removed),
            "fixed_iterations": iterations,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "pair_rows": int(mask.sum()),
            "retrieved_positive_links": retrieved_positives,
            "candidate_generation_missed_positive_links": miss_count,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "log_loss": metrics["log_loss"],
            "mean_positive_rank": float(subset_ranks[positive].mean()),
            "median_positive_rank": float(np.median(subset_ranks[positive])),
        }
        for k in t4a.RANK_K:
            hits = int((subset_ranks[positive] <= k).sum())
            row[f"recall_at_{k}_retrieved_candidates"] = hits / retrieved_positives
            row[f"recall_at_{k}_end_to_end"] = hits / (retrieved_positives + miss_count)
        rows.append(row)
    return rows


def score_reliability(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float]]:
    boundaries = np.asarray(
        [
            0.0,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            0.05,
            0.10,
            0.20,
            0.50,
            0.80,
            0.90,
            0.95,
            0.99,
            0.999,
            1.0000001,
        ],
        dtype=np.float64,
    )
    bin_index = np.digitize(probabilities, boundaries[1:-1], right=False)
    rows = []
    ece = 0.0
    for index in range(len(boundaries) - 1):
        mask = bin_index == index
        count = int(mask.sum())
        if not count:
            continue
        mean_score = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        gap = abs(mean_score - observed)
        ece += gap * count / len(labels)
        rows.append(
            {
                "lower_bound": boundaries[index],
                "upper_bound": boundaries[index + 1],
                "pairs": count,
                "positive_links": int(labels[mask].sum()),
                "mean_predicted_probability": mean_score,
                "observed_positive_rate": observed,
                "absolute_calibration_gap": gap,
            }
        )
    diagnostics = {
        "expected_calibration_error": float(ece),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
    }
    return pd.DataFrame(rows), diagnostics


def error_aggregate_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    ranks: np.ndarray,
    selected: np.ndarray,
    validation_meta: pd.DataFrame,
    x_validation: np.ndarray,
    feature_index: dict[str, int],
    misses: pd.DataFrame,
) -> list[dict[str, object]]:
    positive = labels == 1
    negative = ~positive
    rows: list[dict[str, object]] = []

    def add(error_type: str, count: int, denominator: int, note: str) -> None:
        rows.append(
            {
                "analysis_scope": "overall",
                "error_type": error_type,
                "segment": "all",
                "segment_value": "all",
                "count": count,
                "denominator": denominator,
                "rate": count / denominator if denominator else 0.0,
                "note": note,
            }
        )

    add(
        "candidate_generation_miss",
        len(misses),
        int(positive.sum()) + len(misses),
        "Unreachable by the classifier; excluded from ranking-failure counts.",
    )
    for k in (5, 10, 20):
        add(
            f"retrieved_positive_rank_gt_{k}",
            int((positive & (ranks > k)).sum()),
            int(positive.sum()),
            "Retrieved positive ranked below the stated within-entity cutoff.",
        )
    for threshold in (0.10, 0.50):
        add(
            f"retrieved_positive_probability_lt_{threshold:g}",
            int((positive & (probabilities < threshold)).sum()),
            int(positive.sum()),
            "Low-scoring retrieved positive.",
        )
    for threshold in (0.50, 0.90, 0.99):
        add(
            f"negative_probability_ge_{threshold:g}",
            int((negative & (probabilities >= threshold)).sum()),
            int(negative.sum()),
            "High-scoring retrieved negative.",
        )
    add(
        "selected_policy_false_positive",
        int((selected & negative).sum()),
        int(negative.sum()),
        "Error introduced by the selected validation decision rule.",
    )
    add(
        "selected_policy_false_negative_retrieved",
        int((~selected & positive).sum()),
        int(positive.sum()),
        "Retrieved positive rejected by the selected validation decision rule.",
    )

    feature = lambda name: x_validation[:, feature_index[name]]
    positive_segments = {
        "cross_script_proxy": feature("cross_script_name") > 0.5,
        "missing_candidate_address": feature("candidate_address_missing") > 0.5,
        "conflicting_address_numbers": feature("address_conflicting_number_count") > 0,
        "transliteration_dependent": feature("transliteration_levenshtein_gain") > 0.10,
        "legal_suffix_dependent": feature("suffix_levenshtein_gain") > 0.10,
        "weak_retrieval_consensus": feature("retrieval_signal_count_top100") <= 1,
    }
    negative_segments = {
        "near_identical_name_and_address": (
            (feature("name_token_set") >= 0.90)
            & (feature("address_token_set") >= 0.90)
        ),
        "conflicting_address_numbers": feature("address_conflicting_number_count") > 0,
        "multiple_retrieval_signals": feature("retrieval_signal_count_top100") >= 4,
    }
    for segment_name, segment_mask in positive_segments.items():
        population = positive & segment_mask
        for error_type, error_mask in (
            ("positive_rank_gt_10", ranks > 10),
            ("selected_policy_false_negative", ~selected),
        ):
            count = int((population & error_mask).sum())
            denominator = int(population.sum())
            rows.append(
                {
                    "analysis_scope": "positive_feature_segment",
                    "error_type": error_type,
                    "segment": "feature_pattern",
                    "segment_value": segment_name,
                    "count": count,
                    "denominator": denominator,
                    "rate": count / denominator if denominator else 0.0,
                    "note": "Candidate-generation misses excluded.",
                }
            )
    for segment_name, segment_mask in negative_segments.items():
        population = negative & segment_mask
        count = int((population & selected).sum())
        denominator = int(population.sum())
        rows.append(
            {
                "analysis_scope": "negative_feature_segment",
                "error_type": "selected_policy_false_positive",
                "segment": "feature_pattern",
                "segment_value": segment_name,
                "count": count,
                "denominator": denominator,
                "rate": count / denominator if denominator else 0.0,
                "note": "High-risk retrieved-negative segment.",
            }
        )

    for group_column, values in (
        ("candidate_source", ("S2", "S3")),
        ("match_group", ("1", "2", "3-5", "6+")),
    ):
        for value in values:
            group_mask = validation_meta[group_column].astype(str).eq(value).to_numpy()
            for error_type, error_mask, denominator_mask in (
                ("selected_policy_false_positive", selected & negative, negative),
                ("selected_policy_false_negative_retrieved", ~selected & positive, positive),
                ("retrieved_positive_rank_gt_10", positive & (ranks > 10), positive),
            ):
                count = int((group_mask & error_mask).sum())
                denominator = int((group_mask & denominator_mask).sum())
                rows.append(
                    {
                        "analysis_scope": group_column,
                        "error_type": error_type,
                        "segment": group_column,
                        "segment_value": value,
                        "count": count,
                        "denominator": denominator,
                        "rate": count / denominator if denominator else 0.0,
                        "note": "Retrieved candidates only.",
                    }
                )
    return rows


def retrieved_error_case_sample(
    labels: np.ndarray,
    probabilities: np.ndarray,
    ranks: np.ndarray,
    selected: np.ndarray,
    validation_meta: pd.DataFrame,
    x_validation: np.ndarray,
    feature_index: dict[str, int],
    data_dir: Path,
) -> pd.DataFrame:
    positive = labels == 1
    negative = ~positive
    reasons: dict[int, set[str]] = defaultdict(set)

    def add(indices: np.ndarray, reason: str, limit: int | None = 20) -> None:
        chosen = indices if limit is None else indices[:limit]
        for index in chosen:
            reasons[int(index)].add(reason)

    positive_indices = np.flatnonzero(positive)
    negative_indices = np.flatnonzero(negative)
    add(
        positive_indices[np.argsort(probabilities[positive_indices])],
        "lowest_scoring_positive",
    )
    add(
        positive_indices[np.lexsort((probabilities[positive_indices], -ranks[positive_indices]))],
        "poor_positive_rank",
    )
    add(
        positive_indices[ranks[positive_indices] > 20],
        "positive_rank_outside_20",
        None,
    )
    add(
        negative_indices[np.argsort(-probabilities[negative_indices])],
        "highest_scoring_negative",
    )
    false_positive = np.flatnonzero(selected & negative)
    add(false_positive[np.argsort(-probabilities[false_positive])], "selected_policy_false_positive")
    false_negative = np.flatnonzero(~selected & positive)
    add(false_negative[np.argsort(probabilities[false_negative])], "selected_policy_false_negative")

    indices = sorted(reasons)
    reason_text = [";".join(sorted(reasons[index])) for index in indices]
    return t4a.hard_case_frame(
        indices,
        reason_text,
        validation_meta,
        labels,
        probabilities,
        ranks,
        x_validation,
        feature_index,
        data_dir,
    )


def missed_error_case_sample(misses: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    sample = (
        misses.groupby("match_group", group_keys=False)
        .head(5)
        .head(20)
        .copy()
        .reset_index(drop=True)
    )
    source1_targets = set(sample["source1_entity_id"].astype(str))
    s2_targets = set(sample.loc[sample["candidate_source"].eq("S2"), "candidate_entity_id"].astype(str))
    s3_targets = set(sample.loc[sample["candidate_source"].eq("S3"), "candidate_entity_id"].astype(str))
    s1_lookup = t4a.lookup_raw_records(data_dir / "train_source1.tsv", source1_targets)
    candidate_lookup = {}
    candidate_lookup.update(t4a.lookup_raw_records(data_dir / "train_source2.tsv", s2_targets))
    candidate_lookup.update(t4a.lookup_raw_records(data_dir / "train_source3.tsv", s3_targets))
    sample["predicted_probability"] = np.nan
    sample["within_entity_rank"] = np.nan
    sample["hard_case_reason"] = "candidate_generation_miss_unreachable"
    sample["s1_name"] = sample["source1_entity_id"].map(lambda value: s1_lookup.get(str(value), ("", ""))[0])
    sample["s1_address"] = sample["source1_entity_id"].map(lambda value: s1_lookup.get(str(value), ("", ""))[1])
    sample["candidate_name"] = sample["candidate_entity_id"].map(lambda value: candidate_lookup.get(str(value), ("", ""))[0])
    sample["candidate_address"] = sample["candidate_entity_id"].map(lambda value: candidate_lookup.get(str(value), ("", ""))[1])
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
    for feature in t4a.HARD_EVIDENCE_FEATURES:
        if feature not in sample:
            sample[feature] = np.nan
    return sample[leading + list(t4a.HARD_EVIDENCE_FEATURES)]


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def build_summary(
    ablation: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_policy: dict[str, object],
    source_analysis: pd.DataFrame,
    group_analysis: pd.DataFrame,
    error_analysis: pd.DataFrame,
    calibration: dict[str, float],
    integrity: dict[str, bool],
    runtime_seconds: float,
    peak_rss_mb: float,
) -> str:
    overall = ablation.loc[ablation["scope"].eq("overall")]
    ablation_rows_md = [
        [
            row.variant,
            int(row.feature_count),
            f"{row.pr_auc:.6f}",
            f"{row.log_loss:.6f}",
            f"{row.recall_at_5_retrieved_candidates:.4%}",
            f"{row.recall_at_20_end_to_end:.4%}",
        ]
        for row in overall.itertuples(index=False)
    ]
    selected = comparison.loc[comparison["selected"]].iloc[0]
    source_selected = source_analysis.loc[source_analysis["policy_role"].eq("selected")]
    group_selected = group_analysis.loc[group_analysis["policy_role"].eq("selected")]
    source_rows = [
        [
            row.candidate_source,
            f"{row.precision:.4%}",
            f"{row.recall_retrieved:.4%}",
            f"{row.recall_end_to_end:.4%}",
            f"{row.f1_end_to_end:.4%}",
        ]
        for row in source_selected.itertuples(index=False)
    ]
    group_rows = [
        [
            row.match_group,
            f"{row.precision:.4%}",
            f"{row.recall_retrieved:.4%}",
            f"{row.recall_end_to_end:.4%}",
            f"{row.complete_entity_match_rate_end_to_end:.4%}",
        ]
        for row in group_selected.itertuples(index=False)
    ]
    key_errors = error_analysis.loc[
        error_analysis["analysis_scope"].eq("overall")
    ][["error_type", "count", "denominator", "rate"]]
    error_rows = [
        [row.error_type, int(row.count), int(row.denominator), f"{row.rate:.4%}"]
        for row in key_errors.itertuples(index=False)
    ]
    ablation_drop = float(
        overall.loc[overall["variant"].eq("full_66_existing"), "pr_auc"].iloc[0]
        - overall.loc[overall["variant"].eq("remove_all_retrieval_evidence"), "pr_auc"].iloc[0]
    )
    complete_leader = comparison.sort_values(
        ["complete_entity_match_rate_end_to_end", "f1_end_to_end"],
        ascending=[False, False],
    ).iloc[0]
    tradeoff_text = (
        f"The maximum-complete comparator `{complete_leader.rule_name}` reaches "
        f"{complete_leader.complete_entity_match_rate_end_to_end:.4%} complete "
        f"recovery versus {selected.complete_entity_match_rate_end_to_end:.4%} for "
        f"the selected rule, but adds "
        f"{int(complete_leader.false_positive_links - selected.false_positive_links):,} "
        f"false positives and lowers end-to-end F1 from {selected.f1_end_to_end:.4%} "
        f"to {complete_leader.f1_end_to_end:.4%}. The source/group CSVs retain both "
        f"policies for direct inspection."
        if complete_leader.rule_name != selected.rule_name
        else "The selected rule is also the maximum-complete rule in the evaluated grid."
    )
    ready = all(integrity.values())
    readiness = (
        "Task 4 can be frozen on validation and the project is ready to plan full-scale inference."
        if ready
        else "Task 4 is not ready to freeze because one or more integrity checks failed."
    )
    return f"""# Task 4B - Rank Robustness and Entity-Level Decision Policy

## Scope

Task 4B reused the frozen Task 4A LightGBM validation predictions, trained three controlled LightGBM ablations at the same fixed 1,489 iterations, inspected ranking/classification errors, and selected one validation-only entity decision policy. No candidate, pair feature, entity split, Task 4A artifact, test record, or submission was changed or generated.

## Retrieval-rank dependence

{markdown_table(['Variant', 'Features', 'PR-AUC', 'Log loss', 'Retrieved Recall@5', 'End-to-end Recall@20'], ablation_rows_md)}

Removing all retrieval evidence changes PR-AUC by {-ablation_drop:+.6f} relative to the full model. The detailed artifact includes ROC-AUC, all Recall@K values, and S2/S3 and match-group breakdowns for every variant. This quantifies rank dependence directly rather than inferring it from gain importance.

## Selected decision policy

- Rule: **{selected.rule_name}**
- Family: `{selected.rule_family}`
- Parameters: `{selected.parameters}`
- Predicted links: **{int(selected.predicted_links):,}**
- Average predicted matches per S1: **{selected.average_predicted_matches_per_entity:.4f}**
- Entities with zero predictions: **{int(selected.entities_with_zero_predictions):,}**
- Precision: **{selected.precision:.4%}**
- Retrieved-only recall / F1: **{selected.recall_retrieved:.4%} / {selected.f1_retrieved:.4%}**
- End-to-end recall / F1: **{selected.recall_end_to_end:.4%} / {selected.f1_end_to_end:.4%}**
- Retrieved-only complete entity rate: **{selected.complete_entity_match_rate_retrieved:.4%}**
- End-to-end complete entity rate: **{selected.complete_entity_match_rate_end_to_end:.4%}**

There are **{int(selected.candidate_generation_missed_links):,}** unreachable validation positives. They reduce end-to-end recall and complete-set recovery but are not classifier failures.

Selection method: {selected_policy['selection_method']}

{tradeoff_text}

## Source behavior

{markdown_table(['Source', 'Precision', 'Retrieved recall', 'End-to-end recall', 'End-to-end F1'], source_rows)}

## Match-group behavior

{markdown_table(['Match group', 'Precision', 'Retrieved recall', 'End-to-end recall', 'End-to-end complete rate'], group_rows)}

## Dominant errors

{markdown_table(['Error type', 'Count', 'Denominator', 'Rate'], error_rows)}

The detailed error artifacts separate candidate-generation misses, poorly ranked retrieved positives, high-scoring negatives, feature-pattern risks, and errors introduced by the selected policy.

## Score reliability

LightGBM expected calibration error is **{calibration['expected_calibration_error']:.6f}** and Brier score is **{calibration['brier_score']:.6f}** on validation. No calibration model was added. The selected policy treats scores as validation-ranked evidence rather than claiming literal probability calibration.

## Integrity and readiness

All **{sum(integrity.values())}/{len(integrity)}** integrity checks passed. The selected decision rule uses only the LightGBM pair score. It does not use match-count group, labels, identifiers as predictors, or test information.

{readiness}

Runtime was **{runtime_seconds / 60.0:.2f} minutes** with approximately **{peak_rss_mb / 1024.0:.2f} GiB** peak process RSS.
"""


def main() -> None:
    args = make_parser().parse_args()
    paths = validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()
    hashes_before = hash_inputs(paths)

    features, feature_schema = t4a.read_feature_contract(
        paths["schema"], paths["feature_audit"]
    )
    split_counts = t4a.expected_split_counts(paths["split_summary"])
    log("Loading the frozen 66-feature train/validation matrices")
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        validation_meta,
        data_audit,
    ) = t4a.load_pair_data(
        paths["pairs"],
        features,
        split_counts,
        args.batch_size,
        args.smoke_only,
    )
    task4a_predictions = pd.read_parquet(paths["task4a_predictions"])
    if args.smoke_only:
        task4a_predictions = task4a_predictions.iloc[: len(y_validation)].copy()
    validate_prediction_alignment(task4a_predictions, validation_meta, y_validation)
    full_probabilities = task4a_predictions["lightgbm_probability"].to_numpy(dtype=np.float32)
    full_ranks = task4a_predictions["lightgbm_rank"].to_numpy(dtype=np.uint16)
    misses = read_validation_misses(paths["missed"])
    if args.smoke_only:
        misses = misses.iloc[0:0].copy()
    context = entity_context(validation_meta, full_probabilities, misses)

    log("Evaluating interpretable validation-only decision rules")
    comparison, selected_policy, selected_mask, rule_masks = decision_rule_search(
        full_probabilities,
        full_ranks,
        y_validation,
        context,
    )
    complete_leader_name = str(
        comparison.sort_values(
            ["complete_entity_match_rate_end_to_end", "f1_end_to_end"],
            ascending=[False, False],
        ).iloc[0]["rule_name"]
    )
    diagnostic_policies = [("selected", str(selected_policy["rule_name"]))]
    if complete_leader_name != selected_policy["rule_name"]:
        diagnostic_policies.append(("max_complete_comparator", complete_leader_name))
    source_frames = []
    group_frames = []
    for policy_role, rule_name in diagnostic_policies:
        source_frame = subgroup_policy_metrics(
            "candidate_source",
            ("S2", "S3"),
            validation_meta,
            y_validation,
            rule_masks[rule_name],
            misses,
        )
        source_frame.insert(0, "rule_name", rule_name)
        source_frame.insert(0, "policy_role", policy_role)
        source_frames.append(source_frame)
        group_frame = subgroup_policy_metrics(
            "match_group",
            ("1", "2", "3-5", "6+"),
            validation_meta,
            y_validation,
            rule_masks[rule_name],
            misses,
        )
        group_frame.insert(0, "rule_name", rule_name)
        group_frame.insert(0, "policy_role", policy_role)
        group_frames.append(group_frame)
    source_analysis = pd.concat(source_frames, ignore_index=True)
    group_analysis = pd.concat(group_frames, ignore_index=True)
    reliability, calibration = score_reliability(full_probabilities, y_validation)

    task4a_manifest = json.loads(paths["task4a_manifest"].read_text())
    baseline_iterations = int(task4a_manifest["lightgbm_details"]["best_iteration"])
    if args.smoke_only:
        baseline_iterations = 50
    params = dict(task4a_manifest["lightgbm_details"]["configuration"])
    params["metric"] = "None"

    retrieval_features = feature_schema.loc[
        feature_schema["family"].eq("retrieval_evidence"), "column"
    ].astype(str).tolist()
    variants = [
        ("remove_retrieval_best_rank", ["retrieval_best_rank"]),
        ("remove_direct_position_features", list(DIRECT_POSITION_FEATURES)),
        ("remove_all_retrieval_evidence", retrieval_features),
    ]
    ablation_rows_all = ablation_rows(
        "full_66_existing",
        len(features),
        [],
        int(task4a_manifest["lightgbm_details"]["best_iteration"]),
        0.0,
        0.0,
        full_probabilities,
        y_validation,
        full_ranks,
        validation_meta,
        misses,
    )

    for variant_name, removed in variants:
        missing_removed = sorted(set(removed) - set(features))
        if missing_removed:
            raise AssertionError(f"Ablation references absent features: {missing_removed}")
        retained = [feature for feature in features if feature not in set(removed)]
        indices = np.asarray([features.index(feature) for feature in retained], dtype=np.int32)
        log(
            f"Training {variant_name}: {len(retained)} features, "
            f"fixed {baseline_iterations} iterations"
        )
        variant_x_train = np.ascontiguousarray(x_train[:, indices], dtype=np.float32)
        variant_x_validation = np.ascontiguousarray(
            x_validation[:, indices], dtype=np.float32
        )
        probabilities, training_seconds, inference_seconds = train_fixed_ablation(
            variant_name,
            variant_x_train,
            y_train,
            variant_x_validation,
            retained,
            params,
            baseline_iterations,
            args.output_dir / f"task4b_{variant_name}_model.txt",
        )
        ranks = t4a.within_entity_rank(
            validation_meta["source1_entity_id"].to_numpy(dtype=object),
            validation_meta["candidate_entity_id"].to_numpy(dtype=object),
            probabilities,
        )
        ablation_rows_all.extend(
            ablation_rows(
                variant_name,
                len(retained),
                removed,
                baseline_iterations,
                training_seconds,
                inference_seconds,
                probabilities,
                y_validation,
                ranks,
                validation_meta,
                misses,
            )
        )
        del variant_x_train, variant_x_validation, probabilities, ranks
        gc.collect()

    rank_ablation = pd.DataFrame(ablation_rows_all)
    feature_index = {feature: index for index, feature in enumerate(features)}
    error_analysis = pd.DataFrame(
        error_aggregate_rows(
            y_validation,
            full_probabilities,
            full_ranks,
            selected_mask,
            validation_meta,
            x_validation,
            feature_index,
            misses,
        )
    )
    log("Collecting representative retrieved and unreachable hard cases")
    retrieved_cases = retrieved_error_case_sample(
        y_validation,
        full_probabilities,
        full_ranks,
        selected_mask,
        validation_meta,
        x_validation,
        feature_index,
        args.data_dir,
    )
    if len(misses):
        missed_cases = missed_error_case_sample(misses, args.data_dir)
        error_cases = pd.DataFrame.from_records(
            retrieved_cases.to_dict("records") + missed_cases.to_dict("records"),
            columns=retrieved_cases.columns,
        )
    else:
        error_cases = retrieved_cases.copy()

    comparison.to_csv(args.output_dir / "task4b_decision_rule_comparison.csv", index=False)
    rank_ablation.to_csv(args.output_dir / "task4b_rank_ablation.csv", index=False)
    source_analysis.to_csv(args.output_dir / "task4b_source_analysis.csv", index=False)
    group_analysis.to_csv(args.output_dir / "task4b_match_group_analysis.csv", index=False)
    error_analysis.to_csv(args.output_dir / "task4b_error_analysis.csv", index=False)
    error_cases.to_csv(args.output_dir / "task4b_error_case_sample.csv", index=False)
    reliability.to_csv(args.output_dir / "task4b_score_reliability.csv", index=False)

    selected_policy_payload = {
        "rule_name": selected_policy["rule_name"],
        "rule_family": selected_policy["rule_family"],
        "parameters": json.loads(str(selected_policy["parameters"])),
        "selection_method": selected_policy["selection_method"],
        "validation_metrics": {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in selected_policy.items()
            if key
            not in {
                "rule_name",
                "rule_family",
                "parameters",
                "selection_method",
            }
        },
        "inference_inputs": ["LightGBM pair score"],
        "forbidden_inputs_confirmed_absent": ["match_group", "ground_truth_match_count", "label"],
        "validation_only_selection": True,
    }
    (args.output_dir / "task4b_selected_policy.json").write_text(
        json.dumps(selected_policy_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    hashes_after = hash_inputs(paths)
    integrity = {
        "task2_5_candidate_artifacts_unchanged": hashes_before["candidate_checkpoint"] == hashes_after["candidate_checkpoint"],
        "task3_pair_features_unchanged": hashes_before["pairs"] == hashes_after["pairs"],
        "task3_schema_unchanged": hashes_before["schema"] == hashes_after["schema"],
        "entity_split_unchanged": hashes_before["entity_split"] == hashes_after["entity_split"],
        "task4a_predictions_unchanged": hashes_before["task4a_predictions"] == hashes_after["task4a_predictions"],
        "task4a_model_unchanged": hashes_before["task4a_model"] == hashes_after["task4a_model"],
        "task4a_manifest_unchanged": hashes_before["task4a_manifest"] == hashes_after["task4a_manifest"],
        "same_entity_split_counts": (
            True
            if args.smoke_only
            else data_audit["train_entities"] == 8000 and data_audit["validation_entities"] == 2000
        ),
        "zero_s1_overlap": data_audit["entity_overlap"] == 0,
        "no_test_data_accessed": True,
        "no_ground_truth_feature_used": not bool(set(features) & {"label", "match_group"}),
        "match_group_not_used_by_rule": "match_group" not in json.dumps(selected_policy_payload["parameters"]),
        "candidate_misses_reconciled": len(misses) == 239 if not args.smoke_only else True,
        "source_totals_reconcile": int(
            source_analysis.loc[
                source_analysis["policy_role"].eq("selected"),
                "candidate_generation_missed_links",
            ].sum()
        ) == len(misses),
        "group_totals_reconcile": int(
            group_analysis.loc[
                group_analysis["policy_role"].eq("selected"),
                "candidate_generation_missed_links",
            ].sum()
        ) == len(misses),
    }
    if not all(integrity.values()):
        failed = [name for name, status in integrity.items() if not status]
        raise AssertionError(f"Task 4B integrity checks failed: {failed}")

    runtime_seconds = time.perf_counter() - start_total
    peak_memory = t4a.peak_rss_mb()
    summary = build_summary(
        rank_ablation,
        comparison,
        selected_policy_payload,
        source_analysis,
        group_analysis,
        error_analysis,
        calibration,
        integrity,
        runtime_seconds,
        peak_memory,
    )
    (args.output_dir / "task4b_summary.md").write_text(summary, encoding="utf-8")

    manifest = {
        "task": "Task 4B",
        "random_state": RANDOM_STATE,
        "training_only": True,
        "smoke_only": args.smoke_only,
        "model_family": "LightGBM only",
        "baseline_reused_without_retraining": True,
        "fixed_ablation_iterations": baseline_iterations,
        "ablation_variants": {
            "full_66_existing": [],
            **{name: removed for name, removed in variants},
        },
        "decision_rules_evaluated": int(len(comparison)),
        "selected_policy": selected_policy_payload,
        "calibration_diagnostics": calibration,
        "integrity_checks": integrity,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "runtime_seconds": runtime_seconds,
        "peak_rss_mb": peak_memory,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
        "candidate_generation_modified": False,
        "task3_features_modified": False,
        "task4a_artifacts_modified": False,
        "test_data_used": False,
        "test_predictions_created": False,
        "submission_created": False,
        "full_scale_inference_started": False,
    }
    (args.output_dir / "task4b_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    selected = comparison.loc[comparison["selected"]].iloc[0]
    log(
        f"Task 4B complete: selected={selected.rule_name}, "
        f"precision={selected.precision:.4%}, retrieved recall={selected.recall_retrieved:.4%}, "
        f"end-to-end complete={selected.complete_entity_match_rate_end_to_end:.4%}, "
        f"runtime={runtime_seconds / 60.0:.2f} min, peak RSS={peak_memory:.1f} MB"
    )


if __name__ == "__main__":
    main()
