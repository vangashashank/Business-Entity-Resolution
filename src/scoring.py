"""Per-S1 macro F0.5 scorer with explicit singleton behavior."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _id_set(value: object) -> set[str]:
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _f05(precision: float, recall: float) -> float:
    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall / denominator) if denominator else 0.0


def per_entity_scores(predictions_df: pd.DataFrame,
                      ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Return one precision/recall/F0.5 row per ground-truth S1 entity.

    Empty prediction against empty truth scores 1.0. Either empty side when the
    other is non-empty scores 0.0, matching the challenge singleton cases.
    """
    required = {"source1_entity_id", "matched_entity_ids"}
    for frame, label in ((predictions_df, "predictions"), (ground_truth_df, "ground truth")):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
        if frame.source1_entity_id.isna().any():
            raise ValueError(f"{label} contains null source1_entity_id")
        if frame.source1_entity_id.duplicated().any():
            raise ValueError(f"{label} contains duplicate source1_entity_id rows")

    pred_map = dict(zip(predictions_df.source1_entity_id.astype(str), predictions_df.matched_entity_ids))
    truth_map = dict(zip(ground_truth_df.source1_entity_id.astype(str), ground_truth_df.matched_entity_ids))
    extra = set(pred_map) - set(truth_map)
    if extra:
        raise ValueError(f"Predictions contain S1 IDs absent from ground truth: {sorted(extra)[:5]}")

    rows = []
    for sid, raw_truth in truth_map.items():
        truth = _id_set(raw_truth)
        predicted = _id_set(pred_map.get(sid))
        tp = len(truth & predicted)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        if not truth and not predicted:
            precision = recall = entity_f05 = 1.0
        elif not predicted or not truth:
            precision = recall = entity_f05 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            entity_f05 = _f05(precision, recall)
        rows.append({
            "source1_entity_id": sid,
            "precision": precision,
            "recall": recall,
            "f0_5": entity_f05,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    return pd.DataFrame(rows)


def score(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """Exact macro-averaged per-entity F0.5."""
    breakdown = per_entity_scores(predictions_df, ground_truth_df)
    return float(breakdown.f0_5.mean()) if len(breakdown) else 0.0


def macro_metrics(predictions_df: pd.DataFrame,
                  ground_truth_df: pd.DataFrame) -> dict[str, float]:
    """Return macro F0.5, precision, and recall for experiment reporting."""
    breakdown = per_entity_scores(predictions_df, ground_truth_df)
    if breakdown.empty:
        return {"val_f0_5": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "val_f0_5": float(breakdown.f0_5.mean()),
        "precision": float(breakdown.precision.mean()),
        "recall": float(breakdown.recall.mean()),
    }


def _self_check() -> None:
    predictions = pd.DataFrame({
        "source1_entity_id": ["A", "B", "C"],
        "matched_entity_ids": ["x", "", "q"],
    })
    truth = pd.DataFrame({
        "source1_entity_id": ["A", "B", "C"],
        "matched_entity_ids": ["x,y", "", "z"],
    })
    hand = pd.DataFrame({
        "source1_entity_id": ["A", "B", "C"],
        "precision": [1.0, 1.0, 0.0],
        "recall": [0.5, 1.0, 0.0],
        "f0_5": [5.0 / 6.0, 1.0, 0.0],
    })
    actual_rows = per_entity_scores(predictions, truth)
    actual = score(predictions, truth)
    expected = float(hand.f0_5.mean())
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"Scorer self-check failed: got {actual}, expected {expected}")
    for metric in ("precision", "recall", "f0_5"):
        if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                   for a, b in zip(hand[metric], actual_rows[metric])):
            raise AssertionError(f"Scorer per-entity {metric} differs from hand calculation")

    shown = actual_rows[["source1_entity_id", "precision", "recall", "f0_5"]].copy()
    for metric in ("precision", "recall", "f0_5"):
        shown.insert(len(shown.columns), f"hand_{metric}", hand[metric])
    print("Toy check: hand-computed values alongside scorer-computed values")
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Macro F0.5: hand={expected:.6f}; scorer={actual:.6f} — PASS")


if __name__ == "__main__":
    _self_check()
