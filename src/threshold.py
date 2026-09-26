"""Sweep model probability thresholds against the local validation scorer."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.compare_models import load_validation_sample  # noqa: E402
from src.scoring import macro_metrics  # noqa: E402

DEFAULT_THRESHOLDS = tuple(
    [round(0.30 + 0.05 * step, 2) for step in range(13)]
    + [round(0.90 + 0.01 * step, 2) for step in range(1, 10)]
)
PALETTE = {
    "correct": "#2E9E7A",
    "error": "#E4572E",
    "miss": "#F2A93B",
    "neutral": "#3B5BA5",
    "background": "#F5F3EF",
    "grid": "#DAD6CE",
}


def _prediction_frame(candidate_rows: pd.DataFrame,
                      pairs: pd.DataFrame,
                      probabilities: np.ndarray,
                      threshold: float) -> pd.DataFrame:
    if len(pairs) != len(probabilities):
        raise ValueError("pairs and probabilities must have the same number of rows")
    matches = {sid: [] for sid in candidate_rows.source1_entity_id}
    for sid, mid, probability in zip(
        pairs.source1_entity_id, pairs.candidate_entity_id, probabilities
    ):
        if probability >= threshold:
            matches[sid].append(mid)
    return pd.DataFrame({
        "source1_entity_id": list(matches),
        "matched_entity_ids": [",".join(sorted(set(matches[sid]))) for sid in matches],
    })


def sweep_thresholds(probabilities: Iterable[float],
                     pairs: pd.DataFrame,
                     candidate_rows: pd.DataFrame,
                     ground_truth: pd.DataFrame,
                     thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Return macro F0.5, precision, and recall at each probability threshold."""
    probabilities = np.asarray(list(probabilities), dtype=float)
    if len(probabilities) != len(pairs):
        raise ValueError("pairs and probabilities must have the same number of rows")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    values = [float(value) for value in thresholds]
    if not values:
        raise ValueError("at least one threshold is required")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("thresholds must be in [0, 1]")

    rows = []
    for threshold in values:
        predictions = _prediction_frame(candidate_rows, pairs, probabilities, threshold)
        metrics = macro_metrics(predictions, ground_truth)
        rows.append({"threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def plot_threshold_sweep(results: pd.DataFrame, output_path: str) -> str:
    """Plot validation F0.5 and mark the maximum with a dashed vertical line."""
    if results.empty or not {"threshold", "val_f0_5"}.issubset(results.columns):
        raise ValueError("results must contain threshold and val_f0_5 columns")
    best_idx = results.val_f0_5.idxmax()
    best = results.loc[best_idx]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.plot(
        results.threshold, results.val_f0_5,
        color=PALETTE["neutral"], marker="o", linewidth=2,
        label="Validation macro F0.5",
    )
    ax.axvline(
        float(best.threshold), color=PALETTE["error"], linestyle="--", linewidth=1.8,
        label=f"Best threshold = {best.threshold:.2f}",
    )
    baseline = results[np.isclose(results.threshold, 0.5)]
    if not baseline.empty:
        ax.scatter(
            [0.5], [float(baseline.iloc[0].val_f0_5)],
            color=PALETTE["correct"], marker="s", s=65, zorder=3,
            label="Current threshold = 0.50",
        )
    ax.set_xlabel("Match probability threshold")
    ax.set_ylabel("Macro F0.5")
    ax.set_title("Validation threshold sweep")
    ax.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    low = max(0.0, float(results.val_f0_5.min()) - 0.02)
    high = min(1.0, float(results.val_f0_5.max()) + 0.01)
    ax.set_ylim(low, high)
    ax.grid(color=PALETTE["grid"])
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_abs = os.path.join(ROOT, output_path)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    fig.savefig(output_abs, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_abs


def run_sweep(model_path: str = "artifacts/random_forest_v2.joblib",
              candidate_file: str = "output/candidate_pairs_val.tsv",
              plot_path: str = "output/threshold_sweep_rf_v2.png",
              results_path: str = "output/threshold_sweep_rf_v2.tsv") -> pd.DataFrame:
    artifact_path = os.path.join(ROOT, model_path)
    artifact = joblib.load(artifact_path)
    s1, pool, candidate_rows, ground_truth = load_validation_sample(candidate_file)
    pairs = pd.DataFrame(
        [(row.source1_entity_id, mid)
         for row in candidate_rows.itertuples(index=False)
         for mid in row.candidate_entity_ids.split(",") if mid],
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    if pairs.empty:
        raise ValueError("validation candidate file contains no candidate pairs")

    features = artifact["feature_extractor"].transform(s1, pool, pairs)
    probabilities = artifact["model"].predict_proba(features)[:, 1]
    results = sweep_thresholds(probabilities, pairs, candidate_rows, ground_truth)
    best = results.loc[results.val_f0_5.idxmax()]
    baseline_rows = results[np.isclose(results.threshold, 0.5)]
    if baseline_rows.empty:
        raise ValueError("threshold grid must include 0.5 for the baseline comparison")
    baseline = baseline_rows.iloc[0]
    plot_abs = plot_threshold_sweep(results, plot_path)
    results_abs = os.path.join(ROOT, results_path)
    os.makedirs(os.path.dirname(results_abs), exist_ok=True)
    results.to_csv(results_abs, sep="\t", index=False)

    print(f"Model: {model_path}")
    print(f"Validation S1 entities: {len(candidate_rows):,}")
    print(f"Threshold 0.50 baseline: F0.5={baseline.val_f0_5:.6f}, "
          f"precision={baseline.precision:.6f}, recall={baseline.recall:.6f}")
    print(f"Best threshold {best.threshold:.2f}: F0.5={best.val_f0_5:.6f}, "
          f"precision={best.precision:.6f}, recall={best.recall:.6f}")
    print(f"F0.5 change vs 0.50: {best.val_f0_5 - baseline.val_f0_5:+.6f}")
    print("Full threshold curve (all metrics are macro-averaged across validation S1s):")
    print(results.to_string(index=False, formatters={
        "threshold": "{:.2f}".format,
        "val_f0_5": "{:.6f}".format,
        "precision": "{:.6f}".format,
        "recall": "{:.6f}".format,
    }))
    print(f"Saved plot: {os.path.relpath(plot_abs, ROOT)}")
    print(f"Saved curve data: {os.path.relpath(results_abs, ROOT)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/random_forest_v2.joblib",
                        help="Trained model artifact (default: selected Random Forest v2)")
    parser.add_argument("--candidate-file", default="output/candidate_pairs_val.tsv",
                        help="Validation candidates; include singleton S1 rows")
    parser.add_argument("--plot", default="output/threshold_sweep_rf_v2.png")
    parser.add_argument("--results", default="output/threshold_sweep_rf_v2.tsv",
                        help="Write full threshold metrics as a TSV")
    args = parser.parse_args()
    run_sweep(args.model, args.candidate_file, args.plot, args.results)
