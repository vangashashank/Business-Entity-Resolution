"""Score saved validation candidates and extract deterministic FP/FN examples.

This module deliberately loads an existing candidate TSV. It never imports or
invokes blocking code, so feature/model/threshold iterations reuse Phase 1 output.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import FEATURE_COLUMNS_V2  # noqa: E402
from src.preprocessing import preprocess_dataframe  # noqa: E402
from src.scoring import macro_metrics, per_entity_scores  # noqa: E402

TRAIN_DIR = os.path.join(ROOT, "student_resource", "dataset", "train")


def _read(name: str, usecols=None) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(TRAIN_DIR, name), sep="\t", usecols=usecols,
        dtype={"entity_id": str, "source1_entity_id": str,
               "matched_entity_ids": str, "country": str},
    )


def _read_ids(name: str, wanted: set[str]) -> pd.DataFrame:
    """Read just requested records, scanning source TSVs in bounded chunks."""
    path = os.path.join(TRAIN_DIR, name)
    pieces = []
    for chunk in pd.read_csv(
        path, sep="\t", dtype={"entity_id": str, "country": str},
        usecols=["entity_id", "business_name", "business_address", "country"],
        chunksize=500_000,
    ):
        selected = chunk[chunk.entity_id.isin(wanted)]
        if not selected.empty:
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "country"]
    )


def _id_set(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _load_validation_data(candidate_path: str):
    candidate_rows = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "candidate_entity_ids"}
    if missing := required - set(candidate_rows.columns):
        raise ValueError(f"Candidate file missing columns: {sorted(missing)}")
    if candidate_rows.source1_entity_id.duplicated().any():
        raise ValueError("Candidate file has duplicate S1 rows")

    sids = set(candidate_rows.source1_entity_id)
    gt = _read("train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    truth = gt[gt.source1_entity_id.isin(sids)].copy()
    if set(truth.source1_entity_id) != sids:
        raise ValueError("Candidate S1 IDs do not match rows in train_ground_truth.tsv")

    candidate_ids = {
        mid for value in candidate_rows.candidate_entity_ids
        for mid in value.split(",") if mid
    }
    true_ids = {mid for value in truth.matched_entity_ids for mid in _id_set(value)}
    needed_ids = candidate_ids | true_ids

    s1 = _read_ids("train_source1.tsv", sids)
    s2_all = _read_ids("train_source2.tsv", needed_ids)
    s3_all = _read_ids("train_source3.tsv", needed_ids)
    pool = pd.concat([
        s2_all[s2_all.entity_id.isin(needed_ids)],
        s3_all[s3_all.entity_id.isin(needed_ids)],
    ]).drop_duplicates("entity_id").copy()
    if len(pool) != len(needed_ids):
        missing_ids = needed_ids - set(pool.entity_id)
        raise ValueError(f"Could not load {len(missing_ids)} candidate/true-match records")

    return (
        preprocess_dataframe(s1), preprocess_dataframe(pool),
        candidate_rows, truth,
    )


def _candidate_pairs(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        (row.source1_entity_id, mid)
        for row in candidate_rows.itertuples(index=False)
        for mid in row.candidate_entity_ids.split(",") if mid
    ]
    return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])


def _candidate_recall(candidate_rows: pd.DataFrame, truth: pd.DataFrame) -> float:
    candidates = {
        row.source1_entity_id: set(row.candidate_entity_ids.split(",")) - {""}
        for row in candidate_rows.itertuples(index=False)
    }
    total = hits = 0
    for row in truth.itertuples(index=False):
        true_ids = _id_set(row.matched_entity_ids)
        total += len(true_ids)
        hits += len(true_ids & candidates.get(row.source1_entity_id, set()))
    return hits / total if total else 0.0


_URL_RE = re.compile(r"(?:https?://|www\.|\.(?:com|net|org|in)\b)", re.I)
_ADDRESS_ABBREVIATION_PAIRS = (
    (re.compile(r"\brd\b", re.I), re.compile(r"\broad\b", re.I)),
    (re.compile(r"\bst\b", re.I), re.compile(r"\bstreet\b", re.I)),
    (re.compile(r"\bave\b", re.I), re.compile(r"\bavenue\b", re.I)),
    (re.compile(r"\bcir\b", re.I), re.compile(r"\bcircle\b", re.I)),
    (re.compile(r"\bapt\b", re.I), re.compile(r"\bapartment\b", re.I)),
    (re.compile(r"\bste\b", re.I), re.compile(r"\bsuite\b", re.I)),
    (re.compile(r"\bflr\b", re.I), re.compile(r"\bfloor\b", re.I)),
)


def _has_address_abbreviation_difference(left: str, right: str) -> bool:
    for short, long in _ADDRESS_ABBREVIATION_PAIRS:
        if ((short.search(left) and long.search(right))
                or (long.search(left) and short.search(right))):
            return True
    return False


def _scripts(value: str) -> set[str]:
    scripts = set()
    for char in value:
        if char.isalpha():
            name = unicodedata.name(char, "")
            if name:
                scripts.add(name.split(" ", 1)[0])
    return scripts


def _pattern(row: dict, feature: dict) -> str:
    name_a, name_b = str(row["source1_name"] or ""), str(row["other_name"] or "")
    addr_a, addr_b = str(row["source1_address"] or ""), str(row["other_address"] or "")
    if _URL_RE.search(name_a) or _URL_RE.search(name_b):
        return "URL/domain noise"
    scripts_a, scripts_b = _scripts(name_a), _scripts(name_b)
    if scripts_a and scripts_b and scripts_a.isdisjoint(scripts_b):
        return "Transliteration / script difference"
    if feature.get("abbreviation_normalized_match", 0) == 1:
        return "Legal suffix variation / shared suffix-stripped name"
    if (feature.get("name_jaccard", 0) >= 0.7
            and feature.get("name_levenshtein_ratio", 1) < 0.88):
        return "Word-order variation"
    if _has_address_abbreviation_difference(addr_a, addr_b):
        return "Address abbreviation variation"
    if (not addr_a.strip() or not addr_b.strip()
            or feature.get("address_component_count_diff", 0) >= 1):
        return "Missing / differing address components"
    if (feature.get("numeric_token_match", 0) == 1
            and feature.get("name_tfidf_cosine", 0) < 0.35):
        return "Shared numeric token / possible address-key collision"
    if (0.55 <= feature.get("name_levenshtein_ratio", 0) < 0.99
            or 0.3 <= feature.get("name_tfidf_cosine", 0) < 0.85):
        return "Spelling / typo variation"
    if (feature.get("address_component_count_diff", 0) >= 1
            or feature.get("address_tfidf_cosine", 0) < 0.25):
        return "Missing PIN / landmark / address mismatch"
    return "Other / mixed noise"


def _feature_comment(error_type: str, pattern: str, feature: dict,
                     probability: float, threshold: float) -> str:
    notes = []
    if feature.get("name_tfidf_cosine", 0) < 0.25:
        notes.append("weak name character similarity")
    if feature.get("name_jaccard", 0) < 0.20:
        notes.append("little name token overlap")
    if feature.get("address_tfidf_cosine", 0) < 0.25:
        notes.append("weak address character similarity")
    if feature.get("address_component_count_diff", 0) >= 2:
        notes.append("large address component-count difference")
    if feature.get("numeric_token_match", 0) == 1 and feature.get("name_tfidf_cosine", 0) < 0.35:
        notes.append("numeric-address match may be nonspecific")
    if feature.get("abbreviation_normalized_match", 0) == 1:
        notes.append("suffix-stripped names match exactly")
    if not notes:
        notes.append("no single feature is an obvious outlier; inspect the feature vector")
    if error_type == "blocking miss":
        notes.append("pair absent from saved candidates; reported model score is counterfactual")
    elif abs(probability - threshold) <= 0.05:
        notes.append("within 0.05 of threshold; threshold-sensitive")
    elif error_type == "false positive":
        notes.append("candidate scored above threshold despite not being a true match")
    else:
        notes.append("true candidate scored below threshold")
    return "; ".join(notes)


def _make_case(sid: str, mid: str, error_type: str, probability: float,
               feature: dict, s1_lookup: pd.DataFrame, pool_lookup: pd.DataFrame,
               threshold: float) -> dict:
    a, b = s1_lookup.loc[sid], pool_lookup.loc[mid]
    row = {
        "source1_entity_id": sid,
        "other_entity_id": mid,
        "error_type": error_type,
        "diagnosis": (
            "blocking problem" if error_type == "blocking miss" else
            "threshold problem" if abs(probability - threshold) <= 0.05 else
            "modeling problem"
        ),
        "source1_name": a.business_name,
        "other_name": b.business_name,
        "source1_address": a.business_address,
        "other_address": b.business_address,
        "model_score": probability,
        "pattern": "",
    }
    for name in FEATURE_COLUMNS_V2:
        row[name] = feature.get(name, np.nan)
    row["pattern"] = _pattern(row, feature)
    row["feature_diagnosis"] = _feature_comment(error_type, row["pattern"], feature,
                                                 probability, threshold)
    return row


def analyze(candidate_file: str = "output/candidate_pairs_val.tsv",
            model_file: str = "artifacts/random_forest_v2.joblib",
            threshold: float = 0.70,
            prediction_file: str = "output/validation_predictions_phase4.tsv",
            error_file: str = "output/phase4_error_samples.tsv",
            sample_size: int = 20,
            random_state: int = 42):
    candidate_path = os.path.join(ROOT, candidate_file)
    model_path = os.path.join(ROOT, model_file)
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(
            f"Saved validation candidates not found: {candidate_path}. "
            "Run blocking evaluation explicitly only if blocking changed."
        )
    print(f"Reusing saved candidates: {os.path.relpath(candidate_path, ROOT)}")
    s1, pool, candidate_rows, truth = _load_validation_data(candidate_path)
    pairs = _candidate_pairs(candidate_rows)
    artifact = joblib.load(model_path)
    extractor, model = artifact["feature_extractor"], artifact["model"]

    print(f"Extracting features for {len(pairs):,} saved candidate pairs...")
    features = extractor.transform(s1, pool, pairs)
    probabilities = model.predict_proba(features)[:, 1]
    feature_values = features.to_dict(orient="records")
    truth_map = dict(zip(truth.source1_entity_id,
                         (_id_set(value) for value in truth.matched_entity_ids)))
    candidate_map = {
        row.source1_entity_id: set(row.candidate_entity_ids.split(",")) - {""}
        for row in candidate_rows.itertuples(index=False)
    }

    matched = {sid: [] for sid in candidate_rows.source1_entity_id}
    false_positives = []
    model_false_negatives = []
    for (sid, mid), probability, feature in zip(
        pairs.itertuples(index=False, name=None), probabilities, feature_values
    ):
        is_true = mid in truth_map.get(sid, set())
        if probability >= threshold:
            matched[sid].append(mid)
            if not is_true:
                false_positives.append((sid, mid, float(probability), feature))
        elif is_true:
            model_false_negatives.append((sid, mid, float(probability), feature))

    predictions = pd.DataFrame({
        "source1_entity_id": list(matched),
        "matched_entity_ids": [",".join(sorted(set(matched[sid]))) for sid in matched],
    })
    metrics = macro_metrics(predictions, truth)
    breakdown = per_entity_scores(predictions, truth)
    recall_candidates = _candidate_recall(candidate_rows, truth)

    blocking_misses = []
    for sid, true_ids in truth_map.items():
        missing = true_ids - candidate_map.get(sid, set())
        blocking_misses.extend((sid, mid) for mid in missing)

    rng = np.random.RandomState(random_state)
    fp_sample = (rng.choice(len(false_positives), size=min(sample_size, len(false_positives)),
                 replace=False) if false_positives else [])
    cases = []
    s1_lookup = s1.set_index("entity_id")
    pool_lookup = pool.set_index("entity_id")
    for idx in fp_sample:
        sid, mid, probability, feature = false_positives[int(idx)]
        cases.append(_make_case(sid, mid, "false positive", probability, feature,
                                s1_lookup, pool_lookup, threshold))

    # Reserve up to half the FN sample for blocking misses so both causes are visible.
    half = sample_size // 2
    selected_blocking = []
    if blocking_misses:
        selected_blocking = rng.choice(
            len(blocking_misses), size=min(half, len(blocking_misses)), replace=False
        ).tolist()
    selected_model = []
    model_quota = sample_size - len(selected_blocking)
    if model_false_negatives and model_quota:
        selected_model = rng.choice(
            len(model_false_negatives),
            size=min(model_quota, len(model_false_negatives)), replace=False,
        ).tolist()

    blocked_pairs = [blocking_misses[int(i)] for i in selected_blocking]
    if blocked_pairs:
        counterfactual_pairs = pd.DataFrame(
            blocked_pairs, columns=["source1_entity_id", "candidate_entity_id"]
        )
        counterfactual_features = extractor.transform(s1, pool, counterfactual_pairs)
        counterfactual_probs = model.predict_proba(counterfactual_features)[:, 1]
        for (sid, mid), probability, feature in zip(
            blocked_pairs, counterfactual_probs,
            counterfactual_features.to_dict(orient="records"),
        ):
            cases.append(_make_case(sid, mid, "blocking miss", float(probability), feature,
                                    s1_lookup, pool_lookup, threshold))

    for idx in selected_model:
        sid, mid, probability, feature = model_false_negatives[int(idx)]
        cases.append(_make_case(sid, mid, "model false negative", probability, feature,
                                s1_lookup, pool_lookup, threshold))

    prediction_path = os.path.join(ROOT, prediction_file)
    error_path = os.path.join(ROOT, error_file)
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    os.makedirs(os.path.dirname(error_path), exist_ok=True)
    predictions.to_csv(prediction_path, sep="\t", index=False)
    pd.DataFrame(cases).to_csv(error_path, sep="\t", index=False)

    print(f"Validation S1 entities: {len(candidate_rows):,}; "
          f"true match pairs: {sum(map(len, truth_map.values())):,}")
    print(f"Candidate recall: {recall_candidates:.4%}")
    print(f"Validation macro F0.5={metrics['val_f0_5']:.6f}, "
          f"precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}; "
          f"threshold={threshold:.2f}")
    print(f"False positive pairs: {len(false_positives):,}; "
          f"model false negative pairs: {len(model_false_negatives):,}; "
          f"blocking false negative pairs: {len(blocking_misses):,}")
    print("Per-entity F0.5 summary:")
    print(breakdown.f0_5.describe().to_string())
    print("Sampled case patterns:")
    if cases:
        print(pd.DataFrame(cases).groupby(["error_type", "pattern"]).size().to_string())
    print(f"Predictions: {os.path.relpath(prediction_path, ROOT)}")
    print(f"Error cases: {os.path.relpath(error_path, ROOT)}")
    return metrics, pd.DataFrame(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", default="output/candidate_pairs_val.tsv")
    parser.add_argument("--model", default="artifacts/random_forest_v2.joblib")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--prediction-output", default="output/validation_predictions_phase4.tsv")
    parser.add_argument("--error-output", default="output/phase4_error_samples.tsv")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of FP examples and total FN examples to write")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.candidate_file, args.model, args.threshold,
            args.prediction_output, args.error_output, args.sample_size, args.seed)
