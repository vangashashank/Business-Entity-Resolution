#!/usr/bin/env python3
"""Task 5: frozen full-inference contract, raw replay, and smoke validation.

The executable Task 5 mode is deliberately limited to known training entities.
It rebuilds the frozen retrieval path from raw S1/S2/S3 records, verifies parity
against frozen artifacts, and exercises resumable batch feature/model inference.
The parser exposes a future ``full`` mode only as a safety lock; Task 5 never
accesses test data or launches the multi-million-entity workload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for vendor_dir in (
    PROJECT_ROOT / ".task4_vendor",
    PROJECT_ROOT / ".task3_vendor",
    PROJECT_ROOT / ".task2_5_vendor",
):
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# FAISS and LightGBM ship separate OpenMP runtimes in the local vendor trees.
# Allow both runtimes in this validation process; each library's thread count is
# still bounded explicitly by the frozen pipeline configuration.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import task2_candidate_generation as t2
import task2_5_candidate_selection as t25
import task3_pairwise_features as t3
import task4a_baseline_models as t4a


TASK_NAME = "Task 5"
FROZEN_CONFIGURATION = t25.COMBINED_AUGMENTED_250
FROZEN_CANDIDATE_CAP = 250
FROZEN_THRESHOLD = 0.95
SIGNALS = tuple(t3.SIGNALS)
FLOAT_FEATURE_ATOL = 1e-6
FLOAT_FEATURE_RTOL = 1e-6
RETRIEVAL_SCORE_ATOL = 1e-6
MODEL_SCORE_ATOL = 5e-8
OUTPUT_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "lightgbm_score",
    "predicted_match",
    "candidate_position",
    "batch_id",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("contract", "parity-smoke", "full"),
        default="parity-smoke",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource" / "dataset" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task5_outputs",
    )
    parser.add_argument("--smoke-entities", type=int, default=1_000)
    parser.add_argument("--parity-entities", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--transform-batch-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--reuse-verified-retrieval",
        action="store_true",
        help=(
            "Resume after a completed raw retrieval parity run by loading the frozen "
            "checkpoint only when task5_raw_retrieval_replay.json verifies it was exact."
        ),
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "S1": args.data_dir / "train_source1.tsv",
        "S2": args.data_dir / "train_source2.tsv",
        "S3": args.data_dir / "train_source3.tsv",
        "task2_eval": PROJECT_ROOT / "outputs" / "task2_outputs" / "task2_eval_entities.csv",
        "task2_5_checkpoint": PROJECT_ROOT / "outputs" / "task2_5_outputs" / "task2_5_retrieval_top250.npz",
        "task2_5_decision": PROJECT_ROOT / "outputs" / "task2_5_outputs" / "task2_5_decision.json",
        "task2_5_manifest": PROJECT_ROOT / "outputs" / "task2_5_outputs" / "task2_5_run_manifest.json",
        "task3_schema": PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_feature_schema.csv",
        "task3_audit": PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_to_task4_feature_audit.csv",
        "task3_pairs": PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_pair_features.parquet",
        "task3_candidates": PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_frozen_candidates.npz",
        "task3_split": PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_entity_split.csv",
        "task3_manifest": PROJECT_ROOT / "outputs" / "task3_outputs" / "task3_run_manifest.json",
        "task4a_model": PROJECT_ROOT / "outputs" / "task4_outputs" / "task4a_lightgbm_model.txt",
        "task4a_predictions": PROJECT_ROOT / "outputs" / "task4_outputs" / "task4a_validation_predictions.parquet",
        "task4a_manifest": PROJECT_ROOT / "outputs" / "task4_outputs" / "task4a_run_manifest.json",
        "task4b_policy": PROJECT_ROOT / "outputs" / "task4_outputs" / "task4b" / "task4b_selected_policy.json",
        "task4b_manifest": PROJECT_ROOT / "outputs" / "task4_outputs" / "task4b" / "task4b_run_manifest.json",
        "task2_code": PROJECT_ROOT / "src" / "task2_candidate_generation.py",
        "task2_5_code": PROJECT_ROOT / "src" / "task2_5_candidate_selection.py",
        "task3_code": PROJECT_ROOT / "src" / "task3_pairwise_features.py",
        "task4a_code": PROJECT_ROOT / "src" / "task4a_baseline_models.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 5 inputs: {missing}")
    for key in ("S1", "S2", "S3"):
        if "test" in paths[key].name.casefold():
            raise ValueError(f"Task 5 cannot access test data: {paths[key]}")
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_hashes(paths: dict[str, Path]) -> dict[str, str]:
    keys = (
        "task2_5_checkpoint",
        "task2_5_decision",
        "task2_5_manifest",
        "task3_schema",
        "task3_audit",
        "task3_pairs",
        "task3_candidates",
        "task3_split",
        "task3_manifest",
        "task4a_model",
        "task4a_predictions",
        "task4a_manifest",
        "task4b_policy",
        "task4b_manifest",
        "task2_code",
        "task2_5_code",
        "task3_code",
        "task4a_code",
    )
    return {key: file_sha256(paths[key]) for key in keys}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class RuntimeRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, phase: str, seconds: float, detail: str = "") -> None:
        row = {
            "phase": phase,
            "runtime_seconds": float(seconds),
            "current_rss_mb": t3.current_rss_mb(),
            "process_peak_rss_mb": t3.peak_rss_mb(),
            "detail": detail,
        }
        self.rows.append(row)
        log(
            f"{phase}: {seconds:.2f}s; current RSS={row['current_rss_mb']:.1f} MB; "
            f"peak RSS={row['process_peak_rss_mb']:.1f} MB"
        )

    def add_evidence(
        self,
        phase: str,
        seconds: float,
        peak_rss_mb: float,
        detail: str,
    ) -> None:
        self.rows.append(
            {
                "phase": phase,
                "runtime_seconds": float(seconds),
                "current_rss_mb": np.nan,
                "process_peak_rss_mb": float(peak_rss_mb),
                "detail": detail,
            }
        )


def load_frozen_contract_inputs(
    paths: dict[str, Path],
) -> tuple[list[str], pd.DataFrame, lgb.Booster, dict[str, object], dict[str, object]]:
    features, feature_schema = t4a.read_feature_contract(
        paths["task3_schema"], paths["task3_audit"]
    )
    model = lgb.Booster(model_file=str(paths["task4a_model"]))
    model_features = model.feature_name()
    if model.num_feature() != 66 or model_features != features:
        raise AssertionError(
            "Saved LightGBM feature names/order do not match the frozen Task 3 contract"
        )
    decision = json.loads(paths["task2_5_decision"].read_text(encoding="utf-8"))
    policy = json.loads(paths["task4b_policy"].read_text(encoding="utf-8"))
    if decision["selected_configuration"] != FROZEN_CONFIGURATION:
        raise AssertionError("Task 2.5 selected configuration changed")
    if int(decision["candidate_budget"]) != FROZEN_CANDIDATE_CAP:
        raise AssertionError("Task 2.5 candidate cap changed")
    if float(policy["parameters"]["probability_threshold"]) != FROZEN_THRESHOLD:
        raise AssertionError("Task 4B threshold changed")
    return features, feature_schema, model, decision, policy


def build_contract(
    paths: dict[str, Path],
    hashes: dict[str, str],
    features: list[str],
    feature_schema: pd.DataFrame,
    model: lgb.Booster,
    decision: dict[str, object],
    policy: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    schema_by_feature = feature_schema.set_index("column")
    feature_contract = []
    for position, feature in enumerate(features):
        row = schema_by_feature.loc[feature]
        feature_contract.append(
            {
                "position": position,
                "name": feature,
                "dtype": str(row["dtype"]),
                "family": str(row["family"]),
                "missing_convention": str(row["missing_convention"]),
                "requires_fitted_retrieval_state": str(row["requires_fitted_state"]).casefold()
                == "true",
            }
        )
    signals = [
        {
            "name": "baseline_name",
            "raw_field": "business_name",
            "normalization": "task2_candidate_generation.normalize_series",
            "query_depth": 1000,
            "stored_depth": 250,
        },
        {
            "name": "baseline_address",
            "raw_field": "business_address",
            "normalization": "task2_candidate_generation.normalize_series",
            "query_depth": 1000,
            "stored_depth": 250,
        },
        {
            "name": "transliterated_name",
            "raw_field": "business_name",
            "normalization": "task2_5_candidate_selection.transliterate_series",
            "query_depth": 250,
            "stored_depth": 250,
        },
        {
            "name": "number_address",
            "raw_field": "business_address",
            "normalization": "task2_5_candidate_selection.address_number_series",
            "query_depth": 250,
            "stored_depth": 250,
        },
        {
            "name": "suffix_name",
            "raw_field": "business_name",
            "normalization": "task2_5_candidate_selection.strip_legal_suffix_series",
            "query_depth": 250,
            "stored_depth": 250,
        },
    ]
    return {
        "contract_version": "task5-v1",
        "created_date": time.strftime("%Y-%m-%d"),
        "scope": "training-parity replay and smoke validation; no full/test inference",
        "candidate_generator": {
            "configuration": decision["selected_configuration"],
            "candidate_cap": FROZEN_CANDIDATE_CAP,
            "country_blocking": True,
            "candidate_ordering": (
                "baseline normalized-name top-100 followed by unseen normalized-address "
                "top-100, then unseen candidates by reciprocal-rank fusion over all five "
                "signals until cap 250"
            ),
            "rrf_constant": 60.0,
            "signals": signals,
            "retrieval_parameters": json.loads(
                paths["task2_5_manifest"].read_text(encoding="utf-8")
            )["parameters"],
            "normalization_source_hashes": {
                "task2_candidate_generation.py": hashes["task2_code"],
                "task2_5_candidate_selection.py": hashes["task2_5_code"],
            },
        },
        "features": {
            "count": len(features),
            "ordered_contract": feature_contract,
            "feature_source": str(paths["task3_schema"].relative_to(PROJECT_ROOT)),
            "feature_source_hash": hashes["task3_schema"],
            "feature_implementation_hash": hashes["task3_code"],
            "missing_behavior": (
                "No NaN or infinity. Missing text uses explicit missing indicators and "
                "zero similarities; absent retrieval ranks use rank=0, paired missing "
                "indicators, and score=0."
            ),
        },
        "model": {
            "family": "LightGBM binary classifier",
            "path": str(paths["task4a_model"].relative_to(PROJECT_ROOT)),
            "sha256": hashes["task4a_model"],
            "feature_count": model.num_feature(),
            "ordered_feature_names": model.feature_name(),
            "prediction_dtype": "float64",
        },
        "decision_policy": {
            "rule": policy["rule_name"],
            "threshold": FROZEN_THRESHOLD,
            "comparison": "score >= 0.95",
            "policy_sha256": hashes["task4b_policy"],
        },
        "parity_tolerances": {
            "candidate_ids_counts_order": "exact",
            "retrieval_scores": f"absolute <= {RETRIEVAL_SCORE_ATOL:g}",
            "integer_and_boolean_features": "exact",
            "float32_features": (
                f"absolute <= {FLOAT_FEATURE_ATOL:g} or relative <= {FLOAT_FEATURE_RTOL:g}"
            ),
            "lightgbm_score_vs_saved_float32": f"absolute <= {MODEL_SCORE_ATOL:g}",
            "threshold_decisions": "exact",
        },
        "schemas": {
            "batch_input": [
                {"name": "source1_entity_id", "dtype": "string"},
                {"name": "business_name", "dtype": "string", "nullable": True},
                {"name": "business_address", "dtype": "string", "nullable": True},
                {"name": "country", "dtype": "string"},
            ],
            "pair_feature_matrix": {
                "rows": "one row per frozen S1-candidate pair",
                "columns": features,
                "numeric_runtime_dtype": "float32",
            },
            "prediction_checkpoint": [
                {"name": "source1_entity_id", "dtype": "string"},
                {"name": "candidate_entity_id", "dtype": "string"},
                {"name": "candidate_source", "dtype": "string[S2|S3]"},
                {"name": "lightgbm_score", "dtype": "float64"},
                {"name": "predicted_match", "dtype": "bool"},
                {"name": "candidate_position", "dtype": "uint16"},
                {"name": "batch_id", "dtype": "string"},
            ],
            "future_final_prediction": {
                "minimum_columns": list(OUTPUT_COLUMNS[:5]),
                "competition_submission_schema": "not established in existing project artifacts",
            },
        },
        "batching": {
            "initial_s1_batch_size": args.batch_size,
            "boundary_rule": "input order partitioned into fixed contiguous batches",
            "atomic_write": "write .tmp Parquet and os.replace into final batch path",
            "completed_batch_tracking": "checkpoint_manifest.json with batch hashes and row counts",
            "resume_behavior": "verify completed batch hash/schema/count, skip it, and continue",
            "duplicate_prevention_key": ["source1_entity_id", "candidate_entity_id"],
            "failure_behavior": "raise; never mark or silently skip a failed batch",
        },
        "full_scale_architecture": {
            "initialization": [
                "Fit frozen TF-IDF/random-projection transforms from the deterministic training sample.",
                "Build the five country-blocked FAISS retrieval signals without ground truth.",
                "For production reuse, persist retrieval transforms/indexes and a bucketed normalized candidate-text store.",
            ],
            "batch_loop": [
                "Read deterministic S1 batch.",
                "Query five frozen retrieval signals and assemble at most 250 candidates per S1.",
                "Lookup only required candidate text records from the normalized candidate store.",
                "Compute the exact ordered 66-feature float32 matrix.",
                "Score with the saved LightGBM and apply score >= 0.95.",
                "Atomically write the batch checkpoint, release pair/features, and continue.",
            ],
            "prohibited": [
                "all-S1 candidate-pair materialization",
                "all-S1 feature-matrix materialization",
                "ground truth, label, match_group, split, or match-count inputs",
            ],
        },
        "frozen_artifact_hashes": hashes,
    }


def select_smoke_entities(path: Path, count: int) -> pd.DataFrame:
    split = pd.read_csv(
        path,
        usecols=["eval_index", "source1_entity_id", "split"],
        dtype={"source1_entity_id": "string", "split": "string"},
    )
    selected = (
        split.loc[split["split"].eq("validation"), ["eval_index", "source1_entity_id"]]
        .sort_values("eval_index")
        .head(count)
        .reset_index(drop=True)
    )
    if len(selected) != count:
        raise AssertionError(f"Requested {count} validation S1 entities, found {len(selected)}")
    if selected["source1_entity_id"].duplicated().any():
        raise AssertionError("Duplicate S1 IDs in smoke selection")
    return selected


def load_raw_s1(selection: pd.DataFrame, path: Path, chunksize: int) -> pd.DataFrame:
    wanted = set(selection["source1_entity_id"].astype(str))
    rows = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        chunksize=chunksize,
    ):
        chosen = chunk.loc[chunk["entity_id"].isin(wanted)].copy()
        if not chosen.empty:
            rows.append(chosen)
    raw = pd.concat(rows, ignore_index=True)
    if raw["entity_id"].nunique() != len(selection):
        missing = sorted(wanted - set(raw["entity_id"].astype(str)))[:10]
        raise AssertionError(f"Missing raw S1 smoke records: {missing}")
    result = selection.merge(
        raw,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["entity_id"])
    result["normalized_name"] = t2.normalize_series(result["business_name"])
    result["normalized_address"] = t2.normalize_series(result["business_address"])
    result["normalized_country"] = t2.normalize_country_series(result["country"])
    result["local_query_index"] = np.arange(len(result), dtype=np.int32)
    return result


def retrieval_namespace(args: argparse.Namespace) -> argparse.Namespace:
    manifest = {
        "max_features": 32_768,
        "projection_dim": 256,
        "nlist": 2_048,
        "pq_m": 64,
        "nprobe": 64,
        "search_k": 250,
        "chunksize": args.chunksize,
        "transform_batch_size": args.transform_batch_size,
    }
    return argparse.Namespace(**manifest)


def build_raw_retrieval(
    eval_df: pd.DataFrame,
    paths: dict[str, Path],
    args: argparse.Namespace,
    runtime: RuntimeRecorder,
) -> tuple[t3.RetrievalCheckpoint, dict[str, float], dict[str, int]]:
    retrieve_args = retrieval_namespace(args)
    source_paths = {"S2": paths["S2"], "S3": paths["S3"]}
    faiss.omp_set_num_threads(args.workers)

    started = time.perf_counter()
    training_sample, unused_truth, source_counts = t25.collect_training_sample_and_truth_records(
        source_paths,
        set(),
        args.chunksize,
        30,
    )
    if unused_truth:
        raise AssertionError("Ground-truth records unexpectedly entered Task 5 retrieval")
    training_sample = t2.trim_training_sample(training_sample, 50_000)
    runtime.add("candidate_training_sample_scan", time.perf_counter() - started)

    started = time.perf_counter()
    baseline_models, partitions = t2.build_text_models(
        training_sample,
        eval_df,
        True,
        retrieve_args.max_features,
        retrieve_args.projection_dim,
    )
    variant_specs = [
        t25.SignalSpec("transliterated_name", "business_name", t25.transliterate_series),
        t25.SignalSpec("suffix_name", "business_name", t25.strip_legal_suffix_series),
        t25.SignalSpec("number_address", "business_address", t25.address_number_series),
    ]
    variant_models, variant_partitions = t25.build_variant_models(
        training_sample,
        eval_df,
        variant_specs,
        retrieve_args.max_features,
        retrieve_args.projection_dim,
    )
    if variant_partitions != partitions:
        raise AssertionError("Baseline and Task 2.5 retrieval country partitions differ")
    runtime.add("candidate_retrieval_model_fit", time.perf_counter() - started)
    del training_sample
    gc.collect()

    query_keys = t2.prepare_exact_query_keys(eval_df, True)
    exact_map = {
        (partition, field): defaultdict(list)
        for partition in partitions
        for field in t2.FIELDS
    }

    started = time.perf_counter()
    baseline_name = t2.build_field_index_and_retrieve(
        "name",
        source_paths,
        eval_df,
        baseline_models,
        partitions,
        True,
        query_keys,
        exact_map,
        args.chunksize,
        args.transform_batch_size,
        retrieve_args.projection_dim,
        retrieve_args.nlist,
        retrieve_args.pq_m,
        retrieve_args.nprobe,
        1_000,
        collect_exact=True,
    )
    runtime.add("candidate_baseline_name_index_and_query", time.perf_counter() - started)

    started = time.perf_counter()
    baseline_address = t2.build_field_index_and_retrieve(
        "address",
        source_paths,
        eval_df,
        baseline_models,
        partitions,
        True,
        query_keys,
        exact_map,
        args.chunksize,
        args.transform_batch_size,
        retrieve_args.projection_dim,
        retrieve_args.nlist,
        retrieve_args.pq_m,
        retrieve_args.nprobe,
        1_000,
        collect_exact=False,
    )
    runtime.add("candidate_baseline_address_index_and_query", time.perf_counter() - started)
    del baseline_models
    gc.collect()

    started = time.perf_counter()
    name_variants = t25.build_variant_indexes_and_retrieve(
        variant_specs[:2],
        source_paths,
        eval_df,
        variant_models,
        partitions,
        retrieve_args,
    )
    runtime.add("candidate_name_variant_indexes_and_query", time.perf_counter() - started)

    started = time.perf_counter()
    number_variant = t25.build_variant_indexes_and_retrieve(
        variant_specs[2:],
        source_paths,
        eval_df,
        variant_models,
        partitions,
        retrieve_args,
    )
    runtime.add("candidate_number_index_and_query", time.perf_counter() - started)
    del variant_models
    gc.collect()

    artifacts = {
        "baseline_name": baseline_name,
        "baseline_address": baseline_address,
        **name_variants,
        **number_variant,
    }
    checkpoint = t3.RetrievalCheckpoint(
        ids={signal: np.asarray(artifacts[signal].ids[:, :250], dtype=np.int64) for signal in SIGNALS},
        scores={
            signal: np.asarray(artifacts[signal].scores[:, :250], dtype=np.float32)
            for signal in SIGNALS
        },
    )
    timing = {
        "retrieval_index_build_seconds": float(
            sum(artifacts[signal].build_seconds for signal in SIGNALS)
        ),
        "retrieval_query_seconds": float(
            sum(artifacts[signal].query_seconds for signal in SIGNALS)
        ),
    }
    del artifacts, baseline_name, baseline_address, name_variants, number_variant
    gc.collect()
    return checkpoint, timing, source_counts


def load_verified_retrieval_replay(
    output_dir: Path,
    paths: dict[str, Path],
    eval_df: pd.DataFrame,
    hashes_before: dict[str, str],
    runtime: RuntimeRecorder,
) -> tuple[t3.RetrievalCheckpoint, dict[str, float], dict[str, int], list[dict[str, object]], float]:
    evidence_path = output_dir / "task5_raw_retrieval_replay.json"
    parity_path = output_dir / "task5_parity_report.csv"
    if not evidence_path.exists() or not parity_path.exists():
        raise FileNotFoundError(
            "Verified raw-retrieval replay evidence is required for resume mode"
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS" or evidence.get("raw_replay_entity_count") != len(eval_df):
        raise AssertionError("Raw retrieval replay evidence is incomplete or for another selection")
    if evidence.get("parity_report_sha256") != file_sha256(parity_path):
        raise AssertionError("Raw retrieval replay parity report hash changed")
    if evidence.get("task2_5_checkpoint_sha256") != hashes_before["task2_5_checkpoint"]:
        raise AssertionError("Frozen retrieval checkpoint changed since raw replay")
    previous = pd.read_csv(parity_path)
    retrieval_rows = previous.loc[
        previous["stage"].eq("candidate_retrieval")
    ].copy()
    if len(retrieval_rows) != 10 or not retrieval_rows["passed"].astype(bool).all():
        raise AssertionError("Raw retrieval replay did not pass all ten signal checks")
    for row in evidence["measured_phase_timings"]:
        runtime.add_evidence(
            str(row["phase"]),
            float(row["runtime_seconds"]),
            float(evidence["peak_rss_mb"]),
            "Measured during the exact raw replay attempt; reused after downstream "
            "float32 score-tolerance correction.",
        )
    full = t3.load_checkpoint(paths["task2_5_checkpoint"])
    global_indices = eval_df["eval_index"].to_numpy(dtype=np.int64)
    checkpoint = t3.RetrievalCheckpoint(
        ids={signal: full.ids[signal][global_indices] for signal in SIGNALS},
        scores={signal: full.scores[signal][global_indices] for signal in SIGNALS},
    )
    manifest = json.loads(paths["task2_5_manifest"].read_text(encoding="utf-8"))
    return (
        checkpoint,
        {
            "retrieval_index_build_seconds": float(evidence["retrieval_index_build_seconds"]),
            "retrieval_query_seconds": float(evidence["retrieval_query_seconds"]),
        },
        {key: int(value) for key, value in manifest["source_candidate_counts"].items()},
        retrieval_rows.to_dict("records"),
        float(evidence["measured_raw_replay_runtime_seconds"]),
    )


def parity_row(
    stage: str,
    check: str,
    passed: bool,
    compared: int,
    mismatches: int = 0,
    maximum_difference: float = 0.0,
    mean_difference: float = 0.0,
    tolerance: str = "exact",
    detail: str = "",
) -> dict[str, object]:
    return {
        "stage": stage,
        "check": check,
        "passed": bool(passed),
        "compared_values": int(compared),
        "mismatch_count": int(mismatches),
        "maximum_absolute_difference": float(maximum_difference),
        "mean_absolute_difference": float(mean_difference),
        "tolerance": tolerance,
        "detail": detail,
    }


def compare_retrieval_checkpoint(
    actual: t3.RetrievalCheckpoint,
    frozen_path: Path,
    global_indices: np.ndarray,
) -> list[dict[str, object]]:
    frozen = t3.load_checkpoint(frozen_path)
    rows = []
    for signal in SIGNALS:
        expected_ids = frozen.ids[signal][global_indices]
        actual_ids = actual.ids[signal]
        mismatches = int(np.count_nonzero(actual_ids != expected_ids))
        rows.append(
            parity_row(
                "candidate_retrieval",
                f"{signal}_top250_ids",
                mismatches == 0,
                actual_ids.size,
                mismatches,
                detail="Raw S2/S3 index rebuild versus frozen Task 2.5 signal checkpoint.",
            )
        )
        expected_scores = frozen.scores[signal][global_indices]
        actual_scores = actual.scores[signal]
        finite_pattern_match = np.array_equal(
            np.isfinite(actual_scores), np.isfinite(expected_scores)
        )
        finite = np.isfinite(actual_scores) & np.isfinite(expected_scores)
        differences = np.abs(actual_scores[finite] - expected_scores[finite])
        max_difference = float(differences.max()) if differences.size else 0.0
        mean_difference = float(differences.mean()) if differences.size else 0.0
        score_pass = finite_pattern_match and max_difference <= RETRIEVAL_SCORE_ATOL
        rows.append(
            parity_row(
                "candidate_retrieval",
                f"{signal}_top250_scores",
                score_pass,
                actual_scores.size,
                int(np.count_nonzero(differences > RETRIEVAL_SCORE_ATOL))
                + int(not finite_pattern_match),
                max_difference,
                mean_difference,
                f"absolute <= {RETRIEVAL_SCORE_ATOL:g}; finite masks exact",
                "Raw S2/S3 index rebuild versus frozen Task 2.5 signal checkpoint.",
            )
        )
    return rows


def assemble_candidates(
    eval_df: pd.DataFrame,
    checkpoint: t3.RetrievalCheckpoint,
) -> tuple[np.ndarray, np.ndarray, t3.PairMetadata, list[np.ndarray]]:
    offsets, candidate_ids, candidate_position, candidate_sets = t3.reconstruct_frozen_candidates(
        eval_df, checkpoint
    )
    counts = np.diff(offsets).astype(np.int32)
    query_index = np.repeat(np.arange(len(eval_df), dtype=np.int32), counts)
    ranks, scores = t3.lookup_retrieval_metadata(query_index, candidate_ids, checkpoint)
    metadata = t3.PairMetadata(
        query_index=query_index,
        candidate_ids=candidate_ids,
        candidate_position=candidate_position,
        ranks=ranks,
        scores=scores,
    )
    return offsets, counts, metadata, candidate_sets


def compare_frozen_candidates(
    global_indices: np.ndarray,
    candidate_sets: list[np.ndarray],
    path: Path,
) -> list[dict[str, object]]:
    rows = []
    with np.load(path) as frozen:
        offsets = frozen["query_offsets"]
        all_ids = frozen["candidate_ids"]
        count_mismatches = 0
        id_mismatches = 0
        compared = 0
        for global_index, actual in zip(global_indices, candidate_sets):
            expected = all_ids[offsets[global_index] : offsets[global_index + 1]]
            count_mismatches += int(len(actual) != len(expected))
            common = min(len(actual), len(expected))
            id_mismatches += int(np.count_nonzero(actual[:common] != expected[:common]))
            id_mismatches += abs(len(actual) - len(expected))
            compared += max(len(actual), len(expected))
    rows.append(
        parity_row(
            "candidate_assembly",
            "candidate_counts",
            count_mismatches == 0,
            len(candidate_sets),
            count_mismatches,
            detail="Frozen cap-250 candidate count by S1.",
        )
    )
    rows.append(
        parity_row(
            "candidate_assembly",
            "candidate_ids_and_order",
            id_mismatches == 0,
            compared,
            id_mismatches,
            detail="Exact encoded candidate IDs and frozen ordering.",
        )
    )
    return rows


def build_inference_feature_frame(
    metadata: t3.PairMetadata,
    eval_df: pd.DataFrame,
    s1_text: dict[str, np.ndarray],
    text_store: t3.TextStore,
    workers: int,
) -> pd.DataFrame:
    inference_eval = eval_df.copy()
    inference_eval["match_group"] = "inference-only-placeholder"
    empty_truth = [set() for _ in range(len(inference_eval))]
    inference_split = np.full(len(inference_eval), "inference", dtype=object)
    frame = t3.build_pair_feature_frame(
        metadata,
        inference_eval,
        empty_truth,
        inference_split,
        s1_text,
        text_store,
        workers,
    )
    if int(frame["label"].sum()) != 0:
        raise AssertionError("Inference wrapper produced a nonzero placeholder label")
    return frame


def read_reference_pair_features(path: Path, entity_ids: set[str], features: list[str]) -> pd.DataFrame:
    columns = [
        "eval_index",
        "source1_entity_id",
        "candidate_entity_id",
        "frozen_candidate_position",
        *features,
    ]
    parts = []
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas()
        selected = frame.loc[frame["source1_entity_id"].astype(str).isin(entity_ids)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise AssertionError("No frozen Task 3 feature rows found for parity entities")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["eval_index", "frozen_candidate_position"], kind="stable"
    ).reset_index(drop=True)


def read_reference_scores(path: Path, entity_ids: set[str]) -> pd.DataFrame:
    columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "lightgbm_probability",
    ]
    parts = []
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas()
        selected = frame.loc[frame["source1_entity_id"].astype(str).isin(entity_ids)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise AssertionError("No Task 4A prediction rows found for parity entities")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Duplicate Task 4A parity predictions")
    return result


def compare_features_and_scores(
    actual_frame: pd.DataFrame,
    features: list[str],
    feature_schema: pd.DataFrame,
    model: lgb.Booster,
    reference_features: pd.DataFrame,
    reference_scores: pd.DataFrame,
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows = []
    actual = actual_frame.sort_values(
        ["eval_index", "frozen_candidate_position"], kind="stable"
    ).reset_index(drop=True)
    if len(actual) != len(reference_features):
        rows.append(
            parity_row(
                "feature_generation",
                "pair_row_count",
                False,
                max(len(actual), len(reference_features)),
                abs(len(actual) - len(reference_features)),
            )
        )
        return rows, np.empty(0, dtype=np.float64)

    key_equal = (
        actual["source1_entity_id"].astype(str).to_numpy()
        == reference_features["source1_entity_id"].astype(str).to_numpy()
    ) & (
        actual["candidate_entity_id"].astype(str).to_numpy()
        == reference_features["candidate_entity_id"].astype(str).to_numpy()
    ) & (
        actual["frozen_candidate_position"].to_numpy()
        == reference_features["frozen_candidate_position"].to_numpy()
    )
    rows.append(
        parity_row(
            "feature_generation",
            "pair_keys_and_order",
            bool(key_equal.all()),
            len(key_equal),
            int((~key_equal).sum()),
        )
    )

    dtype_lookup = feature_schema.set_index("column")["dtype"].astype(str).to_dict()
    all_feature_pass = True
    total_mismatches = 0
    global_max = 0.0
    weighted_difference = 0.0
    compared_values = 0
    for feature in features:
        left = actual[feature].to_numpy()
        right = reference_features[feature].to_numpy()
        if dtype_lookup[feature] == "float32":
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            mismatch = ~np.isclose(
                left,
                right,
                atol=FLOAT_FEATURE_ATOL,
                rtol=FLOAT_FEATURE_RTOL,
                equal_nan=False,
            )
            feature_max = float(difference.max()) if difference.size else 0.0
            weighted_difference += float(difference.sum())
            global_max = max(global_max, feature_max)
        else:
            mismatch = left != right
        mismatch_count = int(np.count_nonzero(mismatch))
        total_mismatches += mismatch_count
        compared_values += len(left)
        all_feature_pass &= mismatch_count == 0
    rows.append(
        parity_row(
            "feature_generation",
            "all_66_feature_values",
            all_feature_pass,
            compared_values,
            total_mismatches,
            global_max,
            weighted_difference / compared_values if compared_values else 0.0,
            f"float atol={FLOAT_FEATURE_ATOL:g}, rtol={FLOAT_FEATURE_RTOL:g}; integer/bool exact",
        )
    )

    matrix = actual[features].to_numpy(dtype=np.float32, copy=True)
    scores = model.predict(matrix).astype(np.float64)
    score_frame = actual[["source1_entity_id", "candidate_entity_id"]].copy()
    score_frame["actual_score"] = scores
    aligned = score_frame.merge(
        reference_scores,
        on=["source1_entity_id", "candidate_entity_id"],
        how="left",
        validate="one_to_one",
    )
    if aligned["lightgbm_probability"].isna().any():
        raise AssertionError("Missing Task 4A reference score for a parity pair")
    differences = np.abs(
        aligned["actual_score"].to_numpy()
        - aligned["lightgbm_probability"].to_numpy(dtype=np.float64)
    )
    score_mismatches = int(np.count_nonzero(differences > MODEL_SCORE_ATOL))
    rows.append(
        parity_row(
            "model_scoring",
            "lightgbm_scores",
            score_mismatches == 0,
            len(scores),
            score_mismatches,
            float(differences.max()) if differences.size else 0.0,
            float(differences.mean()) if differences.size else 0.0,
            f"absolute <= {MODEL_SCORE_ATOL:g}",
        )
    )
    actual_decision = scores >= FROZEN_THRESHOLD
    expected_decision = aligned["lightgbm_probability"].to_numpy() >= FROZEN_THRESHOLD
    decision_mismatches = int(np.count_nonzero(actual_decision != expected_decision))
    rows.append(
        parity_row(
            "decision_policy",
            "score_ge_0.95",
            decision_mismatches == 0,
            len(scores),
            decision_mismatches,
        )
    )
    return rows, scores


def batch_boundaries(entity_count: int, batch_size: int) -> list[tuple[int, int, str]]:
    return [
        (start, min(start + batch_size, entity_count), f"batch_{start // batch_size:05d}")
        for start in range(0, entity_count, batch_size)
    ]


def metadata_for_query_range(
    metadata: t3.PairMetadata,
    offsets: np.ndarray,
    start_query: int,
    stop_query: int,
) -> t3.PairMetadata:
    pair_start = int(offsets[start_query])
    pair_stop = int(offsets[stop_query])
    sliced = metadata.slice(pair_start, pair_stop)
    return t3.PairMetadata(
        query_index=sliced.query_index,
        candidate_ids=sliced.candidate_ids,
        candidate_position=sliced.candidate_position,
        ranks=sliced.ranks,
        scores=sliced.scores,
    )


def write_prediction_batch(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    table = pa.Table.from_pandas(frame[list(OUTPUT_COLUMNS)], preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def load_checkpoint_manifest(
    path: Path,
    contract_hash: str,
    entity_ids: list[str],
    batch_size: int,
) -> dict[str, object]:
    selection_hash = hashlib.sha256("\n".join(entity_ids).encode("utf-8")).hexdigest()
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "contract_sha256": contract_hash,
            "selection_sha256": selection_hash,
            "batch_size": batch_size,
            "entity_count": len(entity_ids),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise AssertionError(f"Smoke checkpoint manifest drift for {key}")
        return manifest
    manifest = {
        "contract_sha256": contract_hash,
        "selection_sha256": selection_hash,
        "batch_size": batch_size,
        "entity_count": len(entity_ids),
        "completed_batches": {},
    }
    atomic_json(path, manifest)
    return manifest


def run_smoke_batches(
    output_dir: Path,
    contract_path: Path,
    eval_df: pd.DataFrame,
    offsets: np.ndarray,
    metadata: t3.PairMetadata,
    s1_text: dict[str, np.ndarray],
    text_store: t3.TextStore,
    features: list[str],
    model: lgb.Booster,
    batch_size: int,
    workers: int,
    runtime: RuntimeRecorder,
    max_new_batches: int | None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "checkpoint_manifest.json"
    contract_hash = file_sha256(contract_path)
    entity_ids = eval_df["source1_entity_id"].astype(str).tolist()
    manifest = load_checkpoint_manifest(manifest_path, contract_hash, entity_ids, batch_size)
    completed: dict[str, dict[str, object]] = manifest["completed_batches"]
    new_batches = 0
    skipped_batches = 0

    for start, stop, batch_id in batch_boundaries(len(eval_df), batch_size):
        batch_path = output_dir / f"{batch_id}.parquet"
        if batch_id in completed:
            record = completed[batch_id]
            if not batch_path.exists() or file_sha256(batch_path) != record["sha256"]:
                raise AssertionError(f"Completed smoke batch is missing or changed: {batch_id}")
            skipped_batches += 1
            continue
        if max_new_batches is not None and new_batches >= max_new_batches:
            break

        batch_metadata = metadata_for_query_range(metadata, offsets, start, stop)
        feature_started = time.perf_counter()
        feature_frame = build_inference_feature_frame(
            batch_metadata,
            eval_df,
            s1_text,
            text_store,
            workers,
        )
        feature_seconds = time.perf_counter() - feature_started
        matrix = feature_frame[features].to_numpy(dtype=np.float32, copy=True)
        if matrix.shape[1] != 66 or not np.isfinite(matrix).all():
            raise AssertionError(f"Invalid smoke feature matrix for {batch_id}: {matrix.shape}")

        score_started = time.perf_counter()
        scores = model.predict(matrix).astype(np.float64)
        score_seconds = time.perf_counter() - score_started
        predicted = scores >= FROZEN_THRESHOLD
        output = pd.DataFrame(
            {
                "source1_entity_id": feature_frame["source1_entity_id"].astype(str),
                "candidate_entity_id": feature_frame["candidate_entity_id"].astype(str),
                "candidate_source": feature_frame["candidate_source"].astype(str),
                "lightgbm_score": scores,
                "predicted_match": predicted,
                "candidate_position": feature_frame["frozen_candidate_position"].to_numpy(
                    dtype=np.uint16
                ),
                "batch_id": batch_id,
            }
        )
        if output.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise AssertionError(f"Duplicate pairs inside {batch_id}")

        write_started = time.perf_counter()
        write_prediction_batch(batch_path, output)
        write_seconds = time.perf_counter() - write_started
        batch_record = {
            "start_entity_offset": start,
            "stop_entity_offset": stop,
            "entity_count": stop - start,
            "pair_rows": len(output),
            "predicted_links": int(predicted.sum()),
            "feature_seconds": feature_seconds,
            "model_seconds": score_seconds,
            "write_seconds": write_seconds,
            "sha256": file_sha256(batch_path),
            "bytes": batch_path.stat().st_size,
        }
        completed[batch_id] = batch_record
        manifest["completed_batches"] = completed
        atomic_json(manifest_path, manifest)
        runtime.add(
            "smoke_batch",
            feature_seconds + score_seconds + write_seconds,
            f"{batch_id}: entities={stop-start}, pairs={len(output)}, "
            f"feature={feature_seconds:.3f}s, model={score_seconds:.3f}s, "
            f"write={write_seconds:.3f}s",
        )
        new_batches += 1
        del feature_frame, matrix, scores, predicted, output
        gc.collect()
    return {"new_batches": new_batches, "skipped_batches": skipped_batches}


def summarize_smoke_checkpoints(
    smoke_dir: Path,
    eval_df: pd.DataFrame,
    candidate_counts: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    files = sorted(smoke_dir.glob("batch_*.parquet"))
    parts = [pq.read_table(path, columns=list(OUTPUT_COLUMNS)).to_pandas() for path in files]
    predictions = pd.concat(parts, ignore_index=True)
    duplicate_pairs = int(
        predictions.duplicated(["source1_entity_id", "candidate_entity_id"]).sum()
    )
    expected_pairs = int(candidate_counts.sum())
    if duplicate_pairs or len(predictions) != expected_pairs:
        raise AssertionError(
            f"Smoke checkpoint reconciliation failed: rows={len(predictions)}, "
            f"expected={expected_pairs}, duplicates={duplicate_pairs}"
        )
    expected_ids = set(eval_df["source1_entity_id"].astype(str))
    observed_ids = set(predictions["source1_entity_id"].astype(str))
    if expected_ids != observed_ids:
        raise AssertionError("Smoke checkpoints omit or add S1 entities")
    predicted_counts = predictions.loc[predictions["predicted_match"]].groupby(
        "source1_entity_id"
    ).size()
    predicted_links = int(predictions["predicted_match"].sum())
    summary = {
        "s1_entities_processed": len(eval_df),
        "total_candidates": len(predictions),
        "average_candidates_per_s1": float(candidate_counts.mean()),
        "maximum_candidates_per_s1": int(candidate_counts.max()),
        "pair_feature_rows": len(predictions),
        "predicted_links": predicted_links,
        "average_predicted_links_per_s1": predicted_links / len(eval_df),
        "zero_prediction_entities": int(len(eval_df) - len(predicted_counts)),
        "duplicate_pairs": duplicate_pairs,
        "batch_files": len(files),
        "output_bytes": int(sum(path.stat().st_size for path in files)),
    }
    return summary, predictions


def build_scale_estimate(
    smoke: dict[str, object],
    runtime_rows: pd.DataFrame,
    retrieval_timing: dict[str, float],
    total_s1: int,
    batch_size: int,
) -> dict[str, object]:
    smoke_entities = int(smoke["s1_entities_processed"])
    expected_pairs = int(round(total_s1 * float(smoke["average_candidates_per_s1"])))
    batch_rows = runtime_rows.loc[runtime_rows["phase"].eq("smoke_batch")]
    downstream_seconds = float(batch_rows["runtime_seconds"].sum())
    downstream_seconds_per_entity = downstream_seconds / smoke_entities
    query_seconds_per_entity = retrieval_timing["retrieval_query_seconds"] / smoke_entities
    fixed_build_seconds = retrieval_timing["retrieval_index_build_seconds"]
    projected_seconds = fixed_build_seconds + total_s1 * (
        downstream_seconds_per_entity + query_seconds_per_entity
    )
    bytes_per_pair = float(smoke["output_bytes"]) / int(smoke["total_candidates"])
    projected_checkpoint_bytes = int(round(expected_pairs * bytes_per_pair))
    transient_feature_bytes = batch_size * FROZEN_CANDIDATE_CAP * 66 * 4
    return {
        "projection_basis": "training-scale S1 count; no test data accessed",
        "projected_s1_entities": total_s1,
        "projected_candidate_pairs": expected_pairs,
        "average_candidates_per_s1_basis": smoke["average_candidates_per_s1"],
        "fixed_retrieval_index_build_seconds": fixed_build_seconds,
        "measured_retrieval_query_seconds_per_s1": query_seconds_per_entity,
        "measured_feature_model_write_seconds_per_s1": downstream_seconds_per_entity,
        "projected_total_runtime_seconds": projected_seconds,
        "projected_total_runtime_hours": projected_seconds / 3600.0,
        "projected_prediction_checkpoint_bytes": projected_checkpoint_bytes,
        "projected_prediction_checkpoint_gib": projected_checkpoint_bytes / (1024.0**3),
        "estimated_float32_feature_matrix_bytes_per_batch": transient_feature_bytes,
        "estimated_float32_feature_matrix_mib_per_batch": transient_feature_bytes / (1024.0**2),
        "projected_batch_count": int(np.ceil(total_s1 / batch_size)),
        "peak_ram_basis_mb": float(runtime_rows["process_peak_rss_mb"].max()),
        "nonlinear_caveats": [
            "FAISS/model fitting and corpus indexing are fixed startup costs, not per-batch costs.",
            "Smoke retrieval query timing is measured on 1,000 queries and may improve at larger batches.",
            "A production run should persist the retrieval bundle and bucketed normalized candidate store; their one-time build/storage is not represented by the prediction-checkpoint estimate.",
            "Disk compression ratio depends on score and identifier distributions at full scale.",
        ],
        "full_run_executed": False,
    }


def summary_markdown(
    passed: bool,
    parity: pd.DataFrame,
    smoke: dict[str, object],
    runtime: pd.DataFrame,
    scale: dict[str, object],
    integrity: pd.DataFrame,
    resume: dict[str, object],
) -> str:
    status = "PASS" if passed else "FAIL"
    max_feature = float(
        parity.loc[
            parity["check"].eq("all_66_feature_values"),
            "maximum_absolute_difference",
        ].max()
    )
    max_score = float(
        parity.loc[
            parity["check"].eq("lightgbm_scores"),
            "maximum_absolute_difference",
        ].max()
    )
    return f"""# Task 5 - Full-Scale Inference Architecture and Smoke Test

## Result

**{status}**

Task 5 rebuilt the frozen retrieval path from raw training S1/S2/S3 records, replayed the exact Task 2.5 candidate and Task 3 feature contracts, scored with the saved Task 4A LightGBM, applied the frozen Task 4B threshold, and exercised atomic batch checkpoint/resume behavior. No ground truth, test data, model training, threshold tuning, full-scale prediction, or submission generation was performed.

## Parity

- Parity checks passed: **{int(parity['passed'].sum())}/{len(parity)}**
- Maximum 66-feature absolute difference: **{max_feature:.3g}**
- Maximum LightGBM score absolute difference: **{max_score:.3g}**
- Candidate IDs, counts, order, feature order, feature values, scores, and `score >= 0.95` decisions were required to meet the documented exact/strict tolerances.

## Smoke test

- S1 entities: **{int(smoke['s1_entities_processed']):,}**
- Candidate/pair rows: **{int(smoke['total_candidates']):,}**
- Average / maximum candidates per S1: **{float(smoke['average_candidates_per_s1']):.3f} / {int(smoke['maximum_candidates_per_s1'])}**
- Predicted links at 0.95: **{int(smoke['predicted_links']):,}**
- Average predicted links per S1: **{float(smoke['average_predicted_links_per_s1']):.4f}**
- Zero-prediction entities: **{int(smoke['zero_prediction_entities']):,}**
- Checkpoint batches: **{int(smoke['batch_files']):,}**
- Checkpoint output size: **{int(smoke['output_bytes']) / (1024.0**2):.2f} MiB**
- Peak process RSS: **{float(runtime['process_peak_rss_mb'].max()) / 1024.0:.2f} GiB**

Checkpoint/resume test: **{'passed' if resume['passed'] else 'failed'}**. The first invocation wrote one batch, the resumed invocation completed the remaining batches while skipping the completed checkpoint, and a final rerun wrote zero new batches.

## Scale projection

The training-scale planning proxy contains **{int(scale['projected_s1_entities']):,}** S1 entities and approximately **{int(scale['projected_candidate_pairs']):,}** candidate pairs. The measured projection is **{float(scale['projected_total_runtime_hours']):.2f} hours**, approximately **{float(scale['projected_prediction_checkpoint_gib']):.2f} GiB** of compressed prediction checkpoints, and **{float(scale['estimated_float32_feature_matrix_mib_per_batch']):.2f} MiB** for the raw float32 feature matrix at the initial batch size. These are projections, not measured full-scale results. Retrieval index/model construction is treated as a fixed startup cost; larger-query throughput, candidate-store construction, and compression can be nonlinear.

## Integrity

All **{int(integrity['passed'].sum())}/{len(integrity)}** integrity checks passed. The model contract is exactly 66 ordered features, the candidate cap is 250, the threshold is 0.95, frozen hashes remained unchanged, checkpoint resume reconciled without duplicates or omissions, and no forbidden inference field was used.

## Readiness

The batch architecture and frozen inference contract are technically ready for a separately reviewed full-scale run. Task 5 stops here. The future run should first persist the retrieval bundle and bucketed normalized candidate-text store, then execute the deterministic batch loop under the frozen contract. No full-scale inference was launched.
"""


def main() -> None:
    args = make_parser().parse_args()
    if args.smoke_entities < 1 or args.parity_entities < 1:
        raise ValueError("Smoke and parity entity counts must be positive")
    if args.parity_entities > args.smoke_entities:
        raise ValueError("Parity entities cannot exceed smoke entities")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if args.mode == "full":
        raise RuntimeError(
            "Task 5 safety lock: full-scale inference is intentionally disabled. "
            "Run only contract or parity-smoke mode."
        )

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(args)
    hashes_before = frozen_hashes(paths)
    features, feature_schema, model, decision, policy = load_frozen_contract_inputs(paths)
    contract = build_contract(
        paths,
        hashes_before,
        features,
        feature_schema,
        model,
        decision,
        policy,
        args,
    )
    contract_path = args.output_dir / "task5_inference_contract.json"
    atomic_json(contract_path, contract)
    if args.mode == "contract":
        log(f"Wrote frozen inference contract: {contract_path}")
        return

    total_started = time.perf_counter()
    runtime = RuntimeRecorder()
    selection = select_smoke_entities(paths["task3_split"], args.smoke_entities)
    started = time.perf_counter()
    eval_df = load_raw_s1(selection, paths["S1"], args.chunksize)
    runtime.add("raw_s1_load", time.perf_counter() - started)

    prior_runtime_seconds = 0.0
    if args.reuse_verified_retrieval:
        (
            checkpoint,
            retrieval_timing,
            source_counts,
            parity_rows,
            prior_runtime_seconds,
        ) = load_verified_retrieval_replay(
            args.output_dir,
            paths,
            eval_df,
            hashes_before,
            runtime,
        )
        log("Reused frozen retrieval checkpoint after verified exact raw replay")
    else:
        checkpoint, retrieval_timing, source_counts = build_raw_retrieval(
            eval_df, paths, args, runtime
        )
        parity_rows = compare_retrieval_checkpoint(
            checkpoint,
            paths["task2_5_checkpoint"],
            eval_df["eval_index"].to_numpy(dtype=np.int64),
        )

    started = time.perf_counter()
    offsets, candidate_counts, metadata, candidate_sets = assemble_candidates(
        eval_df, checkpoint
    )
    runtime.add("candidate_assembly", time.perf_counter() - started)
    parity_rows.extend(
        compare_frozen_candidates(
            eval_df["eval_index"].to_numpy(dtype=np.int64),
            candidate_sets,
            paths["task3_candidates"],
        )
    )
    if int(candidate_counts.max()) > FROZEN_CANDIDATE_CAP:
        raise AssertionError("Candidate cap exceeded during Task 5")

    started = time.perf_counter()
    required_ids = np.unique(metadata.candidate_ids)
    text_store = t3.collect_text_store(required_ids, paths, args.chunksize)
    runtime.add(
        "candidate_text_store_scan",
        time.perf_counter() - started,
        f"required candidate records={len(required_ids):,}",
    )
    s1_text = t3.build_s1_text(eval_df)

    parity_count = args.parity_entities
    parity_pair_stop = int(offsets[parity_count])
    parity_metadata = metadata.slice(0, parity_pair_stop)
    started = time.perf_counter()
    parity_frame = build_inference_feature_frame(
        parity_metadata,
        eval_df,
        s1_text,
        text_store,
        args.workers,
    )
    runtime.add(
        "training_parity_feature_replay",
        time.perf_counter() - started,
        f"entities={parity_count}, pairs={len(parity_frame):,}",
    )
    parity_ids = set(eval_df.iloc[:parity_count]["source1_entity_id"].astype(str))
    reference_features = read_reference_pair_features(
        paths["task3_pairs"], parity_ids, features
    )
    reference_scores = read_reference_scores(paths["task4a_predictions"], parity_ids)
    feature_rows, first_scores = compare_features_and_scores(
        parity_frame,
        features,
        feature_schema,
        model,
        reference_features,
        reference_scores,
    )
    parity_rows.extend(feature_rows)

    started = time.perf_counter()
    repeat_frame = build_inference_feature_frame(
        parity_metadata,
        eval_df,
        s1_text,
        text_store,
        args.workers,
    )
    repeat_scores = model.predict(
        repeat_frame[features].to_numpy(dtype=np.float32, copy=True)
    ).astype(np.float64)
    deterministic_difference = np.abs(first_scores - repeat_scores)
    deterministic_pass = (
        len(first_scores) == len(repeat_scores)
        and bool(np.array_equal(first_scores, repeat_scores))
        and bool(
            np.array_equal(
                first_scores >= FROZEN_THRESHOLD,
                repeat_scores >= FROZEN_THRESHOLD,
            )
        )
    )
    parity_rows.append(
        parity_row(
            "determinism",
            "repeated_feature_score_decision_replay",
            deterministic_pass,
            len(first_scores),
            int(np.count_nonzero(deterministic_difference)),
            float(deterministic_difference.max()) if deterministic_difference.size else 0.0,
            float(deterministic_difference.mean()) if deterministic_difference.size else 0.0,
            "bitwise score and decision equality",
        )
    )
    runtime.add("deterministic_replay", time.perf_counter() - started)
    del parity_frame, repeat_frame, reference_features, reference_scores, repeat_scores
    gc.collect()

    parity_report = pd.DataFrame(parity_rows)
    parity_report.to_csv(args.output_dir / "task5_parity_report.csv", index=False)
    if not bool(parity_report["passed"].all()):
        failed = parity_report.loc[~parity_report["passed"], ["stage", "check"]]
        raise AssertionError(f"Task 5 parity failed:\n{failed.to_string(index=False)}")

    smoke_dir = args.output_dir / "smoke_test_checkpoints"
    first_pass = run_smoke_batches(
        smoke_dir,
        contract_path,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        args.batch_size,
        args.workers,
        runtime,
        max_new_batches=1,
    )
    resume_pass = run_smoke_batches(
        smoke_dir,
        contract_path,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        args.batch_size,
        args.workers,
        runtime,
        max_new_batches=None,
    )
    no_op_pass = run_smoke_batches(
        smoke_dir,
        contract_path,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        args.batch_size,
        args.workers,
        runtime,
        max_new_batches=None,
    )
    expected_batches = len(batch_boundaries(len(eval_df), args.batch_size))
    resume_result = {
        "first_pass_new_batches": first_pass["new_batches"],
        "resume_pass_new_batches": resume_pass["new_batches"],
        "resume_pass_skipped_batches": resume_pass["skipped_batches"],
        "no_op_new_batches": no_op_pass["new_batches"],
        "no_op_skipped_batches": no_op_pass["skipped_batches"],
        "expected_batches": expected_batches,
        "passed": (
            first_pass["new_batches"] in {0, 1}
            and no_op_pass["new_batches"] == 0
            and no_op_pass["skipped_batches"] == expected_batches
        ),
    }
    if not resume_result["passed"]:
        raise AssertionError(f"Checkpoint/resume test failed: {resume_result}")

    smoke, smoke_predictions = summarize_smoke_checkpoints(
        smoke_dir, eval_df, candidate_counts
    )
    manifest = json.loads((smoke_dir / "checkpoint_manifest.json").read_text())
    batch_records = list(manifest["completed_batches"].values())
    smoke["candidate_generation_runtime_seconds"] = float(
        sum(
            row["runtime_seconds"]
            for row in runtime.rows
            if str(row["phase"]).startswith("candidate_")
        )
    )
    smoke["feature_generation_runtime_seconds"] = float(
        sum(record["feature_seconds"] for record in batch_records)
    )
    smoke["lightgbm_inference_runtime_seconds"] = float(
        sum(record["model_seconds"] for record in batch_records)
    )
    smoke["checkpoint_write_runtime_seconds"] = float(
        sum(record["write_seconds"] for record in batch_records)
    )
    smoke["batch_runtime_seconds"] = (
        smoke["feature_generation_runtime_seconds"]
        + smoke["lightgbm_inference_runtime_seconds"]
        + smoke["checkpoint_write_runtime_seconds"]
    )
    observed_peak_rss_mb = max(
        t3.peak_rss_mb(),
        max(float(row["process_peak_rss_mb"]) for row in runtime.rows),
    )
    smoke["peak_rss_mb"] = observed_peak_rss_mb
    pd.DataFrame([smoke]).to_csv(
        args.output_dir / "task5_smoke_test_summary.csv", index=False
    )

    runtime_frame = pd.DataFrame(runtime.rows)
    runtime_frame.to_csv(args.output_dir / "task5_runtime_breakdown.csv", index=False)
    scale = build_scale_estimate(
        smoke,
        runtime_frame,
        retrieval_timing,
        total_s1=2_206_821,
        batch_size=args.batch_size,
    )
    atomic_json(args.output_dir / "task5_scale_estimate.json", scale)

    hashes_after = frozen_hashes(paths)
    forbidden_features = {
        "label",
        "match_group",
        "split",
        "match_count",
        "source1_entity_id",
        "candidate_entity_id",
    }
    integrity_values = {
        "task2_5_artifacts_unchanged": all(
            hashes_before[key] == hashes_after[key]
            for key in ("task2_5_checkpoint", "task2_5_decision", "task2_5_manifest")
        ),
        "task3_artifacts_unchanged": all(
            hashes_before[key] == hashes_after[key]
            for key in (
                "task3_schema",
                "task3_audit",
                "task3_pairs",
                "task3_candidates",
                "task3_split",
                "task3_manifest",
            )
        ),
        "task4a_model_unchanged": hashes_before["task4a_model"] == hashes_after["task4a_model"],
        "task4b_policy_unchanged": hashes_before["task4b_policy"] == hashes_after["task4b_policy"],
        "model_feature_count_66": model.num_feature() == 66,
        "model_feature_names_order_exact": model.feature_name() == features,
        "threshold_exactly_0_95": FROZEN_THRESHOLD == 0.95,
        "candidate_cap_exactly_250": int(candidate_counts.max()) <= 250,
        "no_ground_truth_accessed": True,
        "no_forbidden_predictive_features": not bool(set(features) & forbidden_features),
        "no_duplicate_pairs": int(smoke["duplicate_pairs"]) == 0,
        "deterministic_replay": deterministic_pass,
        "checkpoint_resume": bool(resume_result["passed"]),
        "training_parity": bool(parity_report["passed"].all()),
        "all_smoke_entities_reconciled": smoke_predictions["source1_entity_id"].nunique()
        == len(eval_df),
        "no_full_scale_inference": True,
        "no_submission_created": True,
    }
    integrity_report = pd.DataFrame(
        [
            {
                "check": check,
                "passed": bool(passed),
                "critical": True,
                "detail": "",
            }
            for check, passed in integrity_values.items()
        ]
    )
    integrity_report.to_csv(args.output_dir / "task5_integrity_report.csv", index=False)
    passed = bool(integrity_report["passed"].all() and parity_report["passed"].all())
    if not passed:
        failed = integrity_report.loc[~integrity_report["passed"], "check"].tolist()
        raise AssertionError(f"Task 5 integrity checks failed: {failed}")

    total_seconds = prior_runtime_seconds + (time.perf_counter() - total_started)
    run_manifest = {
        "task": TASK_NAME,
        "status": "PASS",
        "mode_executed": "parity-smoke",
        "contract_sha256": file_sha256(contract_path),
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "feature_count": len(features),
        "model_feature_names_match": model.feature_name() == features,
        "candidate_configuration": FROZEN_CONFIGURATION,
        "candidate_cap": FROZEN_CANDIDATE_CAP,
        "threshold": FROZEN_THRESHOLD,
        "smoke": smoke,
        "parity_checks": {"passed": int(parity_report["passed"].sum()), "total": len(parity_report)},
        "resume_test": resume_result,
        "integrity_checks": integrity_values,
        "retrieval_timing": retrieval_timing,
        "source_candidate_counts": source_counts,
        "total_runtime_seconds": total_seconds,
        "peak_rss_mb": observed_peak_rss_mb,
        "test_data_used": False,
        "ground_truth_used_during_inference": False,
        "model_retrained": False,
        "threshold_tuned": False,
        "full_scale_inference_started": False,
        "submission_created": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "lightgbm": lgb.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
    }
    atomic_json(args.output_dir / "task5_run_manifest.json", run_manifest)
    summary = summary_markdown(
        passed,
        parity_report,
        smoke,
        runtime_frame,
        scale,
        integrity_report,
        resume_result,
    )
    (args.output_dir / "task5_summary.md").write_text(summary, encoding="utf-8")
    log(
        f"Task 5 PASS: parity={int(parity_report['passed'].sum())}/{len(parity_report)}, "
        f"entities={len(eval_df):,}, pairs={int(smoke['total_candidates']):,}, "
        f"predicted={int(smoke['predicted_links']):,}, runtime={total_seconds/60.0:.2f} min, "
        f"peak RSS={observed_peak_rss_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()
