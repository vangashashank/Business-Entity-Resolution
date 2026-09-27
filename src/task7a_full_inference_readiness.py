#!/usr/bin/env python3
"""Task 7A: full-scale inference readiness audit on a 100-S1 training smoke.

This module audits the frozen production path and supplies reusable input, disk,
and checkpoint preflight contracts. Its full-run mode is intentionally locked.
It never reads ground truth or test data and does not change predictions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import task2_candidate_generation as t2
import task3_pairwise_features as t3
import task5_full_inference_pipeline as t5
import task6_persist_production_retrieval as t6


TASK_NAME = "Task 7A"
SMOKE_ENTITIES = 100
SMOKE_BATCH_SIZE = 100
FAILURE_BATCH_SIZE = 50
SAFETY_MARGIN_FRACTION = 0.25
OUTPUT_COLUMNS = tuple(t5.OUTPUT_COLUMNS)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_values_sha256(values: Iterable[object]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


class RuntimeRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record(self, phase: str, seconds: float, scaling_basis: str, detail: str = "") -> None:
        self.rows.append(
            {
                "phase": phase,
                "runtime_seconds": float(seconds),
                "scaling_basis": scaling_basis,
                "current_rss_mb": t3.current_rss_mb(),
                "process_peak_rss_mb": t3.peak_rss_mb(),
                "detail": detail,
            }
        )
        log(f"{phase}: {seconds:.3f}s")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("audit", "disk-preflight", "input-preflight", "full"),
        default="audit",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource" / "dataset" / "train",
    )
    parser.add_argument(
        "--task6-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task6_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task7a_outputs",
    )
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use one native worker for the readiness smoke to avoid mixed OpenMP instability.",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = t5.resolve_paths(args)
    paths.update(
        {
            "task5_contract": PROJECT_ROOT / "outputs/task5_outputs/task5_inference_contract.json",
            "task6_summary": args.task6_output_dir / "task6_summary.md",
            "task6_run_manifest": args.task6_output_dir / "task6_run_manifest.json",
            "task6_retrieval_manifest": args.task6_output_dir / "task6_retrieval_manifest.json",
            "task6_store_manifest": args.task6_output_dir / "task6_candidate_store_manifest.json",
            "task6_runtime": args.task6_output_dir / "task6_runtime_breakdown.csv",
            "task6_scale": args.task6_output_dir / "task6_scale_estimate.json",
            "task6_integrity": args.task6_output_dir / "task6_integrity_report.csv",
            "task6_smoke_batch": args.task6_output_dir
            / "production_smoke_checkpoints/batch_00000.parquet",
            "task5_code": PROJECT_ROOT / "src/task5_full_inference_pipeline.py",
            "task6_code": PROJECT_ROOT / "src/task6_persist_production_retrieval.py",
            "task7a_code": Path(__file__).resolve(),
            "project_status": PROJECT_ROOT / "PROJECT_STATUS_SUMMARY.md",
        }
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 7A input artifacts: {missing}")
    for key in ("S1", "S2", "S3"):
        if "test" in paths[key].name.casefold():
            raise ValueError(f"Task 7A cannot access test data: {paths[key]}")
    if "GT" in paths:
        raise AssertionError("Ground-truth path entered the Task 7A runtime")
    return paths


def frozen_hashes(paths: dict[str, Path]) -> dict[str, str]:
    keys = (
        "S1",
        "S2",
        "S3",
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
        "task5_contract",
        "task5_code",
        "task6_code",
        "task6_summary",
        "task6_run_manifest",
        "task6_retrieval_manifest",
        "task6_store_manifest",
        "task6_runtime",
        "task6_scale",
        "task6_integrity",
        "task7a_code",
    )
    return {key: file_sha256(paths[key]) for key in keys}


def build_input_contract(paths: dict[str, Path], retrieval_manifest: dict[str, object]) -> dict[str, object]:
    return {
        "contract_version": "task7a-input-v1",
        "format": "tab-separated text read with dtype=string",
        "required_columns_in_order": [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        "columns": {
            "entity_id": {
                "dtype": "string",
                "nullable": False,
                "unique": True,
                "pattern": r"^S1-\d+$",
            },
            "business_name": {
                "dtype": "string",
                "nullable": True,
                "null_normalization": "empty string",
            },
            "business_address": {
                "dtype": "string",
                "nullable": True,
                "null_normalization": "empty string",
            },
            "country": {
                "dtype": "string",
                "nullable": False,
                "normalized_allowed_values": retrieval_manifest["partitions"],
                "normalizer": "task2_candidate_generation.normalize_country_series",
            },
        },
        "row_requirements": [
            "At least one of normalized business_name or normalized business_address is nonempty.",
            "Every normalized country maps to an existing retrieval partition.",
        ],
        "normalization": {
            "name_and_address": "task2_candidate_generation.normalize_series",
            "country": "task2_candidate_generation.normalize_country_series",
            "transliterated_name": "task2_5_candidate_selection.transliterate_series",
            "suffix_name": "task2_5_candidate_selection.strip_legal_suffix_series",
            "number_address": "task2_5_candidate_selection.address_number_series",
            "code_hashes": {
                "task2": file_sha256(paths["task2_code"]),
                "task2_5": file_sha256(paths["task2_5_code"]),
            },
        },
        "ordering": {
            "rule": "Preserve physical input row order; do not sort before batch assignment.",
            "batch_rule": "Fixed contiguous zero-based row offsets, batch size 100 initially.",
            "identity_hash": "SHA-256 of newline-joined entity_id values in physical order.",
        },
        "fingerprint": {
            "required_before_launch": ["source_file_sha256", "source_file_bytes", "ordered_entity_id_sha256", "row_count"],
            "resume_rule": "All four values must exactly match the checkpoint run contract.",
        },
        "inference_safety": {
            "unknown_country": "fail before retrieval",
            "duplicate_or_null_entity_id": "fail before retrieval",
            "both_retrieval_text_fields_empty": "fail before retrieval",
            "extra_columns": "allowed but ignored only after required columns validate",
        },
    }


def validate_input_frame(frame: pd.DataFrame, contract: dict[str, object]) -> dict[str, object]:
    required = contract["required_columns_in_order"]
    missing_columns = [column for column in required if column not in frame.columns]
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})

    add("required_columns", not missing_columns, f"missing={missing_columns}")
    if missing_columns:
        return {"passed": False, "row_count": len(frame), "checks": checks}
    ids = frame["entity_id"].astype("string")
    add("entity_id_non_null", not ids.isna().any(), f"nulls={int(ids.isna().sum())}")
    add("entity_id_unique", not ids.duplicated().any(), f"duplicates={int(ids.duplicated().sum())}")
    pattern_ok = ids.fillna("").str.fullmatch(r"S1-\d+").all()
    add("entity_id_pattern", bool(pattern_ok), "expected S1-<nonnegative integer>")
    country = t2.normalize_country_series(frame["country"])
    allowed = set(contract["columns"]["country"]["normalized_allowed_values"])
    unknown = sorted(set(country.astype(str)) - allowed)
    add("country_non_null", not frame["country"].isna().any() and bool(country.ne("").all()), f"missing={int(country.eq('').sum())}")
    add("country_partition", not unknown, f"unknown={unknown[:10]}")
    name = t2.normalize_series(frame["business_name"])
    address = t2.normalize_series(frame["business_address"])
    unusable = name.eq("") & address.eq("")
    add("retrieval_text_available", not unusable.any(), f"both_empty={int(unusable.sum())}")
    add("row_count_positive", len(frame) > 0, f"rows={len(frame)}")
    return {
        "passed": all(row["passed"] for row in checks),
        "row_count": len(frame),
        "ordered_entity_id_sha256": ordered_values_sha256(ids.fillna("")),
        "normalized_country_counts": country.value_counts().sort_index().to_dict(),
        "checks": checks,
    }


def negative_input_tests(frame: pd.DataFrame, contract: dict[str, object]) -> list[dict[str, object]]:
    tests = []
    cases = {}
    missing = frame.drop(columns=["country"])
    cases["missing_required_column"] = missing
    duplicate = frame.copy()
    duplicate.loc[duplicate.index[1], "entity_id"] = duplicate.iloc[0]["entity_id"]
    cases["duplicate_entity_id"] = duplicate
    unknown = frame.copy()
    unknown.loc[unknown.index[0], "country"] = "unknown-country"
    cases["unknown_country"] = unknown
    unusable = frame.copy()
    unusable.loc[unusable.index[0], ["business_name", "business_address"]] = pd.NA
    cases["both_text_fields_missing"] = unusable
    for name, case in cases.items():
        result = validate_input_frame(case, contract)
        tests.append(
            {
                "case": name,
                "expected": "FAIL",
                "observed": "PASS" if result["passed"] else "FAIL",
                "passed": not result["passed"],
            }
        )
    return tests


def validate_input_path(path: Path, contract: dict[str, object], chunksize: int) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    source_hash = file_sha256(path)
    ids_seen: set[str] = set()
    ordered_digest = hashlib.sha256()
    checks = []
    row_count = 0
    for chunk in pd.read_csv(path, sep="\t", dtype="string", chunksize=chunksize):
        result = validate_input_frame(chunk, contract)
        if not result["passed"]:
            raise AssertionError(f"Input contract failed near row {row_count}: {result['checks']}")
        ids = chunk["entity_id"].astype(str).tolist()
        overlap = ids_seen.intersection(ids)
        if overlap:
            raise AssertionError(f"Duplicate entity IDs across chunks: {sorted(overlap)[:10]}")
        ids_seen.update(ids)
        for entity_id in ids:
            ordered_digest.update(entity_id.encode("utf-8") + b"\n")
        row_count += len(chunk)
        checks.extend(result["checks"])
    return {
        "passed": True,
        "path": str(path),
        "source_file_sha256": source_hash,
        "source_file_bytes": path.stat().st_size,
        "ordered_entity_id_sha256": ordered_digest.hexdigest(),
        "row_count": row_count,
        "chunk_checks": len(checks),
    }


def build_storage_plan(scale: dict[str, object], output_dir: Path) -> dict[str, object]:
    bundle = int(scale["persistent_bundle_bytes"])
    store = int(scale["persistent_candidate_store_bytes"])
    checkpoint = int(scale["projected_checkpoint_bytes"])
    combined = bundle + store + checkpoint
    margin = int(np.ceil(combined * SAFETY_MARGIN_FRACTION))
    free_required = checkpoint + margin
    deployment_capacity = combined + margin
    disk = shutil.disk_usage(output_dir.parent)
    return {
        "persistent_retrieval_bundle_bytes": bundle,
        "persistent_candidate_store_bytes": store,
        "persistent_total_bytes": bundle + store,
        "projected_checkpoint_bytes": checkpoint,
        "combined_expected_bytes": combined,
        "safety_margin_fraction": SAFETY_MARGIN_FRACTION,
        "recommended_additional_safety_margin_bytes": margin,
        "recommended_minimum_free_bytes_before_launch": free_required,
        "recommended_total_capacity_bytes_for_fresh_deployment": deployment_capacity,
        "current_filesystem_free_bytes": disk.free,
        "preflight_passed": disk.free >= free_required,
        "preflight_rule": (
            "With Task 6 persistent state already present, free space must cover projected "
            "checkpoints plus 25% of the combined persistent-and-checkpoint footprint."
        ),
    }


def enforce_disk_preflight(plan: dict[str, object]) -> None:
    if not plan["preflight_passed"]:
        required = int(plan["recommended_minimum_free_bytes_before_launch"])
        available = int(plan["current_filesystem_free_bytes"])
        raise OSError(
            f"Insufficient disk for future full inference: required free={required:,}, "
            f"available={available:,}"
        )


def batch_contracts(entity_ids: list[str], batch_size: int) -> list[dict[str, object]]:
    result = []
    for start, stop, batch_id in t5.batch_boundaries(len(entity_ids), batch_size):
        values = entity_ids[start:stop]
        result.append(
            {
                "batch_id": batch_id,
                "start_entity_offset": start,
                "stop_entity_offset": stop,
                "entity_count": stop - start,
                "first_source1_entity_id": values[0],
                "last_source1_entity_id": values[-1],
                "entity_ids_sha256": ordered_values_sha256(values),
            }
        )
    return result


def build_checkpoint_run_contract(
    paths: dict[str, Path],
    entity_ids: list[str],
    input_validation: dict[str, object],
    source_sha256: str,
    source_bytes: int,
    batch_size: int,
) -> dict[str, object]:
    contract = {
        "contract_version": "task7a-checkpoint-v1",
        "model_sha256": file_sha256(paths["task4a_model"]),
        "feature_schema_sha256": file_sha256(paths["task3_schema"]),
        "task5_inference_contract_sha256": file_sha256(paths["task5_contract"]),
        "retrieval_manifest_sha256": file_sha256(paths["task6_retrieval_manifest"]),
        "candidate_store_manifest_sha256": file_sha256(paths["task6_store_manifest"]),
        "retrieval_configuration": t6.FROZEN_CONFIGURATION,
        "candidate_cap": t6.FROZEN_CANDIDATE_CAP,
        "threshold": t6.FROZEN_THRESHOLD,
        "source_input": {
            "basename": paths["S1"].name,
            "sha256": source_sha256,
            "bytes": source_bytes,
            "full_source_row_count": 2_206_821,
            "selected_row_count": len(entity_ids),
            "selected_ordered_entity_id_sha256": input_validation["ordered_entity_id_sha256"],
        },
        "batch_size": batch_size,
        "batch_boundaries": batch_contracts(entity_ids, batch_size),
        "output_columns": list(OUTPUT_COLUMNS),
        "output_schema_sha256": value_sha256(list(OUTPUT_COLUMNS)),
        "atomic_commit": "write temporary Parquet, os.replace final file, then atomic manifest update",
        "resume_rule": "Only manifest-recorded batches with matching byte size, SHA-256, schema, boundaries, and decisions are complete.",
    }
    contract["run_contract_sha256"] = value_sha256(contract)
    return contract


def load_or_create_checkpoint_manifest(
    checkpoint_dir: Path,
    run_contract: dict[str, object],
) -> dict[str, object]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "checkpoint_manifest.json"
    expected = {
        key: run_contract[key]
        for key in (
            "contract_version",
            "run_contract_sha256",
            "model_sha256",
            "feature_schema_sha256",
            "task5_inference_contract_sha256",
            "retrieval_manifest_sha256",
            "candidate_store_manifest_sha256",
            "retrieval_configuration",
            "candidate_cap",
            "threshold",
            "source_input",
            "batch_size",
            "batch_boundaries",
            "output_columns",
            "output_schema_sha256",
            "atomic_commit",
            "resume_rule",
        )
    }
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise AssertionError(f"Checkpoint manifest drift for {key}")
        return manifest
    manifest = {**expected, "created_at_utc": utc_now(), "completed_batches": {}}
    atomic_json(path, manifest)
    return manifest


def validate_completed_batch(
    path: Path,
    record: dict[str, object],
    boundary: dict[str, object],
    threshold: float,
) -> None:
    if not path.exists():
        raise AssertionError(f"Completed batch is missing: {path.name}")
    if path.stat().st_size != int(record["bytes"]):
        raise AssertionError(f"Completed batch byte size changed: {path.name}")
    if file_sha256(path) != record["sha256"]:
        raise AssertionError(f"Completed batch hash changed: {path.name}")
    frame = pq.read_table(path).to_pandas()
    if frame.columns.tolist() != list(OUTPUT_COLUMNS):
        raise AssertionError(f"Completed batch schema changed: {path.name}")
    if len(frame) != int(record["pair_rows"]):
        raise AssertionError(f"Completed batch row count changed: {path.name}")
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError(f"Completed batch contains duplicate pairs: {path.name}")
    if set(frame["batch_id"].astype(str)) != {boundary["batch_id"]}:
        raise AssertionError(f"Completed batch ID column changed: {path.name}")
    observed_ids = frame["source1_entity_id"].drop_duplicates().astype(str).tolist()
    if ordered_values_sha256(observed_ids) != boundary["entity_ids_sha256"]:
        raise AssertionError(f"Completed batch S1 order changed: {path.name}")
    decisions = frame["lightgbm_score"].to_numpy(dtype=np.float64) >= threshold
    if not np.array_equal(decisions, frame["predicted_match"].to_numpy(dtype=bool)):
        raise AssertionError(f"Completed batch decisions changed: {path.name}")


def verify_checkpoint_directory(checkpoint_dir: Path, run_contract: dict[str, object]) -> dict[str, int]:
    manifest = load_or_create_checkpoint_manifest(checkpoint_dir, run_contract)
    boundaries = {row["batch_id"]: row for row in run_contract["batch_boundaries"]}
    completed = manifest["completed_batches"]
    unknown = set(completed) - set(boundaries)
    if unknown:
        raise AssertionError(f"Unknown completed batch IDs: {sorted(unknown)}")
    rows = 0
    for batch_id, record in completed.items():
        validate_completed_batch(
            checkpoint_dir / f"{batch_id}.parquet",
            record,
            boundaries[batch_id],
            float(run_contract["threshold"]),
        )
        rows += int(record["pair_rows"])
    return {"completed_batches": len(completed), "pair_rows": rows}


def run_checkpoint_batches(
    checkpoint_dir: Path,
    run_contract: dict[str, object],
    eval_df: pd.DataFrame,
    offsets: np.ndarray,
    metadata: t3.PairMetadata,
    s1_text: dict[str, np.ndarray],
    text_store: t3.TextStore,
    features: list[str],
    model: lgb.Booster,
    workers: int,
    max_new_batches: int | None,
    runtime: RuntimeRecorder | None = None,
) -> dict[str, object]:
    manifest = load_or_create_checkpoint_manifest(checkpoint_dir, run_contract)
    boundaries = {row["batch_id"]: row for row in run_contract["batch_boundaries"]}
    completed = manifest["completed_batches"]
    new_batches = skipped_batches = 0
    phase_totals = {"feature_generation": 0.0, "lightgbm_inference": 0.0, "checkpoint_write": 0.0}
    for start, stop, batch_id in t5.batch_boundaries(len(eval_df), int(run_contract["batch_size"])):
        path = checkpoint_dir / f"{batch_id}.parquet"
        boundary = boundaries[batch_id]
        if batch_id in completed:
            validate_completed_batch(path, completed[batch_id], boundary, float(run_contract["threshold"]))
            skipped_batches += 1
            continue
        if max_new_batches is not None and new_batches >= max_new_batches:
            break

        batch_metadata = t5.metadata_for_query_range(metadata, offsets, start, stop)
        feature_started = time.perf_counter()
        frame = t5.build_inference_feature_frame(
            batch_metadata, eval_df, s1_text, text_store, workers
        )
        feature_seconds = time.perf_counter() - feature_started
        matrix = frame[features].to_numpy(dtype=np.float32, copy=True)
        if matrix.shape[1] != 66 or not np.isfinite(matrix).all():
            raise AssertionError(f"Invalid feature matrix for {batch_id}")
        model_started = time.perf_counter()
        scores = model.predict(matrix).astype(np.float64)
        model_seconds = time.perf_counter() - model_started
        output = pd.DataFrame(
            {
                "source1_entity_id": frame["source1_entity_id"].astype(str),
                "candidate_entity_id": frame["candidate_entity_id"].astype(str),
                "candidate_source": frame["candidate_source"].astype(str),
                "lightgbm_score": scores,
                "predicted_match": scores >= float(run_contract["threshold"]),
                "candidate_position": frame["frozen_candidate_position"].to_numpy(dtype=np.uint16),
                "batch_id": batch_id,
            }
        )
        if output.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise AssertionError(f"Duplicate pairs generated for {batch_id}")
        write_started = time.perf_counter()
        t5.write_prediction_batch(path, output)
        write_seconds = time.perf_counter() - write_started
        record = {
            **boundary,
            "pair_rows": len(output),
            "predicted_links": int(output["predicted_match"].sum()),
            "feature_seconds": feature_seconds,
            "model_seconds": model_seconds,
            "write_seconds": write_seconds,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "output_schema_sha256": run_contract["output_schema_sha256"],
            "committed_at_utc": utc_now(),
        }
        completed[batch_id] = record
        manifest["completed_batches"] = completed
        atomic_json(checkpoint_dir / "checkpoint_manifest.json", manifest)
        validate_completed_batch(path, record, boundary, float(run_contract["threshold"]))
        phase_totals["feature_generation"] += feature_seconds
        phase_totals["lightgbm_inference"] += model_seconds
        phase_totals["checkpoint_write"] += write_seconds
        new_batches += 1
        del frame, matrix, scores, output
        gc.collect()
    if runtime is not None:
        runtime.record("feature_generation", phase_totals["feature_generation"], "candidate-pair count")
        runtime.record("lightgbm_inference", phase_totals["lightgbm_inference"], "candidate-pair count")
        runtime.record("checkpoint_write", phase_totals["checkpoint_write"], "output row/byte count")
    return {
        "new_batches": new_batches,
        "skipped_batches": skipped_batches,
        **phase_totals,
    }


def checkpoint_predictions(checkpoint_dir: Path, run_contract: dict[str, object]) -> pd.DataFrame:
    verify_checkpoint_directory(checkpoint_dir, run_contract)
    manifest = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    order = [row["batch_id"] for row in run_contract["batch_boundaries"]]
    if set(manifest["completed_batches"]) != set(order):
        raise AssertionError("Checkpoint set is incomplete")
    frames = [pq.read_table(checkpoint_dir / f"{batch_id}.parquet").to_pandas() for batch_id in order]
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Duplicate pairs across completed batches")
    return result


def failure_simulation(
    root: Path,
    paths: dict[str, Path],
    eval_df: pd.DataFrame,
    offsets: np.ndarray,
    metadata: t3.PairMetadata,
    s1_text: dict[str, np.ndarray],
    text_store: t3.TextStore,
    features: list[str],
    model: lgb.Booster,
    input_validation: dict[str, object],
    source_sha256: str,
    workers: int,
) -> dict[str, object]:
    checkpoint_dir = root / "checkpoint_resume"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Task 7A failure simulation already exists: {checkpoint_dir}")
    entity_ids = eval_df["source1_entity_id"].astype(str).tolist()
    contract = build_checkpoint_run_contract(
        paths,
        entity_ids,
        input_validation,
        source_sha256,
        paths["S1"].stat().st_size,
        FAILURE_BATCH_SIZE,
    )
    interrupted = run_checkpoint_batches(
        checkpoint_dir,
        contract,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        workers,
        max_new_batches=1,
    )
    after_interrupt = verify_checkpoint_directory(checkpoint_dir, contract)
    restarted = run_checkpoint_batches(
        checkpoint_dir,
        contract,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        workers,
        max_new_batches=None,
    )
    after_restart = verify_checkpoint_directory(checkpoint_dir, contract)
    completed_rerun = run_checkpoint_batches(
        checkpoint_dir,
        contract,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        workers,
        max_new_batches=None,
    )

    orphan = checkpoint_dir / "uncommitted_batch.tmp.parquet"
    orphan.write_bytes(b"deliberately invalid uncommitted temporary output")
    orphan_ignored = verify_checkpoint_directory(checkpoint_dir, contract)["completed_batches"] == 2

    corrupt_dir = root / "corrupt_completed_detection"
    shutil.copytree(checkpoint_dir, corrupt_dir)
    corrupt_batch = corrupt_dir / "batch_00000.parquet"
    with corrupt_batch.open("r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 0xFF]))
    corruption_detected = False
    error = ""
    try:
        verify_checkpoint_directory(corrupt_dir, contract)
    except (AssertionError, OSError, ValueError) as exc:
        corruption_detected = True
        error = str(exc)
    return {
        "batch_size": FAILURE_BATCH_SIZE,
        "interrupted_after_one_completed_batch": interrupted,
        "verified_after_interruption": after_interrupt,
        "restart_result": restarted,
        "verified_after_restart": after_restart,
        "completed_rerun_result": completed_rerun,
        "orphan_temporary_file_ignored_as_uncommitted": orphan_ignored,
        "corrupt_completed_batch_detected": corruption_detected,
        "corruption_error": error,
        "passed": (
            interrupted["new_batches"] == 1
            and after_interrupt["completed_batches"] == 1
            and restarted["new_batches"] == 1
            and restarted["skipped_batches"] == 1
            and after_restart["completed_batches"] == 2
            and completed_rerun["new_batches"] == 0
            and completed_rerun["skipped_batches"] == 2
            and orphan_ignored
            and corruption_detected
        ),
    }


def parity_pass(rows: list[dict[str, object]]) -> bool:
    return bool(rows) and all(bool(row["passed"]) for row in rows)


def integrity_row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def write_readiness_report(
    path: Path,
    preflight: dict[str, object],
    storage: dict[str, object],
    checkpoint_audit: dict[str, object],
    integrity: pd.DataFrame,
) -> None:
    runtime = preflight["runtime_seconds"]
    reconciliation = preflight["scale_reconciliation"]
    text = f"""# Task 7A Full-Scale Inference Readiness Audit

Status: **PASS**

## Decision

The frozen production pipeline is operationally ready for a separately authorized Task 7B full-inference run. Task 7A did not launch that run, access test data, create test predictions, or generate a submission.

## Scale reconciliation

- Canonical estimate: **{reconciliation['canonical_projected_hours']:.2f} hours** for {reconciliation['projected_candidate_pairs']:,} pairs.
- Canonical source: `task6_scale_estimate.json`, derived from {reconciliation['canonical_smoke_seconds']:.3f} seconds for 1,000 S1 entities.
- The stale 18.65-hour estimate was a linear extrapolation from an earlier {reconciliation['stale_smoke_seconds']:.3f}-second Task 6 total that excluded artifact hash validation and used the earlier aggregate downstream timing.
- Current Task 6 evidence includes pre-use hash validation and the explicit feature, model, and checkpoint-write phases. The final recommendation in `PROJECT_STATUS_SUMMARY.md` was stale and was corrected to 20.02 hours.

## Storage readiness

- Retrieval bundle: {storage['persistent_retrieval_bundle_bytes'] / 2**30:.2f} GiB.
- Candidate text store: {storage['persistent_candidate_store_bytes'] / 2**30:.2f} GiB.
- Projected checkpoints: {storage['projected_checkpoint_bytes'] / 2**30:.2f} GiB.
- Combined expected footprint: {storage['combined_expected_bytes'] / 2**30:.2f} GiB.
- Recommended safety margin: {storage['recommended_additional_safety_margin_bytes'] / 2**30:.2f} GiB ({storage['safety_margin_fraction']:.0%}).
- Minimum free space before launch with Task 6 state already present: {storage['recommended_minimum_free_bytes_before_launch'] / 2**30:.2f} GiB.
- Fresh-deployment capacity including margin: {storage['recommended_total_capacity_bytes_for_fresh_deployment'] / 2**30:.2f} GiB.
- Current free space: {storage['current_filesystem_free_bytes'] / 2**30:.2f} GiB; disk preflight PASS.

## Measured 100-S1 smoke

- Candidate pairs: {preflight['smoke']['pair_rows']:,}; output: {preflight['smoke']['output_bytes'] / 2**20:.3f} MiB.
- Total readiness-smoke runtime: {runtime['total']:.3f} seconds.
- Hash validation: {runtime['artifact_hash_validation']:.3f}s; bundle load: {runtime['bundle_load']:.3f}s; retrieval query: {runtime['retrieval_query']:.3f}s.
- Candidate assembly: {runtime['candidate_assembly']:.3f}s; selective text lookup: {runtime['candidate_text_lookup']:.3f}s.
- Feature generation: {runtime['feature_generation']:.3f}s; LightGBM: {runtime['lightgbm_inference']:.3f}s; checkpoint write: {runtime['checkpoint_write']:.3f}s.
- Peak RSS: {preflight['smoke']['peak_rss_mb']:.1f} MB.
- Retrieval, candidate order, all 66 features, LightGBM scores, threshold decisions, and the final Parquet batch matched the frozen pipeline.

## Runtime scaling

- Fixed startup: source/input fingerprint validation, persisted-artifact hash validation, and retrieval-bundle loading.
- Primarily S1-count scaling: five-signal retrieval queries and candidate assembly.
- Primarily candidate-pair scaling: selective text lookup, 66-feature generation, LightGBM inference, and checkpoint writing.
- In the Task 6 1,000-S1 benchmark, selective lookup (11.58s) and LightGBM inference (9.01s) were the largest measured repeatable stages, followed by feature generation (5.79s).
- Operational constraint: Task 7A reproduced a native mixed-OpenMP exit with eight FAISS workers. The audited smoke therefore pins one native worker and remains byte-identical. Task 7B should retain one worker unless a separate preflight proves a higher count stable.

## Checkpoint and input contracts

- Checkpoint audit: PASS. The Task 7A contract explicitly binds model, feature schema, Task 5 contract, Task 6 retrieval/store manifests, threshold, source input fingerprint, deterministic batch boundaries, and output schema.
- Completed batches are committed by temporary Parquet plus `os.replace`, followed by an atomic manifest update.
- Restart verified one completed batch, skipped it, completed the remaining batch, and a completed rerun became a no-op.
- A deliberately corrupted completed batch was rejected by SHA-256/size verification; an invalid uncommitted temporary file was not treated as complete.
- Input contract: PASS on 100 training S1 records. Missing column, duplicate ID, unknown country, and both-text-missing negative tests all failed safely.

## Output boundary

Internal checkpoints contain: `source1_entity_id`, `candidate_entity_id`, `candidate_source`, `lightgbm_score`, `predicted_match`, `candidate_position`, and `batch_id`.

The official competition submission schema, required column naming/order, grouping rules, empty-match representation, and file-format constraints are not established in repository artifacts. This remains an external contract and must be handled only by a later submission-transformation task. It does not block internal inference readiness.

## Recommendation

Task 7B may be authorized only as a separate request after confirming at least {storage['recommended_minimum_free_bytes_before_launch'] / 2**30:.2f} GiB free space, fixing the launch input fingerprint in the run manifest, retaining batch size 100 and one native worker initially, and using the strengthened checkpoint contract. No unresolved blocker prevents inference; the submission format remains deliberately unresolved.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = make_parser().parse_args()
    if args.mode == "full":
        raise RuntimeError("Task 7A full-scale inference is safety-locked; audit only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(args)
    retrieval_manifest = json.loads(paths["task6_retrieval_manifest"].read_text(encoding="utf-8"))
    store_manifest = json.loads(paths["task6_store_manifest"].read_text(encoding="utf-8"))
    scale = json.loads(paths["task6_scale"].read_text(encoding="utf-8"))
    input_contract = build_input_contract(paths, retrieval_manifest)

    if args.mode == "disk-preflight":
        plan = build_storage_plan(scale, args.output_dir)
        enforce_disk_preflight(plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.mode == "input-preflight":
        if args.input_path is None:
            raise ValueError("--input-path is required for input-preflight")
        print(json.dumps(validate_input_path(args.input_path, input_contract, args.chunksize), indent=2, sort_keys=True))
        return

    output_files = [
        "task7a_readiness_report.md",
        "task7a_preflight_results.json",
        "task7a_runtime_breakdown.csv",
        "task7a_storage_plan.json",
        "task7a_input_contract.json",
        "task7a_checkpoint_audit.json",
        "task7a_integrity_report.csv",
        "task7a_run_manifest.json",
    ]
    existing = [name for name in output_files if (args.output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Task 7A final outputs already exist: {existing}")
    hashes_before = frozen_hashes(paths)
    task6_run = json.loads(paths["task6_run_manifest"].read_text(encoding="utf-8"))
    task6_integrity = pd.read_csv(paths["task6_integrity"])
    if task6_run["status"] != "PASS" or not task6_integrity["passed"].all():
        raise AssertionError("Task 6 is not in a verified PASS state")
    features, feature_schema, model = t6.validate_frozen_contract(paths, args)
    if len(features) != 66 or model.feature_name() != features:
        raise AssertionError("Frozen model/feature contract changed")

    selection = t5.select_smoke_entities(paths["task3_split"], SMOKE_ENTITIES)
    eval_df = t5.load_raw_s1(selection, paths["S1"], args.chunksize)
    input_frame = eval_df[
        ["source1_entity_id", "business_name", "business_address", "country"]
    ].rename(columns={"source1_entity_id": "entity_id"})
    input_validation = validate_input_frame(input_frame, input_contract)
    negative_tests = negative_input_tests(input_frame, input_contract)
    if not input_validation["passed"] or not all(row["passed"] for row in negative_tests):
        raise AssertionError("Input preflight did not fail/pass as expected")
    atomic_json(args.output_dir / "task7a_input_contract.json", input_contract)

    storage = build_storage_plan(scale, args.output_dir)
    enforce_disk_preflight(storage)
    atomic_json(args.output_dir / "task7a_storage_plan.json", storage)

    runtime = RuntimeRecorder()
    total_started = time.perf_counter()
    source_hash_started = time.perf_counter()
    source_sha256 = file_sha256(paths["S1"])
    if source_sha256 != task6_run["frozen_hashes_before"]["S1"]:
        raise AssertionError("Training S1 source hash changed since Task 6")
    runtime.record(
        "input_source_hash_validation",
        time.perf_counter() - source_hash_started,
        "fixed startup/source bytes",
    )

    artifact_started = time.perf_counter()
    retrieval_hash_rows = t6.verify_manifest_artifacts(
        retrieval_manifest, args.task6_output_dir, "retrieval_bundle"
    )
    store_hash_rows = t6.verify_manifest_artifacts(
        store_manifest, args.task6_output_dir, "candidate_store"
    )
    runtime.record(
        "artifact_hash_validation",
        time.perf_counter() - artifact_started,
        "fixed startup/persistent bytes",
    )

    checkpoint = t6.query_retrieval_bundle(
        retrieval_manifest,
        args.task6_output_dir,
        eval_df,
        args,
        runtime,
        "preflight",
    )
    global_indices = eval_df["eval_index"].to_numpy(dtype=np.int64)
    retrieval_rows = t5.compare_retrieval_checkpoint(
        checkpoint, paths["task2_5_checkpoint"], global_indices
    )
    assembly_started = time.perf_counter()
    offsets, counts, metadata, candidate_sets = t5.assemble_candidates(eval_df, checkpoint)
    runtime.record(
        "candidate_assembly",
        time.perf_counter() - assembly_started,
        "S1 and candidate-pair count",
    )
    retrieval_rows.extend(
        t5.compare_frozen_candidates(global_indices, candidate_sets, paths["task3_candidates"])
    )
    if not parity_pass(retrieval_rows):
        raise AssertionError("100-S1 persisted retrieval parity failed")

    store_init_started = time.perf_counter()
    persisted_store = t6.PersistedTextStore(
        args.task6_output_dir / "candidate_text_store", store_manifest
    )
    runtime.record(
        "candidate_store_initialization",
        time.perf_counter() - store_init_started,
        "fixed startup",
    )
    lookup_started = time.perf_counter()
    text_store = persisted_store.lookup(metadata.candidate_ids)
    runtime.record(
        "candidate_text_lookup",
        time.perf_counter() - lookup_started,
        "unique candidate IDs/candidate-pair count",
    )
    s1_text = t3.build_s1_text(eval_df)

    entity_ids = eval_df["source1_entity_id"].astype(str).tolist()
    run_contract = build_checkpoint_run_contract(
        paths,
        entity_ids,
        input_validation,
        source_sha256,
        paths["S1"].stat().st_size,
        SMOKE_BATCH_SIZE,
    )
    checkpoint_dir = args.output_dir / "preflight_checkpoints"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Task 7A checkpoint output already exists: {checkpoint_dir}")
    batch_result = run_checkpoint_batches(
        checkpoint_dir,
        run_contract,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        args.workers,
        max_new_batches=None,
        runtime=runtime,
    )
    predictions = checkpoint_predictions(checkpoint_dir, run_contract)
    total_seconds = time.perf_counter() - total_started
    runtime.record("total_preflight_smoke", total_seconds, "100-S1 controlled smoke")

    parity_started = time.perf_counter()
    fresh_frame = t5.build_inference_feature_frame(
        metadata, eval_df, s1_text, text_store, args.workers
    )
    parity_ids = set(entity_ids)
    reference_features = t5.read_reference_pair_features(paths["task3_pairs"], parity_ids, features)
    reference_scores = t5.read_reference_scores(paths["task4a_predictions"], parity_ids)
    feature_rows, fresh_scores = t5.compare_features_and_scores(
        fresh_frame,
        features,
        feature_schema,
        model,
        reference_features,
        reference_scores,
    )
    runtime.record(
        "frozen_prediction_parity_validation",
        time.perf_counter() - parity_started,
        "validation-only; excluded from production total",
    )
    if not parity_pass(feature_rows):
        raise AssertionError("100-S1 feature/model parity failed")
    aligned_predictions = predictions.assign(
        _query_order=predictions["source1_entity_id"].map(
            {entity_id: index for index, entity_id in enumerate(entity_ids)}
        )
    ).sort_values(["_query_order", "candidate_position"], kind="stable").reset_index(drop=True)
    aligned_frame = fresh_frame.sort_values(
        ["eval_index", "frozen_candidate_position"], kind="stable"
    ).reset_index(drop=True)
    pair_keys_match = np.array_equal(
        aligned_predictions["source1_entity_id"].astype(str).to_numpy(),
        aligned_frame["source1_entity_id"].astype(str).to_numpy(),
    ) and np.array_equal(
        aligned_predictions["candidate_entity_id"].astype(str).to_numpy(),
        aligned_frame["candidate_entity_id"].astype(str).to_numpy(),
    )
    if not pair_keys_match:
        raise AssertionError("Task 7A checkpoint pair keys/order differ from parity frame")
    if not np.array_equal(
        aligned_predictions["predicted_match"].to_numpy(dtype=bool),
        fresh_scores >= t6.FROZEN_THRESHOLD,
    ):
        raise AssertionError("Task 7A checkpoint decisions differ from parity scores")
    smoke_batch_path = checkpoint_dir / "batch_00000.parquet"
    batch_hash_exact = file_sha256(smoke_batch_path) == file_sha256(paths["task6_smoke_batch"])
    if not batch_hash_exact:
        raise AssertionError("Task 7A checkpoint batch differs byte-for-byte from Task 6")

    failure_root = args.output_dir / "failure_simulation"
    if failure_root.exists():
        raise FileExistsError(f"Task 7A failure-simulation output exists: {failure_root}")
    failure_result = failure_simulation(
        failure_root,
        paths,
        eval_df,
        offsets,
        metadata,
        s1_text,
        text_store,
        features,
        model,
        input_validation,
        source_sha256,
        args.workers,
    )
    if not failure_result["passed"]:
        raise AssertionError("Checkpoint failure simulation failed")

    task6_runtime = pd.read_csv(paths["task6_runtime"])
    current_seconds = float(scale["measured_reload_only_seconds"])
    stale_hours = 18.648496170027084
    stale_seconds = stale_hours * 3600.0 / int(scale["target_s1_entities"]) * 1000.0
    current_hours = float(scale["projected_wall_hours_linear"])
    status_text = paths["project_status"].read_text(encoding="utf-8")
    reconciliation = {
        "canonical_projected_hours": current_hours,
        "canonical_smoke_seconds": current_seconds,
        "canonical_source": str(paths["task6_scale"].relative_to(PROJECT_ROOT)),
        "projected_candidate_pairs": int(scale["projected_candidate_pairs"]),
        "stale_project_status_hours": stale_hours,
        "stale_smoke_seconds": stale_seconds,
        "stale_value_present_in_project_status_before_task7a": "18.65-hour" in status_text,
        "reason_current": (
            "Task 6 scale estimate uses 32.654662 seconds for 1,000 S1 and includes "
            "artifact hash validation plus explicit retrieval, lookup, feature, model, and write phases."
        ),
        "reason_stale": (
            "The 18.65-hour value is the earlier linear extrapolation from approximately "
            "30.420 seconds per 1,000 S1 and excludes pre-use artifact hash validation."
        ),
    }

    phase_lookup = {row["phase"]: row for row in runtime.rows}
    preflight = {
        "task": TASK_NAME,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "scale_reconciliation": reconciliation,
        "input_validation": input_validation,
        "negative_input_tests": negative_tests,
        "smoke": {
            "s1_entities": len(eval_df),
            "batch_size": SMOKE_BATCH_SIZE,
            "native_workers": args.workers,
            "pair_rows": len(predictions),
            "candidate_minimum": int(counts.min()),
            "candidate_average": float(counts.mean()),
            "candidate_maximum": int(counts.max()),
            "predicted_links": int(predictions["predicted_match"].sum()),
            "output_bytes": smoke_batch_path.stat().st_size,
            "output_sha256": file_sha256(smoke_batch_path),
            "task6_batch_sha256": file_sha256(paths["task6_smoke_batch"]),
            "task6_batch_byte_exact": batch_hash_exact,
            "peak_rss_mb": max(float(row["process_peak_rss_mb"]) for row in runtime.rows),
        },
        "runtime_seconds": {
            "input_source_hash_validation": phase_lookup["input_source_hash_validation"]["runtime_seconds"],
            "artifact_hash_validation": phase_lookup["artifact_hash_validation"]["runtime_seconds"],
            "bundle_load": phase_lookup["preflight_retrieval_bundle_load"]["runtime_seconds"],
            "retrieval_query": phase_lookup["preflight_retrieval_query"]["runtime_seconds"],
            "candidate_assembly": phase_lookup["candidate_assembly"]["runtime_seconds"],
            "candidate_store_initialization": phase_lookup["candidate_store_initialization"]["runtime_seconds"],
            "candidate_text_lookup": phase_lookup["candidate_text_lookup"]["runtime_seconds"],
            "feature_generation": phase_lookup["feature_generation"]["runtime_seconds"],
            "lightgbm_inference": phase_lookup["lightgbm_inference"]["runtime_seconds"],
            "checkpoint_write": phase_lookup["checkpoint_write"]["runtime_seconds"],
            "total": total_seconds,
        },
        "parity": {
            "retrieval_checks": retrieval_rows,
            "feature_model_checks": feature_rows,
            "all_passed": parity_pass(retrieval_rows) and parity_pass(feature_rows) and batch_hash_exact,
        },
        "batch_result": batch_result,
        "failure_simulation_passed": failure_result["passed"],
        "operational_constraints": [
            "Use one native worker initially; eight-worker FAISS reproduced a mixed-OpenMP native exit during Task 7A preflight.",
            "Treat the 20.02-hour figure as a linear Task 6 planning estimate, not a full-run measurement.",
        ],
        "full_scale_run": False,
        "test_data_used": False,
        "submission_generated": False,
    }

    checkpoint_audit = {
        "status": "PASS",
        "legacy_task5_findings": {
            "atomic_batch_write": True,
            "atomic_manifest_write": True,
            "completed_hash_verified_on_resume": True,
            "deterministic_boundaries": True,
            "no_op_after_completion": True,
            "metadata_gap": (
                "Legacy manifest binds the Task 5 contract hash and selected-ID hash but does "
                "not explicitly record the Task 6 retrieval/store manifests or full source-file fingerprint."
            ),
        },
        "task7a_safety_fix": {
            "prediction_logic_changed": False,
            "run_contract": run_contract,
            "manifest_path": str((checkpoint_dir / "checkpoint_manifest.json").relative_to(PROJECT_ROOT)),
            "guarantees": [
                "batch Parquet committed atomically before manifest completion record",
                "manifest updated atomically",
                "completed file byte size and SHA-256 verified on every resume",
                "schema, row count, batch ID, S1 order, duplicate pairs, and threshold decisions verified",
                "model, feature schema, Task 5 contract, Task 6 retrieval/store manifests, threshold, source input, and boundaries explicitly bound",
                "only manifest-listed final files count as complete",
            ],
        },
        "failure_simulation": failure_result,
        "output_submission_boundary": {
            "internal_checkpoint_columns": list(OUTPUT_COLUMNS),
            "official_submission_schema_known": False,
            "missing_external_contract": [
                "official filename and file format",
                "official column names and order",
                "whether predictions are one row per link or grouped per S1",
                "required ordering and empty-match representation",
            ],
            "blocks_internal_inference": False,
            "submission_transformation_performed": False,
        },
        "operational_constraints": {
            "initial_native_workers": 1,
            "reason": "Eight-worker FAISS reproduced a mixed-OpenMP native exit; one worker passed with byte-exact output.",
            "prediction_effect": "none",
        },
    }

    hashes_after = frozen_hashes(paths)
    integrity_rows = [
        integrity_row("frozen_inputs_unchanged", hashes_before == hashes_after, "All Task 2.5 through Task 6 inputs retained their pre-run SHA-256."),
        integrity_row("task6_status_pass", task6_run["status"] == "PASS", "Task 6 run manifest PASS."),
        integrity_row("task6_integrity_pass", bool(task6_integrity["passed"].all()), f"{len(task6_integrity)} checks"),
        integrity_row("retrieval_hashes_valid", all(row["passed"] for row in retrieval_hash_rows), f"{len(retrieval_hash_rows)} artifacts"),
        integrity_row("candidate_store_hashes_valid", all(row["passed"] for row in store_hash_rows), f"{len(store_hash_rows)} artifacts"),
        integrity_row("scale_canonical_20_02", abs(current_hours - 20.017498265168324) < 1e-9, f"hours={current_hours}"),
        integrity_row("stale_18_65_traced", abs(stale_seconds - 30.420294) < 0.01, f"earlier_seconds={stale_seconds:.6f}"),
        integrity_row("disk_preflight", bool(storage["preflight_passed"]), f"free={storage['current_filesystem_free_bytes']:,}"),
        integrity_row("input_contract_positive", bool(input_validation["passed"]), f"rows={input_validation['row_count']}"),
        integrity_row("input_contract_negative_tests", all(row["passed"] for row in negative_tests), f"tests={len(negative_tests)}"),
        integrity_row("source_hash_frozen", source_sha256 == task6_run["frozen_hashes_before"]["S1"], source_sha256),
        integrity_row("retrieval_parity", parity_pass(retrieval_rows), f"checks={len(retrieval_rows)}"),
        integrity_row("feature_model_parity", parity_pass(feature_rows), f"checks={len(feature_rows)}"),
        integrity_row("task6_batch_byte_exact", batch_hash_exact, file_sha256(smoke_batch_path)),
        integrity_row("feature_count_66", len(features) == 66, f"features={len(features)}"),
        integrity_row("threshold_frozen", float(run_contract["threshold"]) == 0.95, "score >= 0.95"),
        integrity_row("candidate_cap_frozen", int(counts.max()) <= 250, f"max={int(counts.max())}"),
        integrity_row("no_duplicate_pairs", not predictions.duplicated(["source1_entity_id", "candidate_entity_id"]).any(), f"pairs={len(predictions)}"),
        integrity_row("atomic_batch_and_manifest", failure_result["verified_after_interruption"]["completed_batches"] == 1, "One committed batch survived interruption."),
        integrity_row("restart_skips_completed", failure_result["restart_result"]["skipped_batches"] == 1, str(failure_result["restart_result"])),
        integrity_row("completed_rerun_no_op", failure_result["completed_rerun_result"]["new_batches"] == 0, str(failure_result["completed_rerun_result"])),
        integrity_row("corrupt_batch_detected", bool(failure_result["corrupt_completed_batch_detected"]), failure_result["corruption_error"]),
        integrity_row("uncommitted_temp_not_complete", bool(failure_result["orphan_temporary_file_ignored_as_uncommitted"]), "Temporary file absent from completion manifest."),
        integrity_row("deterministic_batch_order", [row["batch_id"] for row in run_contract["batch_boundaries"]] == ["batch_00000"], str(run_contract["batch_boundaries"])),
        integrity_row("output_schema_exact", predictions.columns.tolist() == list(OUTPUT_COLUMNS), str(predictions.columns.tolist())),
        integrity_row("smoke_only_100_s1", len(eval_df) == 100, "No full-scale inference."),
        integrity_row("no_ground_truth_runtime", "GT" not in paths, "Ground truth was not resolved."),
        integrity_row("no_test_data", all("test" not in paths[key].name.casefold() for key in ("S1", "S2", "S3")), "Training sources only."),
        integrity_row("no_submission", not preflight["submission_generated"], "Internal checkpoint output only."),
        integrity_row("full_scale_locked", not preflight["full_scale_run"], "--mode full raises before input processing."),
        integrity_row("single_worker_safety_mode", args.workers == 1, "One native worker used after eight-worker OpenMP instability."),
    ]
    integrity = pd.DataFrame(integrity_rows)
    if not integrity["passed"].all():
        raise AssertionError(f"Task 7A integrity failed: {integrity.loc[~integrity['passed']].to_dict('records')}")

    atomic_json(args.output_dir / "task7a_preflight_results.json", preflight)
    atomic_json(args.output_dir / "task7a_checkpoint_audit.json", checkpoint_audit)
    atomic_csv(args.output_dir / "task7a_runtime_breakdown.csv", pd.DataFrame(runtime.rows))
    atomic_csv(args.output_dir / "task7a_integrity_report.csv", integrity)
    report_path = args.output_dir / "task7a_readiness_report.md"
    write_readiness_report(report_path, preflight, storage, checkpoint_audit, integrity)

    artifacts = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "task7a_run_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(args.output_dir)),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    run_manifest = {
        "task": TASK_NAME,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "scope": "100-S1 training-only readiness audit",
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "canonical_projected_hours": current_hours,
        "projected_candidate_pairs": int(scale["projected_candidate_pairs"]),
        "minimum_free_bytes_before_launch": storage["recommended_minimum_free_bytes_before_launch"],
        "smoke_s1_entities": len(eval_df),
        "smoke_pair_rows": len(predictions),
        "native_workers": args.workers,
        "smoke_total_seconds": total_seconds,
        "peak_rss_mb": preflight["smoke"]["peak_rss_mb"],
        "checkpoint_resume_status": "PASS",
        "input_contract_status": "PASS",
        "submission_schema_status": "unresolved external contract; does not block inference",
        "task7b_authorized": False,
        "boundaries": {
            "full_scale_inference": False,
            "test_data_accessed": False,
            "test_predictions": False,
            "submission_generated": False,
            "model_or_threshold_changed": False,
            "retrieval_or_features_changed": False,
        },
        "artifacts": artifacts,
    }
    atomic_json(args.output_dir / "task7a_run_manifest.json", run_manifest)
    log("Task 7A PASS: full-scale inference is operationally ready but remains unlaunched")


if __name__ == "__main__":
    main()
