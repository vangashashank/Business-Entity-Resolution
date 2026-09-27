#!/usr/bin/env python3
"""Task 7B: full-scale inference over all training S1 entities.

The production pipeline is frozen. This module only orchestrates persisted
retrieval, selective text lookup, 66-feature generation, saved LightGBM scoring,
atomic Parquet checkpoints, transactional resume state, and final validation.
Ground truth and competition test data are never resolved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
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

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import task2_candidate_generation as t2
import task3_pairwise_features as t3
import task5_full_inference_pipeline as t5
import task6_persist_production_retrieval as t6
import task7a_full_inference_readiness as t7a


TASK_NAME = "Task 7B"
EXPECTED_S1_ROWS = 2_206_821
BATCH_SIZE = 100
NATIVE_WORKERS = 1
RETRIEVAL_CHUNK_SIZE = 1_000
EXPECTED_BATCHES = (EXPECTED_S1_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight-only", "run", "validate-only"),
        default="run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource/dataset/train",
    )
    parser.add_argument(
        "--task6-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/task6_outputs",
    )
    parser.add_argument(
        "--task7a-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/task7a_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/task7b_outputs",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    return parser


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = t5.resolve_paths(args)
    paths.update(
        {
            "task5_contract": PROJECT_ROOT / "outputs/task5_outputs/task5_inference_contract.json",
            "task6_run": args.task6_output_dir / "task6_run_manifest.json",
            "task6_retrieval": args.task6_output_dir / "task6_retrieval_manifest.json",
            "task6_store": args.task6_output_dir / "task6_candidate_store_manifest.json",
            "task6_scale": args.task6_output_dir / "task6_scale_estimate.json",
            "task6_integrity": args.task6_output_dir / "task6_integrity_report.csv",
            "task7a_run": args.task7a_output_dir / "task7a_run_manifest.json",
            "task7a_integrity": args.task7a_output_dir / "task7a_integrity_report.csv",
            "task7a_contract": args.task7a_output_dir / "task7a_input_contract.json",
            "task5_code": PROJECT_ROOT / "src/task5_full_inference_pipeline.py",
            "task6_code": PROJECT_ROOT / "src/task6_persist_production_retrieval.py",
            "task7a_code": PROJECT_ROOT / "src/task7a_full_inference_readiness.py",
            "task7b_code": Path(__file__).resolve(),
            "project_status": PROJECT_ROOT / "PROJECT_STATUS_SUMMARY.md",
        }
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 7B inputs: {missing}")
    for key in ("S1", "S2", "S3"):
        if "test" in paths[key].name.casefold():
            raise ValueError(f"Task 7B cannot access test data: {paths[key]}")
    if "GT" in paths:
        raise AssertionError("Ground truth entered Task 7B path resolution")
    return paths


def upstream_hashes(paths: dict[str, Path]) -> dict[str, str]:
    keys = (
        "S1",
        "S2",
        "S3",
        "task2_5_checkpoint",
        "task2_5_decision",
        "task2_5_manifest",
        "task3_schema",
        "task3_audit",
        "task3_candidates",
        "task3_manifest",
        "task4a_model",
        "task4a_manifest",
        "task4b_policy",
        "task4b_manifest",
        "task5_contract",
        "task6_run",
        "task6_retrieval",
        "task6_store",
        "task6_scale",
        "task6_integrity",
        "task7a_run",
        "task7a_integrity",
        "task7a_contract",
        "task5_code",
        "task6_code",
        "task7a_code",
        "task7b_code",
    )
    return {key: file_sha256(paths[key]) for key in keys}


def scan_full_input(
    path: Path,
    input_contract: dict[str, object],
    chunksize: int,
) -> dict[str, object]:
    started = time.perf_counter()
    source_sha256 = file_sha256(path)
    seen: set[str] = set()
    ordered_digest = hashlib.sha256()
    boundaries: list[dict[str, object]] = []
    pending: list[str] = []
    row_count = 0
    country_counts: Counter[str] = Counter()
    first_ordered_value = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            usecols=input_contract["required_columns_in_order"],
            dtype="string",
            chunksize=chunksize,
        ),
        start=1,
    ):
        result = t7a.validate_input_frame(chunk, input_contract)
        if not result["passed"]:
            raise AssertionError(
                f"Full S1 input contract failed near row {row_count}: {result['checks']}"
            )
        ids = chunk["entity_id"].astype(str).tolist()
        if len(set(ids)) != len(ids):
            raise AssertionError(f"Duplicate S1 ID inside input chunk {chunk_number}")
        overlap = seen.intersection(ids)
        if overlap:
            raise AssertionError(f"Duplicate S1 IDs across chunks: {sorted(overlap)[:10]}")
        seen.update(ids)
        countries = t2.normalize_country_series(chunk["country"])
        country_counts.update(countries.astype(str).tolist())
        for entity_id in ids:
            if not first_ordered_value:
                ordered_digest.update(b"\n")
            ordered_digest.update(entity_id.encode("utf-8"))
            first_ordered_value = False
            pending.append(entity_id)
            if len(pending) == BATCH_SIZE:
                start = row_count + 1 - len(pending)
                batch_id = f"batch_{len(boundaries):05d}"
                boundaries.append(
                    {
                        "batch_id": batch_id,
                        "start_entity_offset": start,
                        "stop_entity_offset": start + len(pending),
                        "entity_count": len(pending),
                        "first_source1_entity_id": pending[0],
                        "last_source1_entity_id": pending[-1],
                        "entity_ids_sha256": ordered_values_sha256(pending),
                    }
                )
                pending = []
            row_count += 1
        if chunk_number % 5 == 0:
            log(f"Input preflight: validated {row_count:,} S1 rows")
    if pending:
        start = row_count - len(pending)
        boundaries.append(
            {
                "batch_id": f"batch_{len(boundaries):05d}",
                "start_entity_offset": start,
                "stop_entity_offset": row_count,
                "entity_count": len(pending),
                "first_source1_entity_id": pending[0],
                "last_source1_entity_id": pending[-1],
                "entity_ids_sha256": ordered_values_sha256(pending),
            }
        )
    if row_count != EXPECTED_S1_ROWS:
        raise AssertionError(f"Expected {EXPECTED_S1_ROWS:,} S1 rows, found {row_count:,}")
    if len(boundaries) != EXPECTED_BATCHES:
        raise AssertionError(f"Expected {EXPECTED_BATCHES:,} batches, found {len(boundaries):,}")
    return {
        "source_file_basename": path.name,
        "source_file_sha256": source_sha256,
        "source_file_bytes": path.stat().st_size,
        "ordered_entity_id_sha256": ordered_digest.hexdigest(),
        "row_count": row_count,
        "country_counts": dict(sorted(country_counts.items())),
        "batch_size": BATCH_SIZE,
        "batch_count": len(boundaries),
        "batch_boundaries": boundaries,
        "validation_seconds": time.perf_counter() - started,
        "validated_at_utc": utc_now(),
    }


def build_launch_contract(
    paths: dict[str, Path],
    input_fingerprint: dict[str, object],
    storage_plan: dict[str, object],
) -> dict[str, object]:
    contract = {
        "contract_version": "task7b-full-training-proxy-v1",
        "scope": "all training Source-1 entities; no ground truth or test data",
        "input_fingerprint": input_fingerprint,
        "model_sha256": file_sha256(paths["task4a_model"]),
        "feature_schema_sha256": file_sha256(paths["task3_schema"]),
        "task5_inference_contract_sha256": file_sha256(paths["task5_contract"]),
        "retrieval_manifest_sha256": file_sha256(paths["task6_retrieval"]),
        "candidate_store_manifest_sha256": file_sha256(paths["task6_store"]),
        "task7a_run_manifest_sha256": file_sha256(paths["task7a_run"]),
        "retrieval_configuration": t6.FROZEN_CONFIGURATION,
        "candidate_cap": t6.FROZEN_CANDIDATE_CAP,
        "threshold": t6.FROZEN_THRESHOLD,
        "feature_count": 66,
        "output_columns": list(OUTPUT_COLUMNS),
        "output_schema_sha256": value_sha256(list(OUTPUT_COLUMNS)),
        "batch_size": BATCH_SIZE,
        "native_workers": NATIVE_WORKERS,
        "retrieval_chunk_size": RETRIEVAL_CHUNK_SIZE,
        "checkpoint_protocol": {
            "batch_commit": "temporary Parquet then os.replace",
            "completion_commit": "SQLite transaction with synchronous=FULL",
            "resume_validation": "size, SHA-256, schema, boundary, pair uniqueness, candidate cap, and decisions",
            "orphan_rule": "Temporary or final files absent from completed_batches are not complete and may be atomically replaced.",
        },
        "storage_preflight": storage_plan,
        "created_at_utc": utc_now(),
    }
    contract["launch_contract_sha256"] = value_sha256(contract)
    return contract


def validate_launch_contract(
    contract: dict[str, object],
    paths: dict[str, Path],
    storage_plan: dict[str, object],
) -> None:
    fingerprint = contract["input_fingerprint"]
    checks = {
        "row_count": int(fingerprint["row_count"]) == EXPECTED_S1_ROWS,
        "batch_count": int(fingerprint["batch_count"]) == EXPECTED_BATCHES,
        "batch_size": int(contract["batch_size"]) == BATCH_SIZE,
        "native_workers": int(contract["native_workers"]) == NATIVE_WORKERS,
        "model": contract["model_sha256"] == file_sha256(paths["task4a_model"]),
        "feature_schema": contract["feature_schema_sha256"] == file_sha256(paths["task3_schema"]),
        "retrieval": contract["retrieval_manifest_sha256"] == file_sha256(paths["task6_retrieval"]),
        "store": contract["candidate_store_manifest_sha256"] == file_sha256(paths["task6_store"]),
        "threshold": float(contract["threshold"]) == 0.95,
        "candidate_cap": int(contract["candidate_cap"]) == 250,
        "feature_count": int(contract["feature_count"]) == 66,
        "output_schema": contract["output_schema_sha256"] == value_sha256(list(OUTPUT_COLUMNS)),
        "source_bytes": int(fingerprint["source_file_bytes"]) == paths["S1"].stat().st_size,
        "source_sha256": fingerprint["source_file_sha256"] == file_sha256(paths["S1"]),
        "disk": bool(storage_plan["preflight_passed"]),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Task 7B launch contract failed: {failed}")
    without_hash = {key: value for key, value in contract.items() if key != "launch_contract_sha256"}
    if contract["launch_contract_sha256"] != value_sha256(without_hash):
        raise AssertionError("Task 7B launch contract self-hash changed")


class PersistentRetriever:
    def __init__(self, task6_output_dir: Path) -> None:
        started = time.perf_counter()
        faiss.omp_set_num_threads(NATIVE_WORKERS)
        self.models: dict[tuple[str, str], t2.TextModel] = {}
        self.indexes: dict[tuple[str, str], object] = {}
        bundle = task6_output_dir / "retrieval_bundle"
        for signal in t6.SIGNALS:
            for partition in t6.PARTITIONS:
                self.models[(partition, signal)] = t6.load_text_model(
                    bundle / f"{signal}.{partition}.model.pkl"
                )
                index = faiss.read_index(str(bundle / f"{signal}.{partition}.faiss"))
                index.nprobe = min(t6.RETRIEVAL_PARAMETERS["nprobe"], index.nlist)
                self.indexes[(partition, signal)] = index
        self.load_seconds = time.perf_counter() - started

    def query(self, eval_df: pd.DataFrame) -> tuple[t3.RetrievalCheckpoint, float]:
        started = time.perf_counter()
        ids: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        for signal in t6.SIGNALS:
            raw_column, transform, _ = t6.SIGNAL_DEFINITIONS[signal]
            signal_ids = np.full((len(eval_df), 250), -1, dtype=np.int64)
            signal_scores = np.full((len(eval_df), 250), -np.inf, dtype=np.float32)
            for partition in t6.PARTITIONS:
                positions = np.flatnonzero(
                    eval_df["normalized_country"].eq(partition).to_numpy()
                )
                if not len(positions):
                    continue
                values = transform(eval_df.loc[positions, raw_column]).fillna("").astype(str)
                nonempty = values.ne("").to_numpy()
                if not bool(nonempty.any()):
                    continue
                vectors = t2.project_texts(
                    self.models[(partition, signal)], values.loc[nonempty].tolist()
                )
                depth = 1_000 if signal.startswith("baseline_") else 250
                distances, labels = self.indexes[(partition, signal)].search(vectors, depth)
                destination = positions[nonempty]
                signal_ids[destination, :] = labels[:, :250]
                signal_scores[destination, :] = distances[:, :250]
                del vectors, distances, labels, values
            ids[signal] = signal_ids
            scores[signal] = signal_scores
        return t3.RetrievalCheckpoint(ids=ids, scores=scores), time.perf_counter() - started


def normalize_s1_chunk(chunk: pd.DataFrame, global_start: int) -> pd.DataFrame:
    result = chunk.rename(columns={"entity_id": "source1_entity_id"}).reset_index(drop=True)
    result["eval_index"] = np.arange(global_start, global_start + len(result), dtype=np.int64)
    result["normalized_name"] = t2.normalize_series(result["business_name"])
    result["normalized_address"] = t2.normalize_series(result["business_address"])
    result["normalized_country"] = t2.normalize_country_series(result["country"])
    result["local_query_index"] = np.arange(len(result), dtype=np.int32)
    return result


def create_checkpoint_database(path: Path, launch_contract: dict[str, object]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    if new:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE completed_batches (
                batch_id TEXT PRIMARY KEY,
                start_entity_offset INTEGER NOT NULL,
                stop_entity_offset INTEGER NOT NULL,
                entity_count INTEGER NOT NULL,
                first_source1_entity_id TEXT NOT NULL,
                last_source1_entity_id TEXT NOT NULL,
                entity_ids_sha256 TEXT NOT NULL,
                pair_rows INTEGER NOT NULL,
                predicted_links INTEGER NOT NULL,
                zero_prediction_entities INTEGER NOT NULL,
                minimum_candidates INTEGER NOT NULL,
                maximum_candidates INTEGER NOT NULL,
                retrieval_seconds_share REAL NOT NULL,
                assembly_seconds_share REAL NOT NULL,
                text_lookup_seconds_share REAL NOT NULL,
                feature_seconds REAL NOT NULL,
                model_seconds REAL NOT NULL,
                write_seconds REAL NOT NULL,
                total_seconds REAL NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                output_schema_sha256 TEXT NOT NULL,
                committed_at_utc TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE retrieval_chunks (
                chunk_id INTEGER PRIMARY KEY,
                start_entity_offset INTEGER NOT NULL,
                stop_entity_offset INTEGER NOT NULL,
                pending_batches INTEGER NOT NULL,
                pair_rows INTEGER NOT NULL,
                retrieval_seconds REAL NOT NULL,
                assembly_seconds REAL NOT NULL,
                text_lookup_seconds REAL NOT NULL,
                peak_rss_mb REAL NOT NULL,
                committed_at_utc TEXT NOT NULL
            );
            """
        )
        metadata = {
            "launch_contract_sha256": launch_contract["launch_contract_sha256"],
            "source_file_sha256": launch_contract["input_fingerprint"]["source_file_sha256"],
            "ordered_entity_id_sha256": launch_contract["input_fingerprint"]["ordered_entity_id_sha256"],
            "model_sha256": launch_contract["model_sha256"],
            "feature_schema_sha256": launch_contract["feature_schema_sha256"],
            "retrieval_manifest_sha256": launch_contract["retrieval_manifest_sha256"],
            "candidate_store_manifest_sha256": launch_contract["candidate_store_manifest_sha256"],
            "threshold": launch_contract["threshold"],
            "batch_size": launch_contract["batch_size"],
            "output_schema_sha256": launch_contract["output_schema_sha256"],
            "created_at_utc": utc_now(),
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
        )
        connection.commit()
    else:
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key,value FROM metadata")
        }
        expected = {
            "launch_contract_sha256": launch_contract["launch_contract_sha256"],
            "source_file_sha256": launch_contract["input_fingerprint"]["source_file_sha256"],
            "ordered_entity_id_sha256": launch_contract["input_fingerprint"]["ordered_entity_id_sha256"],
            "model_sha256": launch_contract["model_sha256"],
            "feature_schema_sha256": launch_contract["feature_schema_sha256"],
            "retrieval_manifest_sha256": launch_contract["retrieval_manifest_sha256"],
            "candidate_store_manifest_sha256": launch_contract["candidate_store_manifest_sha256"],
            "threshold": launch_contract["threshold"],
            "batch_size": launch_contract["batch_size"],
            "output_schema_sha256": launch_contract["output_schema_sha256"],
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise AssertionError(f"Checkpoint database contract drift for {key}")
    return connection


def batch_record_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {key: row[key] for key in row.keys()}


def validate_batch_file(
    path: Path,
    record: dict[str, object],
    boundary: dict[str, object],
) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"Completed batch missing: {path.name}")
    if path.stat().st_size != int(record["bytes"]):
        raise AssertionError(f"Completed batch byte size changed: {path.name}")
    if file_sha256(path) != record["sha256"]:
        raise AssertionError(f"Completed batch SHA-256 changed: {path.name}")
    frame = pq.read_table(path).to_pandas()
    if frame.columns.tolist() != list(OUTPUT_COLUMNS):
        raise AssertionError(f"Completed batch schema changed: {path.name}")
    if len(frame) != int(record["pair_rows"]):
        raise AssertionError(f"Completed batch row count changed: {path.name}")
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError(f"Duplicate S1-candidate pairs: {path.name}")
    if set(frame["batch_id"].astype(str)) != {boundary["batch_id"]}:
        raise AssertionError(f"Batch ID column changed: {path.name}")
    if not np.isfinite(frame["lightgbm_score"].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Non-finite model scores: {path.name}")
    decisions = frame["lightgbm_score"].to_numpy(dtype=np.float64) >= t6.FROZEN_THRESHOLD
    if not np.array_equal(decisions, frame["predicted_match"].to_numpy(dtype=bool)):
        raise AssertionError(f"Threshold decisions changed: {path.name}")
    if not set(frame["candidate_source"].astype(str)) <= {"S2", "S3"}:
        raise AssertionError(f"Unexpected candidate source: {path.name}")
    observed_ids = frame["source1_entity_id"].drop_duplicates().astype(str).tolist()
    if len(observed_ids) != int(boundary["entity_count"]):
        raise AssertionError(f"Missing S1 output rows: {path.name}")
    if ordered_values_sha256(observed_ids) != boundary["entity_ids_sha256"]:
        raise AssertionError(f"S1 ordering changed: {path.name}")
    counts = frame.groupby("source1_entity_id", sort=False).size()
    if int(counts.max()) > t6.FROZEN_CANDIDATE_CAP:
        raise AssertionError(f"Candidate cap exceeded: {path.name}")
    for _, group in frame.groupby("source1_entity_id", sort=False):
        positions = group["candidate_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(1, len(group) + 1)):
            raise AssertionError(f"Candidate position ordering changed: {path.name}")
    predicted_by_s1 = frame.loc[frame["predicted_match"]].groupby("source1_entity_id").size()
    return {
        "pair_rows": len(frame),
        "predicted_links": int(frame["predicted_match"].sum()),
        "zero_prediction_entities": int(boundary["entity_count"] - len(predicted_by_s1)),
        "minimum_candidates": int(counts.min()),
        "maximum_candidates": int(counts.max()),
    }


def load_and_validate_completed(
    connection: sqlite3.Connection,
    checkpoint_dir: Path,
    boundaries: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    connection.row_factory = sqlite3.Row
    completed: dict[str, dict[str, object]] = {}
    rows = connection.execute("SELECT * FROM completed_batches ORDER BY batch_id").fetchall()
    for index, row in enumerate(rows, start=1):
        record = batch_record_from_row(row)
        batch_id = str(record["batch_id"])
        if batch_id not in boundaries:
            raise AssertionError(f"Unknown completed batch: {batch_id}")
        measured = validate_batch_file(
            checkpoint_dir / f"{batch_id}.parquet", record, boundaries[batch_id]
        )
        for key, value in measured.items():
            if int(record[key]) != int(value):
                raise AssertionError(f"Completed batch metadata mismatch for {batch_id}/{key}")
        completed[batch_id] = record
        if index % 1_000 == 0:
            log(f"Resume validation: verified {index:,} completed batches")
    return completed


def preflight_retriever_parity(
    retriever: PersistentRetriever,
    text_store: t6.PersistedTextStore,
    paths: dict[str, Path],
    features: list[str],
    feature_schema: pd.DataFrame,
    model: lgb.Booster,
    args: argparse.Namespace,
) -> dict[str, object]:
    selection = t5.select_smoke_entities(paths["task3_split"], 100)
    eval_df = t5.load_raw_s1(selection, paths["S1"], args.chunksize)
    checkpoint, query_seconds = retriever.query(eval_df)
    indices = eval_df["eval_index"].to_numpy(dtype=np.int64)
    retrieval_rows = t5.compare_retrieval_checkpoint(
        checkpoint, paths["task2_5_checkpoint"], indices
    )
    offsets, counts, metadata, candidate_sets = t5.assemble_candidates(eval_df, checkpoint)
    retrieval_rows.extend(
        t5.compare_frozen_candidates(indices, candidate_sets, paths["task3_candidates"])
    )
    if not all(row["passed"] for row in retrieval_rows):
        raise AssertionError("Task 7B persistent retriever parity failed")
    candidate_text = text_store.lookup(metadata.candidate_ids)
    s1_text = t3.build_s1_text(eval_df)
    frame = t5.build_inference_feature_frame(
        metadata, eval_df, s1_text, candidate_text, NATIVE_WORKERS
    )
    entity_ids = set(eval_df["source1_entity_id"].astype(str))
    reference_features = t5.read_reference_pair_features(paths["task3_pairs"], entity_ids, features)
    reference_scores = t5.read_reference_scores(paths["task4a_predictions"], entity_ids)
    feature_rows, _ = t5.compare_features_and_scores(
        frame, features, feature_schema, model, reference_features, reference_scores
    )
    if not all(row["passed"] for row in feature_rows):
        raise AssertionError("Task 7B feature/model preflight parity failed")
    return {
        "s1_entities": len(eval_df),
        "pair_rows": int(counts.sum()),
        "retrieval_query_seconds": query_seconds,
        "retrieval_checks": len(retrieval_rows),
        "feature_model_checks": len(feature_rows),
        "passed": True,
    }


def commit_batch(
    connection: sqlite3.Connection,
    checkpoint_dir: Path,
    boundary: dict[str, object],
    output: pd.DataFrame,
    timings: dict[str, float],
) -> dict[str, object]:
    batch_id = str(boundary["batch_id"])
    path = checkpoint_dir / f"{batch_id}.parquet"
    t5.write_prediction_batch(path, output)
    counts = output.groupby("source1_entity_id", sort=False).size()
    predicted_by_s1 = output.loc[output["predicted_match"]].groupby("source1_entity_id").size()
    record = {
        **boundary,
        "pair_rows": len(output),
        "predicted_links": int(output["predicted_match"].sum()),
        "zero_prediction_entities": int(boundary["entity_count"] - len(predicted_by_s1)),
        "minimum_candidates": int(counts.min()),
        "maximum_candidates": int(counts.max()),
        "retrieval_seconds_share": timings["retrieval_seconds_share"],
        "assembly_seconds_share": timings["assembly_seconds_share"],
        "text_lookup_seconds_share": timings["text_lookup_seconds_share"],
        "feature_seconds": timings["feature_seconds"],
        "model_seconds": timings["model_seconds"],
        "write_seconds": timings["write_seconds"],
        "total_seconds": sum(timings.values()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "output_schema_sha256": value_sha256(list(OUTPUT_COLUMNS)),
        "committed_at_utc": utc_now(),
    }
    validate_batch_file(path, record, boundary)
    columns = list(record)
    placeholders = ",".join("?" for _ in columns)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        f"INSERT INTO completed_batches ({','.join(columns)}) VALUES ({placeholders})",
        [record[column] for column in columns],
    )
    connection.commit()
    return record


def progress_snapshot(
    connection: sqlite3.Connection,
    total_batches: int,
    started_perf: float,
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(entity_count),0), COALESCE(SUM(pair_rows),0),
               COALESCE(SUM(predicted_links),0), COALESCE(SUM(bytes),0),
               COALESCE(AVG(total_seconds),0), COALESCE(SUM(total_seconds),0)
        FROM completed_batches
        """
    ).fetchone()
    completed_batches = int(row[0])
    elapsed = time.perf_counter() - started_perf
    remaining_batches = total_batches - completed_batches
    average = float(row[5])
    return {
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "completed_s1_entities": int(row[1]),
        "candidate_pair_rows": int(row[2]),
        "predicted_links": int(row[3]),
        "checkpoint_bytes": int(row[4]),
        "average_seconds_per_batch": average,
        "active_batch_seconds": float(row[6]),
        "elapsed_wall_seconds_this_process": elapsed,
        "estimated_remaining_seconds": remaining_batches * average if average else None,
        "peak_rss_mb": t3.peak_rss_mb(),
        "updated_at_utc": utc_now(),
    }


def run_full_inference(
    args: argparse.Namespace,
    paths: dict[str, Path],
    launch_contract: dict[str, object],
    features: list[str],
    model: lgb.Booster,
    retriever: PersistentRetriever,
    candidate_store: t6.PersistedTextStore,
) -> dict[str, object]:
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "checkpoint_manifest.sqlite"
    connection = create_checkpoint_database(database_path, launch_contract)
    boundaries_list = launch_contract["input_fingerprint"]["batch_boundaries"]
    boundaries = {row["batch_id"]: row for row in boundaries_list}
    completed = load_and_validate_completed(connection, checkpoint_dir, boundaries)
    resume_state = {
        "completed_batches": len(completed),
        "completed_s1_entities": sum(int(row["entity_count"]) for row in completed.values()),
        "remaining_batches": len(boundaries) - len(completed),
    }
    log(
        f"Resume state: {resume_state['completed_batches']:,}/{len(boundaries):,} batches; "
        f"{resume_state['completed_s1_entities']:,}/{EXPECTED_S1_ROWS:,} S1 complete"
    )
    if len(completed) == len(boundaries):
        connection.close()
        return {"resume_state": resume_state, "new_batches": 0, "already_complete": True}

    process_started_perf = time.perf_counter()
    process_started_epoch = time.time()
    existing_start = connection.execute(
        "SELECT value FROM metadata WHERE key='inference_first_started_epoch'"
    ).fetchone()
    if existing_start is None:
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            ("inference_first_started_epoch", json.dumps(process_started_epoch)),
        )
        connection.commit()
    progress_path = args.output_dir / "task7b_progress.jsonl"
    reader = pd.read_csv(
        paths["S1"],
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        chunksize=RETRIEVAL_CHUNK_SIZE,
    )
    global_start = 0
    new_batches = 0
    with progress_path.open("a", encoding="utf-8", buffering=1) as progress_handle:
        for chunk_id, raw_chunk in enumerate(reader):
            chunk_stop = global_start + len(raw_chunk)
            chunk_boundaries = [
                row
                for row in boundaries_list
                if int(row["start_entity_offset"]) >= global_start
                and int(row["stop_entity_offset"]) <= chunk_stop
            ]
            pending = [row for row in chunk_boundaries if row["batch_id"] not in completed]
            if not pending:
                global_start = chunk_stop
                continue
            eval_df = normalize_s1_chunk(raw_chunk, global_start)
            for boundary in chunk_boundaries:
                local_start = int(boundary["start_entity_offset"]) - global_start
                local_stop = int(boundary["stop_entity_offset"]) - global_start
                ids = eval_df.iloc[local_start:local_stop]["source1_entity_id"].astype(str).tolist()
                if ordered_values_sha256(ids) != boundary["entity_ids_sha256"]:
                    raise AssertionError(f"Streaming S1 boundary changed: {boundary['batch_id']}")

            checkpoint, retrieval_seconds = retriever.query(eval_df)
            assembly_started = time.perf_counter()
            offsets, counts, metadata, _ = t5.assemble_candidates(eval_df, checkpoint)
            assembly_seconds = time.perf_counter() - assembly_started
            required_parts = []
            for boundary in pending:
                local_start = int(boundary["start_entity_offset"]) - global_start
                local_stop = int(boundary["stop_entity_offset"]) - global_start
                pair_start = int(offsets[local_start])
                pair_stop = int(offsets[local_stop])
                required_parts.append(metadata.candidate_ids[pair_start:pair_stop])
            lookup_started = time.perf_counter()
            text_store = candidate_store.lookup(np.concatenate(required_parts))
            lookup_seconds = time.perf_counter() - lookup_started
            s1_text = t3.build_s1_text(eval_df)
            share_count = len(pending)
            chunk_pair_rows = 0

            for boundary in pending:
                batch_id = str(boundary["batch_id"])
                local_start = int(boundary["start_entity_offset"]) - global_start
                local_stop = int(boundary["stop_entity_offset"]) - global_start
                batch_metadata = t5.metadata_for_query_range(
                    metadata, offsets, local_start, local_stop
                )
                feature_started = time.perf_counter()
                frame = t5.build_inference_feature_frame(
                    batch_metadata, eval_df, s1_text, text_store, NATIVE_WORKERS
                )
                feature_seconds = time.perf_counter() - feature_started
                matrix = frame[features].to_numpy(dtype=np.float32, copy=True)
                if matrix.shape[1] != 66 or not np.isfinite(matrix).all():
                    raise AssertionError(f"Invalid feature matrix: {batch_id}")
                model_started = time.perf_counter()
                scores = model.predict(matrix).astype(np.float64)
                model_seconds = time.perf_counter() - model_started
                output = pd.DataFrame(
                    {
                        "source1_entity_id": frame["source1_entity_id"].astype(str),
                        "candidate_entity_id": frame["candidate_entity_id"].astype(str),
                        "candidate_source": frame["candidate_source"].astype(str),
                        "lightgbm_score": scores,
                        "predicted_match": scores >= t6.FROZEN_THRESHOLD,
                        "candidate_position": frame["frozen_candidate_position"].to_numpy(dtype=np.uint16),
                        "batch_id": batch_id,
                    }
                )
                write_started = time.perf_counter()
                record = commit_batch(
                    connection,
                    checkpoint_dir,
                    boundary,
                    output,
                    {
                        "retrieval_seconds_share": retrieval_seconds / share_count,
                        "assembly_seconds_share": assembly_seconds / share_count,
                        "text_lookup_seconds_share": lookup_seconds / share_count,
                        "feature_seconds": feature_seconds,
                        "model_seconds": model_seconds,
                        "write_seconds": 0.0,
                    },
                )
                actual_write_seconds = time.perf_counter() - write_started
                connection.execute(
                    "UPDATE completed_batches SET write_seconds=?, total_seconds=total_seconds+? WHERE batch_id=?",
                    (actual_write_seconds, actual_write_seconds, batch_id),
                )
                connection.commit()
                record["write_seconds"] = actual_write_seconds
                record["total_seconds"] += actual_write_seconds
                completed[batch_id] = record
                new_batches += 1
                chunk_pair_rows += len(output)
                snapshot = progress_snapshot(connection, len(boundaries), process_started_perf)
                snapshot["recent_batch_id"] = batch_id
                snapshot["recent_batch_runtime_seconds"] = record["total_seconds"]
                progress_handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
                if new_batches == 1 or new_batches % args.progress_every_batches == 0:
                    eta = snapshot["estimated_remaining_seconds"]
                    eta_text = "unknown" if eta is None else f"{eta / 3600.0:.2f}h"
                    log(
                        f"Progress {snapshot['completed_batches']:,}/{len(boundaries):,} batches; "
                        f"S1={snapshot['completed_s1_entities']:,}; pairs={snapshot['candidate_pair_rows']:,}; "
                        f"recent={record['total_seconds']:.2f}s; ETA={eta_text}; "
                        f"peak={snapshot['peak_rss_mb']:.1f} MB"
                    )
                del frame, matrix, scores, output, batch_metadata
                gc.collect()

            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR REPLACE INTO retrieval_chunks
                (chunk_id,start_entity_offset,stop_entity_offset,pending_batches,pair_rows,
                 retrieval_seconds,assembly_seconds,text_lookup_seconds,peak_rss_mb,committed_at_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk_id,
                    global_start,
                    chunk_stop,
                    len(pending),
                    chunk_pair_rows,
                    retrieval_seconds,
                    assembly_seconds,
                    lookup_seconds,
                    t3.peak_rss_mb(),
                    utc_now(),
                ),
            )
            connection.commit()
            global_start = chunk_stop
            del raw_chunk, eval_df, checkpoint, offsets, counts, metadata, text_store, s1_text
            gc.collect()

    if global_start != EXPECTED_S1_ROWS:
        raise AssertionError(f"Streaming reader ended at {global_start:,} S1 rows")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        ("inference_last_completed_epoch", json.dumps(time.time())),
    )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.commit()
    result = {
        "resume_state": resume_state,
        "new_batches": new_batches,
        "already_complete": False,
        "process_wall_seconds": time.perf_counter() - process_started_perf,
    }
    connection.close()
    return result


def final_validation(
    args: argparse.Namespace,
    paths: dict[str, Path],
    launch_contract: dict[str, object],
    hashes_before: dict[str, str],
    run_result: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    checkpoint_dir = args.output_dir / "checkpoints"
    database_path = args.output_dir / "checkpoint_manifest.sqlite"
    connection = create_checkpoint_database(database_path, launch_contract)
    boundaries_list = launch_contract["input_fingerprint"]["batch_boundaries"]
    boundaries = {row["batch_id"]: row for row in boundaries_list}
    validation_started = time.perf_counter()
    completed = load_and_validate_completed(connection, checkpoint_dir, boundaries)
    if len(completed) != EXPECTED_BATCHES:
        raise AssertionError(f"Only {len(completed):,}/{EXPECTED_BATCHES:,} batches complete")
    rows = pd.read_sql_query(
        "SELECT * FROM completed_batches ORDER BY batch_id", connection
    )
    chunks = pd.read_sql_query(
        "SELECT * FROM retrieval_chunks ORDER BY chunk_id", connection
    )
    metadata = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key,value FROM metadata")
    }
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    hashes_after = upstream_hashes(paths)
    source_hash_after = file_sha256(paths["S1"])
    no_op = run_result["new_batches"] == 0 if run_result["already_complete"] else True
    totals = {
        "s1_entities_processed": int(rows["entity_count"].sum()),
        "completed_batches": len(rows),
        "candidate_pair_rows": int(rows["pair_rows"].sum()),
        "predicted_links": int(rows["predicted_links"].sum()),
        "zero_prediction_s1_entities": int(rows["zero_prediction_entities"].sum()),
        "minimum_candidates_per_s1": int(rows["minimum_candidates"].min()),
        "maximum_candidates_per_s1": int(rows["maximum_candidates"].max()),
        "average_candidates_per_s1": float(rows["pair_rows"].sum() / rows["entity_count"].sum()),
        "checkpoint_bytes": int(rows["bytes"].sum()),
        "peak_rss_mb": float(chunks["peak_rss_mb"].max()) if len(chunks) else t3.peak_rss_mb(),
        "active_batch_seconds": float(rows["total_seconds"].sum()),
        "retrieval_seconds": float(chunks["retrieval_seconds"].sum()),
        "assembly_seconds": float(chunks["assembly_seconds"].sum()),
        "candidate_text_lookup_seconds": float(chunks["text_lookup_seconds"].sum()),
        "feature_seconds": float(rows["feature_seconds"].sum()),
        "model_seconds": float(rows["model_seconds"].sum()),
        "checkpoint_write_seconds": float(rows["write_seconds"].sum()),
        "validation_seconds": time.perf_counter() - validation_started,
        "inference_wall_seconds": float(
            metadata["inference_last_completed_epoch"] - metadata["inference_first_started_epoch"]
        ),
    }
    projected_seconds = float(
        json.loads(paths["task6_scale"].read_text(encoding="utf-8"))["projected_wall_seconds_linear"]
    )
    totals["projected_wall_seconds"] = projected_seconds
    totals["actual_to_projected_ratio"] = totals["inference_wall_seconds"] / projected_seconds

    integrity_rows = [
        t7a.integrity_row("exact_s1_count", totals["s1_entities_processed"] == EXPECTED_S1_ROWS, str(totals["s1_entities_processed"])),
        t7a.integrity_row("all_batches_complete", totals["completed_batches"] == EXPECTED_BATCHES, str(totals["completed_batches"])),
        t7a.integrity_row("deterministic_boundaries", rows["batch_id"].tolist() == [row["batch_id"] for row in boundaries_list], f"batches={len(rows)}"),
        t7a.integrity_row("no_missing_s1", int(rows["entity_count"].sum()) == EXPECTED_S1_ROWS, "All contract boundaries completed."),
        t7a.integrity_row("candidate_cap", totals["maximum_candidates_per_s1"] <= 250, str(totals["maximum_candidates_per_s1"])),
        t7a.integrity_row("candidate_counts_positive", totals["minimum_candidates_per_s1"] > 0, str(totals["minimum_candidates_per_s1"])),
        t7a.integrity_row("checkpoint_hash_size_schema", len(completed) == EXPECTED_BATCHES, "Every batch fully revalidated."),
        t7a.integrity_row("threshold_decisions", True, "Every file checked score >= 0.95 during final validation."),
        t7a.integrity_row("duplicate_pairs", True, "Every file checked; disjoint S1 boundaries prevent cross-batch duplicates."),
        t7a.integrity_row("source_sha256_unchanged", source_hash_after == launch_contract["input_fingerprint"]["source_file_sha256"], source_hash_after),
        t7a.integrity_row("source_bytes_unchanged", paths["S1"].stat().st_size == int(launch_contract["input_fingerprint"]["source_file_bytes"]), str(paths["S1"].stat().st_size)),
        t7a.integrity_row("source_row_count", int(launch_contract["input_fingerprint"]["row_count"]) == EXPECTED_S1_ROWS, str(EXPECTED_S1_ROWS)),
        t7a.integrity_row("ordered_entity_id_hash_bound", bool(launch_contract["input_fingerprint"]["ordered_entity_id_sha256"]), launch_contract["input_fingerprint"]["ordered_entity_id_sha256"]),
        t7a.integrity_row("frozen_upstream_hashes", hashes_before == hashes_after, "Task 2.5 through Task 7A artifacts unchanged."),
        t7a.integrity_row("model_hash", launch_contract["model_sha256"] == file_sha256(paths["task4a_model"]), launch_contract["model_sha256"]),
        t7a.integrity_row("feature_schema_hash", launch_contract["feature_schema_sha256"] == file_sha256(paths["task3_schema"]), launch_contract["feature_schema_sha256"]),
        t7a.integrity_row("retrieval_manifest_hash", launch_contract["retrieval_manifest_sha256"] == file_sha256(paths["task6_retrieval"]), launch_contract["retrieval_manifest_sha256"]),
        t7a.integrity_row("store_manifest_hash", launch_contract["candidate_store_manifest_sha256"] == file_sha256(paths["task6_store"]), launch_contract["candidate_store_manifest_sha256"]),
        t7a.integrity_row("threshold_frozen", float(launch_contract["threshold"]) == 0.95, "score >= 0.95"),
        t7a.integrity_row("feature_count_frozen", int(launch_contract["feature_count"]) == 66, "66"),
        t7a.integrity_row("batch_size_frozen", int(launch_contract["batch_size"]) == 100, "100"),
        t7a.integrity_row("native_workers_frozen", int(launch_contract["native_workers"]) == 1, "1"),
        t7a.integrity_row("output_schema_frozen", launch_contract["output_schema_sha256"] == value_sha256(list(OUTPUT_COLUMNS)), launch_contract["output_schema_sha256"]),
        t7a.integrity_row("resume_contract", True, "SQLite completion records and every completed file validated before skip."),
        t7a.integrity_row("completed_rerun_no_op", no_op, str(run_result)),
        t7a.integrity_row("no_ground_truth", "GT" not in paths, "Ground truth not resolved."),
        t7a.integrity_row("no_test_data", all("test" not in paths[key].name.casefold() for key in ("S1", "S2", "S3")), "Training sources only."),
        t7a.integrity_row("no_submission", True, "Internal checkpoint schema only."),
    ]
    integrity = pd.DataFrame(integrity_rows)
    if not integrity["passed"].all():
        raise AssertionError(
            f"Task 7B final integrity failed: {integrity.loc[~integrity['passed']].to_dict('records')}"
        )
    runtime = rows[
        [
            "batch_id",
            "start_entity_offset",
            "stop_entity_offset",
            "entity_count",
            "pair_rows",
            "predicted_links",
            "retrieval_seconds_share",
            "assembly_seconds_share",
            "text_lookup_seconds_share",
            "feature_seconds",
            "model_seconds",
            "write_seconds",
            "total_seconds",
            "bytes",
            "committed_at_utc",
        ]
    ].copy()
    return totals, runtime, integrity


def write_summary(
    path: Path,
    totals: dict[str, object],
    launch_contract: dict[str, object],
    integrity: pd.DataFrame,
) -> None:
    text = f"""# Task 7B Full-Scale Training-Proxy Inference

Status: **PASS**

## Results

- S1 entities processed: {totals['s1_entities_processed']:,}.
- Candidate-pair rows: {totals['candidate_pair_rows']:,}.
- Predicted links at score >= 0.95: {totals['predicted_links']:,}.
- Zero-prediction S1 entities: {totals['zero_prediction_s1_entities']:,}.
- Average candidates per S1: {totals['average_candidates_per_s1']:.3f}.
- Minimum / maximum candidates per S1: {totals['minimum_candidates_per_s1']} / {totals['maximum_candidates_per_s1']}.
- Checkpoint batches: {totals['completed_batches']:,}; size: {totals['checkpoint_bytes'] / 2**30:.2f} GiB.

## Runtime

- Actual inference wall time: {totals['inference_wall_seconds'] / 3600:.2f} hours.
- Task 7A projected runtime: {totals['projected_wall_seconds'] / 3600:.2f} hours.
- Actual / projected ratio: {totals['actual_to_projected_ratio']:.3f}.
- Retrieval: {totals['retrieval_seconds'] / 3600:.2f}h; candidate text lookup: {totals['candidate_text_lookup_seconds'] / 3600:.2f}h.
- Feature generation: {totals['feature_seconds'] / 3600:.2f}h; LightGBM: {totals['model_seconds'] / 3600:.2f}h; checkpoint writes: {totals['checkpoint_write_seconds'] / 3600:.2f}h.
- Peak RSS: {totals['peak_rss_mb']:.1f} MB.

## Contracts

- Source SHA-256: `{launch_contract['input_fingerprint']['source_file_sha256']}`.
- Ordered entity-ID SHA-256: `{launch_contract['input_fingerprint']['ordered_entity_id_sha256']}`.
- Source bytes / rows: {launch_contract['input_fingerprint']['source_file_bytes']:,} / {launch_contract['input_fingerprint']['row_count']:,}.
- Batch size / native workers: {launch_contract['batch_size']} / {launch_contract['native_workers']}.
- Checkpoint/resume: PASS. Every completed batch was atomically committed and fully revalidated; completed rerun was a no-op.
- Frozen artifact integrity: PASS. {len(integrity)}/{len(integrity)} checks passed.

## Scope

This was full-scale inference over training Source-1 records as a production proxy. Ground truth, competition test data, model training, threshold tuning, retrieval changes, test predictions, submission transformation, and submission generation were not used or performed.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def prepare_or_load_contract(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> tuple[dict[str, object], dict[str, object], list[str], pd.DataFrame, lgb.Booster]:
    task6_run = json.loads(paths["task6_run"].read_text(encoding="utf-8"))
    task7a_run = json.loads(paths["task7a_run"].read_text(encoding="utf-8"))
    if task6_run["status"] != "PASS" or task7a_run["status"] != "PASS":
        raise AssertionError("Task 6 and Task 7A must both be PASS")
    if not pd.read_csv(paths["task6_integrity"])["passed"].all():
        raise AssertionError("Task 6 integrity report is not PASS")
    if not pd.read_csv(paths["task7a_integrity"])["passed"].all():
        raise AssertionError("Task 7A integrity report is not PASS")
    scale = json.loads(paths["task6_scale"].read_text(encoding="utf-8"))
    storage_plan = t7a.build_storage_plan(scale, args.output_dir)
    t7a.enforce_disk_preflight(storage_plan)
    retrieval_manifest = json.loads(paths["task6_retrieval"].read_text(encoding="utf-8"))
    store_manifest = json.loads(paths["task6_store"].read_text(encoding="utf-8"))
    t6.verify_manifest_artifacts(retrieval_manifest, args.task6_output_dir, "retrieval_bundle")
    t6.verify_manifest_artifacts(store_manifest, args.task6_output_dir, "candidate_store")
    features, feature_schema, model = t6.validate_frozen_contract(paths, args)
    if len(features) != 66 or model.feature_name() != features:
        raise AssertionError("Frozen 66-feature model contract changed")

    contract_path = args.output_dir / "task7b_launch_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    else:
        input_contract = json.loads(paths["task7a_contract"].read_text(encoding="utf-8"))
        fingerprint = scan_full_input(paths["S1"], input_contract, args.chunksize)
        contract = build_launch_contract(paths, fingerprint, storage_plan)
        atomic_json(contract_path, contract)
        atomic_json(args.output_dir / "task7b_preflight_results.json", {
            "status": "PASS",
            "input_fingerprint": fingerprint,
            "storage_plan": storage_plan,
            "frozen_artifacts_verified": True,
            "created_at_utc": utc_now(),
        })
    validate_launch_contract(contract, paths, storage_plan)
    return contract, storage_plan, features, feature_schema, model


def print_preflight_summary(
    contract: dict[str, object],
    storage: dict[str, object],
    scale: dict[str, object],
    checkpoint_database: Path,
) -> None:
    completed = 0
    if checkpoint_database.exists():
        connection = sqlite3.connect(checkpoint_database)
        completed = int(connection.execute("SELECT COUNT(*) FROM completed_batches").fetchone()[0])
        connection.close()
    fingerprint = contract["input_fingerprint"]
    lines = [
        "Task 7B preflight PASS",
        f"S1 rows: {fingerprint['row_count']:,}",
        f"Projected candidate pairs: {int(scale['projected_candidate_pairs']):,}",
        f"Projected runtime: {float(scale['projected_wall_hours_linear']):.2f} hours",
        f"Free disk: {storage['current_filesystem_free_bytes'] / 2**30:.2f} GiB",
        f"Required free disk: {storage['recommended_minimum_free_bytes_before_launch'] / 2**30:.2f} GiB",
        f"Batch size: {BATCH_SIZE}",
        f"Number of batches: {EXPECTED_BATCHES:,}",
        f"Native workers: {NATIVE_WORKERS}",
        f"Source SHA-256: {fingerprint['source_file_sha256']}",
        f"Ordered ID SHA-256: {fingerprint['ordered_entity_id_sha256']}",
        f"Checkpoint directory: {checkpoint_database.parent / 'checkpoints'}",
        f"Resume state: {completed:,}/{EXPECTED_BATCHES:,} completed batches",
    ]
    print("\n".join(lines), flush=True)


def main() -> None:
    args = make_parser().parse_args()
    if BATCH_SIZE != 100 or NATIVE_WORKERS != 1:
        raise AssertionError("Task 7B launch configuration changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(args)
    hashes_before = upstream_hashes(paths)
    contract, storage, features, feature_schema, model = prepare_or_load_contract(args, paths)
    scale = json.loads(paths["task6_scale"].read_text(encoding="utf-8"))
    database_path = args.output_dir / "checkpoint_manifest.sqlite"
    print_preflight_summary(contract, storage, scale, database_path)
    if args.mode == "preflight-only":
        retriever = PersistentRetriever(args.task6_output_dir)
        candidate_store = t6.PersistedTextStore(
            args.task6_output_dir / "candidate_text_store",
            json.loads(paths["task6_store"].read_text(encoding="utf-8")),
        )
        parity = preflight_retriever_parity(
            retriever, candidate_store, paths, features, feature_schema, model, args
        )
        preflight_path = args.output_dir / "task7b_preflight_results.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["orchestration_parity"] = parity
        preflight["retrieval_bundle_load_seconds"] = retriever.load_seconds
        preflight["peak_rss_mb"] = t3.peak_rss_mb()
        atomic_json(preflight_path, preflight)
        log("Task 7B preflight-only PASS; full inference has not started")
        return

    retriever = PersistentRetriever(args.task6_output_dir)
    candidate_store = t6.PersistedTextStore(
        args.task6_output_dir / "candidate_text_store",
        json.loads(paths["task6_store"].read_text(encoding="utf-8")),
    )
    preflight_path = args.output_dir / "task7b_preflight_results.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if "orchestration_parity" not in preflight:
        preflight["orchestration_parity"] = preflight_retriever_parity(
            retriever, candidate_store, paths, features, feature_schema, model, args
        )
        preflight["retrieval_bundle_load_seconds"] = retriever.load_seconds
        preflight["peak_rss_mb"] = t3.peak_rss_mb()
        atomic_json(preflight_path, preflight)
    if not preflight["orchestration_parity"]["passed"]:
        raise AssertionError("Task 7B prelaunch orchestration parity is not PASS")

    if args.mode == "validate-only":
        run_result = {"new_batches": 0, "already_complete": True, "validation_only": True}
    else:
        run_result = run_full_inference(
            args, paths, contract, features, model, retriever, candidate_store
        )
    database_exists = database_path.exists()
    if not database_exists:
        if args.mode == "validate-only":
            raise FileNotFoundError("No Task 7B checkpoint database to validate")
        raise AssertionError("Task 7B checkpoint database was not created")

    connection = sqlite3.connect(database_path)
    completed_count = int(connection.execute("SELECT COUNT(*) FROM completed_batches").fetchone()[0])
    connection.close()
    if completed_count < EXPECTED_BATCHES:
        log(f"Task 7B paused safely at {completed_count:,}/{EXPECTED_BATCHES:,} batches")
        return

    totals, runtime, integrity = final_validation(
        args, paths, contract, hashes_before, run_result
    )
    no_op_connection = create_checkpoint_database(database_path, contract)
    boundaries = {
        row["batch_id"]: row for row in contract["input_fingerprint"]["batch_boundaries"]
    }
    no_op_completed = load_and_validate_completed(
        no_op_connection, args.output_dir / "checkpoints", boundaries
    )
    no_op_connection.close()
    if len(no_op_completed) != EXPECTED_BATCHES:
        raise AssertionError("Completed no-op rerun verification failed")
    integrity.loc[integrity["check"].eq("completed_rerun_no_op"), "passed"] = True
    integrity.loc[integrity["check"].eq("completed_rerun_no_op"), "detail"] = (
        f"Validated {len(no_op_completed):,} completed batches; zero inference batches required."
    )
    if not integrity["passed"].all():
        raise AssertionError("Task 7B integrity changed after no-op verification")

    atomic_csv(args.output_dir / "task7b_runtime_breakdown.csv", runtime)
    atomic_csv(args.output_dir / "task7b_integrity_report.csv", integrity)
    summary_path = args.output_dir / "task7b_full_inference_summary.md"
    write_summary(summary_path, totals, contract, integrity)
    hashes_after = upstream_hashes(paths)
    manifest = {
        "task": TASK_NAME,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "scope": "full-scale training-proxy inference",
        "launch_contract_sha256": contract["launch_contract_sha256"],
        "input_fingerprint": contract["input_fingerprint"],
        "totals": totals,
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "checkpoint_manifest": {
            "path": str(database_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(database_path),
            "bytes": database_path.stat().st_size,
            "completed_batches": EXPECTED_BATCHES,
        },
        "reports": {
            "runtime": {
                "path": "task7b_runtime_breakdown.csv",
                "sha256": file_sha256(args.output_dir / "task7b_runtime_breakdown.csv"),
            },
            "integrity": {
                "path": "task7b_integrity_report.csv",
                "sha256": file_sha256(args.output_dir / "task7b_integrity_report.csv"),
            },
            "summary": {
                "path": "task7b_full_inference_summary.md",
                "sha256": file_sha256(summary_path),
            },
        },
        "boundaries": {
            "ground_truth_used": False,
            "test_data_used": False,
            "test_predictions_generated": False,
            "submission_generated": False,
            "model_retrained_or_tuned": False,
            "retrieval_or_features_changed": False,
        },
    }
    atomic_json(args.output_dir / "task7b_run_manifest.json", manifest)
    log("Task 7B PASS: full-scale training-proxy inference complete")


if __name__ == "__main__":
    main()
