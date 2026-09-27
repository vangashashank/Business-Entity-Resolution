#!/usr/bin/env python3
"""Task 6: persist and reload the frozen production retrieval and text stores.

This task is deliberately inference-only. It uses training source records to prove
that persisted fitted retrieval state and selective disk text lookup reproduce the
frozen Task 2.5, Task 3, Task 4A, and Task 4B contracts. It never reads ground truth,
test data, labels, or split metadata as inference features.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


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

import task2_candidate_generation as t2
import task2_5_candidate_selection as t25
import task3_pairwise_features as t3
import task5_full_inference_pipeline as t5


TASK_NAME = "Task 6"
FROZEN_CONFIGURATION = t25.COMBINED_AUGMENTED_250
FROZEN_CANDIDATE_CAP = 250
FROZEN_THRESHOLD = 0.95
SIGNALS = tuple(t3.SIGNALS)
PARTITIONS = ("india", "us")
SHARD_COUNT = 64
RETRIEVAL_PARAMETERS = {
    "sample_modulus": 30,
    "sample_per_source_country": 50_000,
    "max_features": 32_768,
    "projection_dim": 256,
    "nlist": 2_048,
    "pq_m": 64,
    "nprobe": 64,
    "baseline_query_depth": 1_000,
    "persisted_result_depth": 250,
    "variant_query_depth": 250,
    "random_state": 42,
    "country_blocking": True,
}
SIGNAL_DEFINITIONS = {
    "baseline_name": ("business_name", t2.normalize_series, "name"),
    "baseline_address": ("business_address", t2.normalize_series, "address"),
    "transliterated_name": (
        "business_name",
        t25.transliterate_series,
        "transliterated_name",
    ),
    "number_address": (
        "business_address",
        t25.address_number_series,
        "number_address",
    ),
    "suffix_name": ("business_name", t25.strip_legal_suffix_series, "suffix_name"),
}
STORE_TEXT_FIELDS = (
    "raw_name",
    "raw_address",
    "normalized_name",
    "transliterated_name",
    "suffix_name",
    "normalized_address",
    "number_sequence",
    "normalized_country",
)


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


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def artifact_record(
    path: Path,
    output_dir: Path,
    artifact_type: str,
    signal: str = "",
    partition: str = "",
    parameters: dict[str, object] | None = None,
    upstream_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "path": str(path.relative_to(output_dir)),
        "type": artifact_type,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "created_at_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "signal": signal,
        "partition": partition,
        "configuration": FROZEN_CONFIGURATION,
        "parameters": parameters or {},
        "code_and_upstream_hashes": upstream_hashes or {},
    }


class RuntimeRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, phase: str, value: float, category: str = "", detail: str = "") -> float:
        known_categories = {"build", "reload-only", "validation-reference"}
        if category in known_categories:
            seconds = time.perf_counter() - value
        else:
            seconds = float(value)
            detail = category if category and not detail else detail
            category = "reload-only"
        row = {
            "category": category,
            "phase": phase,
            "runtime_seconds": seconds,
            "current_rss_mb": t3.current_rss_mb(),
            "process_peak_rss_mb": t3.peak_rss_mb(),
            "detail": detail,
        }
        self.rows.append(row)
        log(
            f"{phase}: {seconds:.2f}s; current RSS={row['current_rss_mb']:.1f} MB; "
            f"peak={row['process_peak_rss_mb']:.1f} MB"
        )
        return seconds


@dataclass
class PersistedTextStore:
    store_dir: Path
    manifest: dict[str, object]

    def lookup(self, encoded_ids: np.ndarray) -> t3.TextStore:
        required = np.unique(np.asarray(encoded_ids, dtype=np.int64))
        if not len(required):
            raise ValueError("Candidate text lookup requires at least one ID")
        size = len(required)
        values = {
            name: np.full(size, None, dtype=object) for name in STORE_TEXT_FIELDS
        }
        name_non_ascii = np.zeros(size, dtype=bool)
        found = np.zeros(size, dtype=bool)
        for shard_id in np.unique(required % SHARD_COUNT):
            positions = np.flatnonzero(required % SHARD_COUNT == shard_id)
            ids = required[positions]
            path = self.store_dir / f"candidate_text_{int(shard_id):02d}.sqlite"
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                for start in range(0, len(ids), 800):
                    batch = ids[start : start + 800]
                    placeholders = ",".join("?" for _ in batch)
                    query = (
                        "SELECT encoded_id, raw_name, raw_address, normalized_name, "
                        "transliterated_name, suffix_name, normalized_address, "
                        "number_sequence, normalized_country, name_non_ascii "
                        f"FROM candidates WHERE encoded_id IN ({placeholders})"
                    )
                    rows = connection.execute(query, [int(value) for value in batch]).fetchall()
                    for row in rows:
                        encoded_id = int(row[0])
                        target = int(np.searchsorted(required, encoded_id))
                        if target >= size or int(required[target]) != encoded_id:
                            raise AssertionError("Candidate store returned an unrequested ID")
                        if found[target]:
                            raise AssertionError(f"Duplicate candidate store ID: {encoded_id}")
                        for column, value in zip(STORE_TEXT_FIELDS, row[1:9]):
                            values[column][target] = str(value)
                        name_non_ascii[target] = bool(row[9])
                        found[target] = True
            finally:
                connection.close()
        if not bool(found.all()):
            raise KeyError(f"Missing candidate text IDs: {required[~found][:10].tolist()}")
        return t3.TextStore(ids=required, name_non_ascii=name_non_ascii, **values)

    def lookup_sources(self, encoded_ids: np.ndarray) -> dict[int, str]:
        required = np.unique(np.asarray(encoded_ids, dtype=np.int64))
        result: dict[int, str] = {}
        for shard_id in np.unique(required % SHARD_COUNT):
            ids = required[required % SHARD_COUNT == shard_id]
            path = self.store_dir / f"candidate_text_{int(shard_id):02d}.sqlite"
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                for start in range(0, len(ids), 800):
                    batch = ids[start : start + 800]
                    placeholders = ",".join("?" for _ in batch)
                    rows = connection.execute(
                        f"SELECT encoded_id, source FROM candidates WHERE encoded_id IN ({placeholders})",
                        [int(value) for value in batch],
                    ).fetchall()
                    for encoded_id, source in rows:
                        result[int(encoded_id)] = str(source)
            finally:
                connection.close()
        if len(result) != len(required):
            raise KeyError("Candidate source lookup did not return every requested ID")
        return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("build-validate", "validate-only", "full"), default="build-validate"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource" / "dataset" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task6_outputs",
    )
    parser.add_argument("--smoke-entities", type=int, default=1_000)
    parser.add_argument("--parity-entities", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--transform-batch-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = t5.resolve_paths(args)
    base.update(
        {
            "task5_contract": PROJECT_ROOT
            / "outputs"
            / "task5_outputs"
            / "task5_inference_contract.json",
            "task5_summary": PROJECT_ROOT
            / "outputs"
            / "task5_outputs"
            / "task5_summary.md",
            "task5_parity": PROJECT_ROOT
            / "outputs"
            / "task5_outputs"
            / "task5_parity_report.csv",
            "task5_runtime": PROJECT_ROOT
            / "outputs"
            / "task5_outputs"
            / "task5_runtime_breakdown.csv",
            "task5_scale": PROJECT_ROOT
            / "outputs"
            / "task5_outputs"
            / "task5_scale_estimate.json",
            "task5_code": PROJECT_ROOT / "src" / "task5_full_inference_pipeline.py",
            "task6_code": Path(__file__).resolve(),
        }
    )
    missing = [str(path) for path in base.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task 6 inputs: {missing}")
    forbidden = [path for key, path in base.items() if key in {"S1", "S2", "S3"} and "test" in path.name.casefold()]
    if forbidden:
        raise ValueError(f"Task 6 cannot access test data: {forbidden}")
    return base


def task6_upstream_hashes(paths: dict[str, Path]) -> dict[str, str]:
    hashes = t5.frozen_hashes(paths)
    for key in ("S1", "S2", "S3", "task5_contract", "task5_summary", "task5_parity", "task5_runtime", "task5_scale", "task5_code", "task6_code"):
        hashes[key] = file_sha256(paths[key])
    return hashes


def validate_frozen_contract(paths: dict[str, Path], args: argparse.Namespace) -> tuple[list[str], pd.DataFrame, lgb.Booster]:
    features, schema, model, decision, policy = t5.load_frozen_contract_inputs(paths)
    contract = json.loads(paths["task5_contract"].read_text(encoding="utf-8"))
    candidate_contract = contract.get("candidate_generator", {})
    if candidate_contract.get("configuration") != FROZEN_CONFIGURATION:
        raise AssertionError("Task 5 candidate-generation contract changed")
    if int(candidate_contract["candidate_cap"]) != FROZEN_CANDIDATE_CAP:
        raise AssertionError("Task 5 candidate cap changed")
    if float(policy["parameters"]["probability_threshold"]) != FROZEN_THRESHOLD:
        raise AssertionError("Task 4B threshold changed")
    if tuple(signal["name"] for signal in candidate_contract["signals"]) != SIGNALS:
        raise AssertionError("Task 5 retrieval signal order changed")
    if model.num_feature() != 66 or len(features) != 66:
        raise AssertionError("Frozen ordered feature contract is not 66 features")
    return features, schema, model


def text_model_payload(model: t2.TextModel) -> dict[str, object]:
    return {
        "vectorizer": model.vectorizer,
        "projector": model.projector,
        "vocabulary_size": model.vocabulary_size,
    }


def load_text_model(path: Path) -> t2.TextModel:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return t2.TextModel(
        vectorizer=payload["vectorizer"],
        projector=payload["projector"],
        train_texts=[],
        vocabulary_size=int(payload["vocabulary_size"]),
    )


def fit_frozen_models(
    sample: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], t2.TextModel], list[str]]:
    baseline, partitions = t2.build_text_models(
        sample,
        eval_df,
        True,
        RETRIEVAL_PARAMETERS["max_features"],
        RETRIEVAL_PARAMETERS["projection_dim"],
    )
    specs = [
        t25.SignalSpec("transliterated_name", "business_name", t25.transliterate_series),
        t25.SignalSpec("suffix_name", "business_name", t25.strip_legal_suffix_series),
        t25.SignalSpec("number_address", "business_address", t25.address_number_series),
    ]
    variants, variant_partitions = t25.build_variant_models(
        sample,
        eval_df,
        specs,
        RETRIEVAL_PARAMETERS["max_features"],
        RETRIEVAL_PARAMETERS["projection_dim"],
    )
    if partitions != variant_partitions or tuple(partitions) != PARTITIONS:
        raise AssertionError(f"Unexpected retrieval partitions: {partitions}")
    models: dict[tuple[str, str], t2.TextModel] = {}
    for partition in partitions:
        models[(partition, "baseline_name")] = baseline[(partition, "name")]
        models[(partition, "baseline_address")] = baseline[(partition, "address")]
        for signal in ("transliterated_name", "number_address", "suffix_name"):
            models[(partition, signal)] = variants[(partition, signal)]
    return models, partitions


def build_one_signal_indexes(
    signal: str,
    paths: dict[str, Path],
    models: dict[tuple[str, str], t2.TextModel],
    partitions: list[str],
    bundle_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[Path], dict[str, int]]:
    raw_column, transform, _ = SIGNAL_DEFINITIONS[signal]
    indexes: dict[str, faiss.IndexIVFPQ] = {}
    for partition in partitions:
        index, _ = t2.create_faiss_index(
            models[(partition, signal)],
            RETRIEVAL_PARAMETERS["projection_dim"],
            RETRIEVAL_PARAMETERS["nlist"],
            RETRIEVAL_PARAMETERS["pq_m"],
            RETRIEVAL_PARAMETERS["nprobe"],
        )
        indexes[partition] = index
    for source in ("S2", "S3"):
        row_count = 0
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                paths[source],
                sep="\t",
                usecols=["entity_id", raw_column, "country"],
                dtype="string",
                chunksize=args.chunksize,
            ),
            start=1,
        ):
            encoded = t2.numeric_id_series(chunk["entity_id"], source)
            countries = t2.normalize_country_series(chunk["country"])
            transformed = transform(chunk[raw_column]).fillna("").astype(str)
            for partition in partitions:
                mask = countries.eq(partition).to_numpy()
                if not bool(mask.any()):
                    continue
                partition_values = transformed.loc[mask].reset_index(drop=True)
                partition_ids = encoded[mask]
                nonempty = partition_values.ne("").to_numpy()
                values = partition_values.loc[nonempty].tolist()
                ids = partition_ids[nonempty]
                for start in range(0, len(values), args.transform_batch_size):
                    stop = min(start + args.transform_batch_size, len(values))
                    vectors = t2.project_texts(models[(partition, signal)], values[start:stop])
                    nonzero = np.linalg.norm(vectors, axis=1) > 0
                    if bool(nonzero.any()):
                        indexes[partition].add_with_ids(
                            vectors[nonzero],
                            np.ascontiguousarray(ids[start:stop][nonzero], dtype=np.int64),
                        )
                    del vectors
            row_count += len(chunk)
            if chunk_number % 5 == 0:
                totals = ", ".join(f"{part}={indexes[part].ntotal:,}" for part in partitions)
                log(f"Persist {signal} {source}: {row_count:,} rows; {totals}")
            del chunk, transformed
            gc.collect()
    created = []
    totals = {}
    for partition in partitions:
        path = bundle_dir / f"{signal}.{partition}.faiss"
        temporary = path.with_suffix(path.suffix + ".tmp")
        faiss.write_index(indexes[partition], str(temporary))
        os.replace(temporary, path)
        created.append(path)
        totals[partition] = int(indexes[partition].ntotal)
    del indexes
    gc.collect()
    return created, totals


def build_retrieval_bundle(
    paths: dict[str, Path],
    output_dir: Path,
    eval_df: pd.DataFrame,
    upstream_hashes: dict[str, str],
    args: argparse.Namespace,
    runtime: RuntimeRecorder,
) -> dict[str, object]:
    bundle_dir = output_dir / "retrieval_bundle"
    manifest_path = output_dir / "task6_retrieval_manifest.json"
    if manifest_path.exists():
        log("Using existing retrieval bundle manifest; hashes will be revalidated")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError("Partial retrieval bundle exists without a manifest")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    sample, unused_truth, source_counts = t25.collect_training_sample_and_truth_records(
        {"S2": paths["S2"], "S3": paths["S3"]}, set(), args.chunksize, 30
    )
    if unused_truth:
        raise AssertionError("Ground truth entered Task 6 retrieval construction")
    sample = t2.trim_training_sample(sample, 50_000)
    runtime.add("retrieval_training_sample_scan", started, "build")

    started = time.perf_counter()
    models, partitions = fit_frozen_models(sample, eval_df)
    runtime.add("retrieval_transform_fit", started, "build")
    del sample
    gc.collect()

    artifacts = []
    index_totals: dict[str, dict[str, int]] = {}
    for signal in SIGNALS:
        for partition in partitions:
            model_path = bundle_dir / f"{signal}.{partition}.model.pkl"
            temporary = model_path.with_suffix(model_path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(text_model_payload(models[(partition, signal)]), handle, protocol=5)
            os.replace(temporary, model_path)
            artifacts.append(
                artifact_record(
                    model_path,
                    output_dir,
                    "sklearn_tfidf_random_projection",
                    signal,
                    partition,
                    RETRIEVAL_PARAMETERS,
                    upstream_hashes,
                )
            )
        started = time.perf_counter()
        index_paths, totals = build_one_signal_indexes(
            signal, paths, models, partitions, bundle_dir, args
        )
        runtime.add(f"retrieval_index_build_{signal}", started, "build")
        index_totals[signal] = totals
        for path, partition in zip(index_paths, partitions):
            artifacts.append(
                artifact_record(
                    path,
                    output_dir,
                    "faiss_index_ivfpq_with_encoded_ids",
                    signal,
                    partition,
                    RETRIEVAL_PARAMETERS,
                    upstream_hashes,
                )
            )
        for partition in partitions:
            models[(partition, signal)].train_texts = []
        gc.collect()

    metadata = {
        "task": TASK_NAME,
        "created_at_utc": utc_now(),
        "configuration": FROZEN_CONFIGURATION,
        "candidate_cap": FROZEN_CANDIDATE_CAP,
        "signal_order": list(SIGNALS),
        "partitions": partitions,
        "parameters": RETRIEVAL_PARAMETERS,
        "source_candidate_counts": source_counts,
        "index_ntotal": index_totals,
        "id_encoding": {
            "S2": "numeric suffix",
            "S3": "1_000_000_000 + numeric suffix",
            "decoder": "task2_candidate_generation.decode_candidate_id",
        },
        "candidate_ordering": (
            "Original baseline name-top-100 plus address-top-100 ordered union; "
            "missing-address records use name-top-200; fill unseen candidates to 250 "
            "with RRF(k=60) over active frozen signals and candidate-ID tie-break."
        ),
        "exact_map_required": False,
        "exact_map_reason": (
            "Task 3 frozen final reconstruction intentionally calls the cap-250 "
            "configuration with empty exact maps."
        ),
        "code_and_upstream_hashes": upstream_hashes,
    }
    metadata_path = bundle_dir / "bundle_metadata.json"
    atomic_json(metadata_path, metadata)
    artifacts.append(
        artifact_record(
            metadata_path,
            output_dir,
            "retrieval_bundle_metadata",
            parameters=RETRIEVAL_PARAMETERS,
            upstream_hashes=upstream_hashes,
        )
    )
    manifest = {
        **metadata,
        "artifact_count": len(artifacts),
        "bundle_bytes": sum(int(row["bytes"]) for row in artifacts),
        "artifacts": artifacts,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def candidate_rows(chunk: pd.DataFrame, source: str) -> list[tuple[object, ...]]:
    encoded = t2.numeric_id_series(chunk["entity_id"], source)
    raw_name = chunk["business_name"].fillna("").astype(str)
    raw_address = chunk["business_address"].fillna("").astype(str)
    normalized_name = t2.normalize_series(raw_name).fillna("").astype(str)
    transliterated = t25.transliterate_series(raw_name).fillna("").astype(str)
    suffix = t25.strip_legal_suffix_series(raw_name).fillna("").astype(str)
    normalized_address = t2.normalize_series(raw_address).fillna("").astype(str)
    numbers = [t3.number_sequence(value) for value in raw_address]
    countries = t2.normalize_country_series(chunk["country"]).fillna("").astype(str)
    non_ascii = [int(t2.contains_non_ascii(value)) for value in raw_name]
    return list(
        zip(
            encoded.tolist(),
            chunk["entity_id"].astype(str).tolist(),
            [source] * len(chunk),
            raw_name.tolist(),
            raw_address.tolist(),
            normalized_name.tolist(),
            transliterated.tolist(),
            suffix.tolist(),
            normalized_address.tolist(),
            numbers,
            countries.tolist(),
            non_ascii,
        )
    )


def build_candidate_store(
    paths: dict[str, Path],
    output_dir: Path,
    upstream_hashes: dict[str, str],
    args: argparse.Namespace,
    runtime: RuntimeRecorder,
) -> dict[str, object]:
    store_dir = output_dir / "candidate_text_store"
    manifest_path = output_dir / "task6_candidate_store_manifest.json"
    if manifest_path.exists():
        log("Using existing candidate text store manifest; hashes will be revalidated")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if store_dir.exists() and any(store_dir.iterdir()):
        raise FileExistsError("Partial candidate text store exists without a manifest")
    store_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = [store_dir / f"candidate_text_{index:02d}.sqlite.tmp" for index in range(SHARD_COUNT)]
    if any(path.exists() for path in temporary_paths):
        raise FileExistsError("Temporary candidate-store shards already exist")
    connections = [sqlite3.connect(path) for path in temporary_paths]
    schema = """
        CREATE TABLE candidates (
            encoded_id INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('S2','S3')),
            raw_name TEXT NOT NULL,
            raw_address TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            transliterated_name TEXT NOT NULL,
            suffix_name TEXT NOT NULL,
            normalized_address TEXT NOT NULL,
            number_sequence TEXT NOT NULL,
            normalized_country TEXT NOT NULL,
            name_non_ascii INTEGER NOT NULL CHECK (name_non_ascii IN (0,1))
        ) WITHOUT ROWID
    """
    for connection in connections:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(schema)
    insert_sql = "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
    source_counts: dict[str, int] = {}
    started = time.perf_counter()
    for source in ("S2", "S3"):
        count = 0
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                paths[source],
                sep="\t",
                usecols=["entity_id", "business_name", "business_address", "country"],
                dtype="string",
                chunksize=args.chunksize,
            ),
            start=1,
        ):
            rows = candidate_rows(chunk, source)
            buckets: list[list[tuple[object, ...]]] = [[] for _ in range(SHARD_COUNT)]
            for row in rows:
                buckets[int(row[0]) % SHARD_COUNT].append(row)
            for shard_id, values in enumerate(buckets):
                if values:
                    connections[shard_id].executemany(insert_sql, values)
                    connections[shard_id].commit()
            count += len(rows)
            if chunk_number % 5 == 0:
                log(f"Candidate store {source}: {count:,} rows")
            del chunk, rows, buckets
            gc.collect()
        source_counts[source] = count
    for connection in connections:
        connection.execute("PRAGMA optimize")
        connection.commit()
        connection.close()
    final_paths = []
    for shard_id, temporary in enumerate(temporary_paths):
        final = store_dir / f"candidate_text_{shard_id:02d}.sqlite"
        os.replace(temporary, final)
        final_paths.append(final)
    runtime.add("candidate_text_store_build", started, "build")
    artifacts = [
        artifact_record(
            path,
            output_dir,
            "sqlite_candidate_text_shard",
            partition=f"encoded_id_mod_{SHARD_COUNT}={index}",
            parameters={"shard_count": SHARD_COUNT, "schema_version": 1},
            upstream_hashes=upstream_hashes,
        )
        for index, path in enumerate(final_paths)
    ]
    manifest = {
        "task": TASK_NAME,
        "created_at_utc": utc_now(),
        "configuration": FROZEN_CONFIGURATION,
        "store_type": "SQLite sharded selective candidate text store",
        "schema_version": 1,
        "shard_count": SHARD_COUNT,
        "shard_function": f"encoded_id % {SHARD_COUNT}",
        "lookup_contract": "Read only requested encoded IDs; return sorted unique TextStore arrays.",
        "columns": [
            "encoded_id",
            "entity_id",
            "source",
            *STORE_TEXT_FIELDS,
            "name_non_ascii",
        ],
        "source_candidate_counts": source_counts,
        "row_count": sum(source_counts.values()),
        "store_bytes": sum(int(row["bytes"]) for row in artifacts),
        "code_and_upstream_hashes": upstream_hashes,
        "artifacts": artifacts,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def verify_manifest_artifacts(
    manifest: dict[str, object], output_dir: Path, expected_type_prefix: str
) -> list[dict[str, object]]:
    rows = []
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise AssertionError(f"{expected_type_prefix} manifest has no artifacts")
    for artifact in artifacts:
        path = output_dir / str(artifact["path"])
        exists = path.exists()
        size_match = exists and path.stat().st_size == int(artifact["bytes"])
        hash_match = size_match and file_sha256(path) == artifact["sha256"]
        rows.append(
            {
                "check": f"{expected_type_prefix}:{artifact['path']}",
                "passed": bool(hash_match),
                "detail": "exists, byte size, and SHA-256 match" if hash_match else "artifact verification failed",
            }
        )
    failures = [row for row in rows if not row["passed"]]
    if failures:
        raise AssertionError(f"Artifact hash verification failed: {failures[:3]}")
    return rows


def query_retrieval_bundle(
    manifest: dict[str, object],
    output_dir: Path,
    eval_df: pd.DataFrame,
    args: argparse.Namespace,
    runtime: RuntimeRecorder,
    run_name: str,
) -> t3.RetrievalCheckpoint:
    bundle_dir = output_dir / "retrieval_bundle"
    ids: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    total_load = total_query = 0.0
    faiss.omp_set_num_threads(args.workers)
    for signal in SIGNALS:
        raw_column, transform, _ = SIGNAL_DEFINITIONS[signal]
        result_ids = np.full((len(eval_df), 250), -1, dtype=np.int64)
        result_scores = np.full((len(eval_df), 250), -np.inf, dtype=np.float32)
        for partition in PARTITIONS:
            load_started = time.perf_counter()
            model = load_text_model(bundle_dir / f"{signal}.{partition}.model.pkl")
            index = faiss.read_index(str(bundle_dir / f"{signal}.{partition}.faiss"))
            index.nprobe = min(RETRIEVAL_PARAMETERS["nprobe"], index.nlist)
            total_load += time.perf_counter() - load_started
            positions = np.flatnonzero(eval_df["normalized_country"].eq(partition).to_numpy())
            values = transform(eval_df.loc[positions, raw_column]).fillna("").astype(str)
            nonempty = values.ne("").to_numpy()
            query_started = time.perf_counter()
            if bool(nonempty.any()):
                vectors = t2.project_texts(model, values.loc[nonempty].tolist())
                depth = 1_000 if signal.startswith("baseline_") else 250
                distances, labels = index.search(vectors, depth)
                destination = positions[nonempty]
                result_ids[destination, :] = labels[:, :250]
                result_scores[destination, :] = distances[:, :250]
                del vectors, distances, labels
            total_query += time.perf_counter() - query_started
            del model, index, values
            gc.collect()
        ids[signal] = result_ids
        scores[signal] = result_scores
    runtime.rows.append(
        {
            "category": "reload-only",
            "phase": f"{run_name}_retrieval_bundle_load",
            "runtime_seconds": total_load,
            "current_rss_mb": t3.current_rss_mb(),
            "process_peak_rss_mb": t3.peak_rss_mb(),
            "detail": "Deserialize ten fitted transform objects and ten FAISS indexes.",
        }
    )
    runtime.rows.append(
        {
            "category": "reload-only",
            "phase": f"{run_name}_retrieval_query",
            "runtime_seconds": total_query,
            "current_rss_mb": t3.current_rss_mb(),
            "process_peak_rss_mb": t3.peak_rss_mb(),
            "detail": f"Query five frozen signals for {len(eval_df):,} S1 entities.",
        }
    )
    return t3.RetrievalCheckpoint(ids=ids, scores=scores)


def compare_store_arrays(actual: t3.TextStore, expected: t3.TextStore) -> list[dict[str, object]]:
    rows = []
    for field in ("ids", *STORE_TEXT_FIELDS, "name_non_ascii"):
        left = getattr(actual, field)
        right = getattr(expected, field)
        mismatch = int(np.count_nonzero(left != right))
        rows.append(
            t5.parity_row(
                "candidate_text_store",
                field,
                mismatch == 0,
                len(left),
                mismatch,
                detail="Persisted selective lookup versus direct raw S2/S3 normalization.",
            )
        )
    return rows


def integrity_row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def build_scale_projection(
    smoke_summary: dict[str, object],
    runtime: RuntimeRecorder,
    bundle_manifest: dict[str, object],
    store_manifest: dict[str, object],
) -> dict[str, object]:
    target_s1 = 2_206_821
    total_rows = [
        row for row in runtime.rows if row["phase"] == "production_total_excluding_build"
    ]
    if len(total_rows) != 1:
        raise AssertionError("Expected one production total runtime row")
    measured = float(total_rows[0]["runtime_seconds"])
    seconds_per_entity = measured / int(smoke_summary["s1_entities_processed"])
    pairs_per_entity = float(smoke_summary["total_candidates"]) / int(smoke_summary["s1_entities_processed"])
    output_bytes_per_pair = float(smoke_summary["output_bytes"]) / int(smoke_summary["total_candidates"])
    projected_checkpoints = int(round(target_s1 * pairs_per_entity * output_bytes_per_pair))
    persistent_bytes = int(bundle_manifest["bundle_bytes"]) + int(store_manifest["store_bytes"])
    return {
        "scope": "projection only; full-scale execution was not run",
        "target_s1_entities": target_s1,
        "projected_candidate_pairs": int(round(target_s1 * pairs_per_entity)),
        "measured_reload_only_smoke_entities": int(smoke_summary["s1_entities_processed"]),
        "measured_reload_only_seconds": measured,
        "projected_wall_seconds_linear": target_s1 * seconds_per_entity,
        "projected_wall_hours_linear": target_s1 * seconds_per_entity / 3600.0,
        "projected_checkpoint_bytes": projected_checkpoints,
        "persistent_bundle_bytes": int(bundle_manifest["bundle_bytes"]),
        "persistent_candidate_store_bytes": int(store_manifest["store_bytes"]),
        "persistent_total_bytes": persistent_bytes,
        "estimated_total_disk_bytes": persistent_bytes + projected_checkpoints,
        "peak_rss_basis_mb": max(float(row["process_peak_rss_mb"]) for row in runtime.rows),
        "recommended_initial_s1_batch_size": 100,
        "assumptions": [
            "Linear extrapolation from the Task 6 1,000-S1 reload-only production smoke.",
            "Same average candidate count, hardware, thread count, batch size, and storage throughput.",
            "Persistent bundle and candidate text store are reused and excluded from per-run build time.",
            "Projection is planning evidence, not a guarantee or a full-scale run.",
        ],
    }


def write_summary(
    path: Path,
    retrieval_parity: pd.DataFrame,
    store_parity: pd.DataFrame,
    integrity: pd.DataFrame,
    runtime: RuntimeRecorder,
    smoke_summary: dict[str, object],
    bundle_manifest: dict[str, object],
    store_manifest: dict[str, object],
    scale: dict[str, object],
) -> None:
    reload_seconds = next(
        float(row["runtime_seconds"])
        for row in runtime.rows
        if row["phase"] == "production_total_excluding_build"
    )
    build_seconds = sum(float(row["runtime_seconds"]) for row in runtime.rows if row["category"] == "build")
    text = f"""# Task 6 Summary

Status: **PASS**

## Persisted production assets

- Frozen candidate generator: `{FROZEN_CONFIGURATION}`, cap `{FROZEN_CANDIDATE_CAP}`.
- Retrieval bundle: {bundle_manifest['artifact_count']:,} artifacts, {bundle_manifest['bundle_bytes'] / 2**30:.2f} GiB.
- Candidate text store: {store_manifest['row_count']:,} S2/S3 records in {SHARD_COUNT} SQLite shards, {store_manifest['store_bytes'] / 2**30:.2f} GiB.
- Candidate lookup is selective by encoded ID. It does not load the full text corpus into memory.

## Reload parity

- Retrieval checks passed: {int(retrieval_parity['passed'].sum())}/{len(retrieval_parity)}.
- Candidate text, feature, score, and decision checks passed: {int(store_parity['passed'].sum())}/{len(store_parity)}.
- Integrity checks passed: {int(integrity['passed'].sum())}/{len(integrity)}.
- Top-250 signal IDs, ordering, candidate counts, and assembled candidates are exact.
- Retrieval scores satisfy absolute tolerance `{t5.RETRIEVAL_SCORE_ATOL:g}`.
- All 66 ordered features satisfy Task 5 dtype-specific tolerances; LightGBM scores satisfy `{t5.MODEL_SCORE_ATOL:g}` and score >= 0.95 decisions are exact.

## Production smoke

- S1 entities: {smoke_summary['s1_entities_processed']:,}.
- Pair rows: {smoke_summary['total_candidates']:,}; average {smoke_summary['average_candidates_per_s1']:.3f}; max {smoke_summary['maximum_candidates_per_s1']}.
- Predicted links: {smoke_summary['predicted_links']:,}; zero-prediction entities: {smoke_summary['zero_prediction_entities']:,}.
- Checkpoint files: {smoke_summary['batch_files']}; bytes: {smoke_summary['output_bytes']:,}.
- One-time build runtime: {build_seconds / 60:.2f} minutes.
- Reload-only production smoke runtime: {reload_seconds:.2f} seconds.
- Bundle load / query: {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_retrieval_bundle_load'):.2f} / {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_retrieval_query'):.2f} seconds.
- Store initialization / candidate assembly / text lookup: {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'candidate_store_initialize'):.4f} / {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_candidate_assembly'):.2f} / {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_candidate_text_lookup'):.2f} seconds.
- Feature generation / LightGBM / checkpoint write: {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_feature_generation'):.2f} / {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_lightgbm_inference'):.2f} / {next(float(row['runtime_seconds']) for row in runtime.rows if row['phase'] == 'production_checkpoint_write'):.2f} seconds.
- Process peak RSS: {max(float(row['process_peak_rss_mb']) for row in runtime.rows):.1f} MB.

## Full-scale projection

- Target S1 entities: {scale['target_s1_entities']:,}.
- Projected candidate pairs: {scale['projected_candidate_pairs']:,}.
- Linear projected runtime: {scale['projected_wall_hours_linear']:.2f} hours.
- Projected checkpoint bytes: {scale['projected_checkpoint_bytes'] / 2**30:.2f} GiB.
- Persistent plus projected checkpoint storage: {scale['estimated_total_disk_bytes'] / 2**30:.2f} GiB.
- Recommended initial S1 batch size: {scale['recommended_initial_s1_batch_size']}.
- This is a projection only. No full-scale run was performed.

## Boundaries

No Task 7 work, classifier training, threshold tuning, retrieval retuning, test-data access, ground-truth use, full-scale inference, test predictions, or submission generation was performed.
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = make_parser().parse_args()
    if args.mode == "full":
        raise RuntimeError("Task 6 full-scale execution is locked; projection only is permitted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(args)
    hashes_before = task6_upstream_hashes(paths)
    features, feature_schema, model = validate_frozen_contract(paths, args)
    runtime = RuntimeRecorder()
    if args.mode == "validate-only" and (args.output_dir / "task6_runtime_breakdown.csv").exists():
        previous_runtime = pd.read_csv(args.output_dir / "task6_runtime_breakdown.csv")
        runtime.rows.extend(
            previous_runtime.loc[previous_runtime["category"].eq("build")].to_dict("records")
        )

    selection = t5.select_smoke_entities(paths["task3_split"], args.smoke_entities)
    eval_df = t5.load_raw_s1(selection, paths["S1"], args.chunksize)
    global_indices = eval_df["eval_index"].to_numpy(dtype=np.int64)
    if tuple(sorted(eval_df["normalized_country"].unique())) != PARTITIONS:
        raise AssertionError("Task 5 smoke selection country partitions changed")

    if args.mode == "validate-only":
        for name in ("task6_retrieval_manifest.json", "task6_candidate_store_manifest.json"):
            if not (args.output_dir / name).exists():
                raise FileNotFoundError(f"validate-only requires {name}")
        bundle_manifest = json.loads((args.output_dir / "task6_retrieval_manifest.json").read_text(encoding="utf-8"))
        store_manifest = json.loads((args.output_dir / "task6_candidate_store_manifest.json").read_text(encoding="utf-8"))
    else:
        bundle_manifest = build_retrieval_bundle(
            paths, args.output_dir, eval_df, hashes_before, args, runtime
        )
        store_manifest = build_candidate_store(
            paths, args.output_dir, hashes_before, args, runtime
        )

    started = time.perf_counter()
    bundle_hash_checks = verify_manifest_artifacts(bundle_manifest, args.output_dir, "retrieval_bundle")
    store_hash_checks = verify_manifest_artifacts(store_manifest, args.output_dir, "candidate_store")
    runtime.add("persisted_artifact_hash_validation", started, "reload-only")

    first = query_retrieval_bundle(
        bundle_manifest, args.output_dir, eval_df, args, runtime, "parity_first"
    )
    retrieval_rows = t5.compare_retrieval_checkpoint(first, paths["task2_5_checkpoint"], global_indices)
    first_offsets, first_counts, first_metadata, first_sets = t5.assemble_candidates(eval_df, first)
    retrieval_rows.extend(t5.compare_frozen_candidates(global_indices, first_sets, paths["task3_candidates"]))

    second = query_retrieval_bundle(
        bundle_manifest, args.output_dir, eval_df, args, runtime, "parity_second"
    )
    for signal in SIGNALS:
        id_mismatch = int(np.count_nonzero(first.ids[signal] != second.ids[signal]))
        score_mask_match = np.array_equal(np.isfinite(first.scores[signal]), np.isfinite(second.scores[signal]))
        finite = np.isfinite(first.scores[signal]) & np.isfinite(second.scores[signal])
        score_diff = np.abs(first.scores[signal][finite] - second.scores[signal][finite])
        retrieval_rows.append(
            t5.parity_row("reload_determinism", f"{signal}_ids", id_mismatch == 0, first.ids[signal].size, id_mismatch)
        )
        retrieval_rows.append(
            t5.parity_row(
                "reload_determinism",
                f"{signal}_scores",
                score_mask_match and (not score_diff.size or float(score_diff.max()) == 0.0),
                first.scores[signal].size,
                int(np.count_nonzero(score_diff != 0.0)) + int(not score_mask_match),
                float(score_diff.max()) if score_diff.size else 0.0,
            )
        )
    retrieval_parity = pd.DataFrame(retrieval_rows)
    atomic_csv(args.output_dir / "task6_retrieval_parity.csv", retrieval_parity)
    if not retrieval_parity["passed"].all():
        raise AssertionError("Persisted retrieval parity failed")

    store_started = time.perf_counter()
    persisted_store = PersistedTextStore(args.output_dir / "candidate_text_store", store_manifest)
    runtime.add("candidate_store_initialize", store_started, "reload-only")
    parity_count = min(args.parity_entities, len(eval_df))
    parity_pair_stop = int(first_offsets[parity_count])
    parity_ids = np.unique(first_metadata.candidate_ids[:parity_pair_stop])
    lookup_started = time.perf_counter()
    parity_store = persisted_store.lookup(parity_ids)
    runtime.add("candidate_store_parity_lookup", lookup_started, "reload-only")
    direct_started = time.perf_counter()
    direct_store = t3.collect_text_store(
        parity_ids, {"S2": paths["S2"], "S3": paths["S3"]}, args.chunksize
    )
    runtime.add("candidate_store_direct_raw_reference_scan", direct_started, "validation-reference")
    store_rows = compare_store_arrays(parity_store, direct_store)
    persisted_sources = persisted_store.lookup_sources(parity_ids)
    source_mismatches = sum(
        persisted_sources[int(encoded_id)] != t2.candidate_source(int(encoded_id))
        for encoded_id in parity_ids
    )
    store_rows.append(
        t5.parity_row(
            "candidate_text_store",
            "candidate_source",
            source_mismatches == 0,
            len(parity_ids),
            source_mismatches,
            detail="Persisted source column versus the frozen encoded-ID source mapping.",
        )
    )

    parity_eval = eval_df.iloc[:parity_count].reset_index(drop=True)
    parity_metadata = t5.metadata_for_query_range(first_metadata, first_offsets, 0, parity_count)
    s1_text_parity = t3.build_s1_text(parity_eval)
    feature_started = time.perf_counter()
    parity_frame = t5.build_inference_feature_frame(
        parity_metadata, parity_eval, s1_text_parity, parity_store, args.workers
    )
    runtime.add("candidate_store_parity_feature_generation", feature_started, "reload-only")
    parity_entity_ids = set(parity_eval["source1_entity_id"].astype(str))
    reference_features = t5.read_reference_pair_features(paths["task3_pairs"], parity_entity_ids, features)
    reference_scores = t5.read_reference_scores(paths["task4a_predictions"], parity_entity_ids)
    feature_rows, _ = t5.compare_features_and_scores(
        parity_frame,
        features,
        feature_schema,
        model,
        reference_features,
        reference_scores,
    )
    store_rows.extend(feature_rows)
    candidate_store_parity = pd.DataFrame(store_rows)
    atomic_csv(args.output_dir / "task6_candidate_store_parity.csv", candidate_store_parity)
    if not candidate_store_parity["passed"].all():
        raise AssertionError("Persisted candidate text or downstream parity failed")
    del direct_store, parity_store, parity_frame, reference_features, reference_scores
    gc.collect()

    del first, second, first_metadata, first_sets
    gc.collect()
    production = query_retrieval_bundle(
        bundle_manifest, args.output_dir, eval_df, args, runtime, "production"
    )
    assembly_started = time.perf_counter()
    offsets, counts, metadata, candidate_sets = t5.assemble_candidates(eval_df, production)
    runtime.add("production_candidate_assembly", assembly_started, "reload-only")
    if int(counts.max()) > FROZEN_CANDIDATE_CAP:
        raise AssertionError("Production candidate cap exceeded")
    if len(metadata.candidate_ids) != len(np.unique(np.column_stack((metadata.query_index, metadata.candidate_ids)), axis=0)):
        raise AssertionError("Duplicate (S1, candidate) pairs in production smoke")
    lookup_started = time.perf_counter()
    production_store = persisted_store.lookup(metadata.candidate_ids)
    runtime.add("production_candidate_text_lookup", lookup_started, "reload-only")
    s1_text = t3.build_s1_text(eval_df)

    smoke_dir = args.output_dir / "production_smoke_checkpoints"
    smoke_started = time.perf_counter()
    t5.run_smoke_batches(
        smoke_dir,
        paths["task5_contract"],
        eval_df,
        offsets,
        metadata,
        s1_text,
        production_store,
        features,
        model,
        args.batch_size,
        args.workers,
        runtime,
        None,
    )
    runtime.add("production_feature_model_checkpoint_total", smoke_started, "reload-only")
    smoke_summary, predictions = t5.summarize_smoke_checkpoints(smoke_dir, eval_df, counts)
    checkpoint_manifest = json.loads(
        (smoke_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    completed_batches = list(checkpoint_manifest["completed_batches"].values())
    phase_fields = (
        ("production_feature_generation", "feature_seconds"),
        ("production_lightgbm_inference", "model_seconds"),
        ("production_checkpoint_write", "write_seconds"),
    )
    for phase, field in phase_fields:
        runtime.rows.append(
            {
                "category": "reload-only",
                "phase": phase,
                "runtime_seconds": sum(float(row[field]) for row in completed_batches),
                "current_rss_mb": t3.current_rss_mb(),
                "process_peak_rss_mb": t3.peak_rss_mb(),
                "detail": f"Aggregate measured {field} across persisted smoke checkpoints.",
            }
        )
    total_phases = {
        "persisted_artifact_hash_validation",
        "production_retrieval_bundle_load",
        "candidate_store_initialize",
        "production_retrieval_query",
        "production_candidate_assembly",
        "production_candidate_text_lookup",
        "production_feature_generation",
        "production_lightgbm_inference",
        "production_checkpoint_write",
    }
    runtime.rows.append(
        {
            "category": "reload-only",
            "phase": "production_total_excluding_build",
            "runtime_seconds": sum(
                float(row["runtime_seconds"])
                for row in runtime.rows
                if row["phase"] in total_phases
            ),
            "current_rss_mb": t3.current_rss_mb(),
            "process_peak_rss_mb": t3.peak_rss_mb(),
            "detail": "Hash validation through checkpoint write; excludes one-time bundle/store construction and parity-only work.",
        }
    )

    hashes_after = task6_upstream_hashes(paths)
    integrity_rows = [
        integrity_row("frozen_upstream_hashes_unchanged", hashes_before == hashes_after, "All frozen inputs and code retain their pre-run SHA-256."),
        integrity_row("no_ground_truth_path", "GT" not in paths, "Task 6 resolved no ground-truth input."),
        integrity_row("no_test_path", all("test" not in paths[key].name.casefold() for key in ("S1", "S2", "S3")), "Only train_source1/2/3 were resolved."),
        integrity_row("configuration_frozen", bundle_manifest["configuration"] == FROZEN_CONFIGURATION, str(bundle_manifest["configuration"])),
        integrity_row("candidate_cap_frozen", FROZEN_CANDIDATE_CAP == 250 and int(counts.max()) <= 250, f"max={int(counts.max())}"),
        integrity_row("signal_order_frozen", tuple(bundle_manifest["signal_order"]) == SIGNALS, str(bundle_manifest["signal_order"])),
        integrity_row("country_partitions_frozen", tuple(bundle_manifest["partitions"]) == PARTITIONS, str(bundle_manifest["partitions"])),
        integrity_row("retrieval_artifact_hashes", all(row["passed"] for row in bundle_hash_checks), f"{len(bundle_hash_checks)} artifacts"),
        integrity_row("candidate_store_artifact_hashes", all(row["passed"] for row in store_hash_checks), f"{len(store_hash_checks)} shards"),
        integrity_row("retrieval_parity_all", bool(retrieval_parity["passed"].all()), f"{len(retrieval_parity)} checks"),
        integrity_row("candidate_store_parity_all", bool(candidate_store_parity["passed"].all()), f"{len(candidate_store_parity)} checks"),
        integrity_row("feature_count_66", len(features) == 66, f"features={len(features)}"),
        integrity_row("model_feature_order", model.feature_name() == features, "Saved LightGBM order equals Task 3 schema."),
        integrity_row("threshold_frozen", FROZEN_THRESHOLD == 0.95, "score >= 0.95"),
        integrity_row("smoke_entity_count", len(eval_df) == args.smoke_entities, f"entities={len(eval_df)}"),
        integrity_row("candidate_pair_reconciliation", len(metadata.candidate_ids) == int(counts.sum()), f"pairs={len(metadata.candidate_ids)}"),
        integrity_row("candidate_counts_nonzero", bool((counts > 0).all()), f"min={int(counts.min())}"),
        integrity_row("candidate_counts_at_cap", int(counts.max()) <= 250, f"max={int(counts.max())}"),
        integrity_row("prediction_pair_reconciliation", len(predictions) == len(metadata.candidate_ids), f"rows={len(predictions)}"),
        integrity_row("prediction_duplicate_pairs", int(predictions.duplicated(["source1_entity_id", "candidate_entity_id"]).sum()) == 0, "No duplicate pairs."),
        integrity_row("prediction_scores_finite", bool(np.isfinite(predictions["lightgbm_score"]).all()), "All saved scores are finite."),
        integrity_row("prediction_decisions_consistent", bool((predictions["predicted_match"].to_numpy() == (predictions["lightgbm_score"].to_numpy() >= FROZEN_THRESHOLD)).all()), "All decisions equal score >= 0.95."),
        integrity_row("candidate_source_domain", set(predictions["candidate_source"].astype(str)) <= {"S2", "S3"}, str(sorted(predictions["candidate_source"].astype(str).unique()))),
        integrity_row("selective_store_lookup", len(production_store.ids) == len(np.unique(metadata.candidate_ids)), f"looked up {len(production_store.ids):,} unique of {store_manifest['row_count']:,} stored records"),
        integrity_row("full_scale_not_run", len(eval_df) == 1_000, "Only the fixed Task 5 1,000-S1 smoke set was executed."),
    ]
    integrity = pd.DataFrame(integrity_rows)
    atomic_csv(args.output_dir / "task6_integrity_report.csv", integrity)
    if len(integrity) < 20 or not integrity["passed"].all():
        raise AssertionError("Task 6 integrity gate failed")

    scale = build_scale_projection(smoke_summary, runtime, bundle_manifest, store_manifest)
    atomic_json(args.output_dir / "task6_scale_estimate.json", scale)
    runtime_frame = pd.DataFrame(runtime.rows)
    atomic_csv(args.output_dir / "task6_runtime_breakdown.csv", runtime_frame)
    summary_path = args.output_dir / "task6_summary.md"
    write_summary(
        summary_path,
        retrieval_parity,
        candidate_store_parity,
        integrity,
        runtime,
        smoke_summary,
        bundle_manifest,
        store_manifest,
        scale,
    )

    output_artifacts = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "task6_run_manifest.json":
            output_artifacts.append(
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
        "mode": args.mode,
        "configuration": FROZEN_CONFIGURATION,
        "candidate_cap": FROZEN_CANDIDATE_CAP,
        "threshold": FROZEN_THRESHOLD,
        "smoke_summary": smoke_summary,
        "retrieval_manifest_sha256": file_sha256(args.output_dir / "task6_retrieval_manifest.json"),
        "candidate_store_manifest_sha256": file_sha256(args.output_dir / "task6_candidate_store_manifest.json"),
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "output_artifacts": output_artifacts,
        "boundaries": {
            "task7_started": False,
            "classifier_trained": False,
            "threshold_tuned": False,
            "retrieval_retuned": False,
            "test_data_used": False,
            "ground_truth_used": False,
            "full_scale_run": False,
            "submission_created": False,
        },
    }
    atomic_json(args.output_dir / "task6_run_manifest.json", run_manifest)
    log("Task 6 PASS: persisted bundle/store reload and production smoke are reproducible")


if __name__ == "__main__":
    main()
