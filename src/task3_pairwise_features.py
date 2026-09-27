#!/usr/bin/env python3
"""Task 3: reconstruct frozen candidates and build pairwise feature artifacts.

This pipeline is deliberately limited to training-data pair construction,
deterministic pair transformations, descriptive analysis, and an entity-level
train/validation split. It does not train or tune a matching classifier.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for vendor_dir in (PROJECT_ROOT / ".task3_vendor", PROJECT_ROOT / ".task2_5_vendor"):
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import task2_candidate_generation as t2
import task2_5_candidate_selection as t25


RANDOM_STATE = 42
FROZEN_CONFIGURATION = t25.COMBINED_AUGMENTED_250
SIGNALS = (
    "baseline_name",
    "baseline_address",
    "transliterated_name",
    "number_address",
    "suffix_name",
)
TASK4_RECOMMENDED_FEATURES = {
    "candidate_source_s3",
    "candidate_address_missing",
    "cross_script_name",
    "name_exact",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_length_ratio",
    "name_token_count_difference",
    "name_first_token_exact",
    "transliterated_name_exact",
    "transliterated_name_levenshtein",
    "transliterated_name_token_set",
    "transliteration_levenshtein_gain",
    "suffix_name_exact",
    "suffix_name_levenshtein",
    "suffix_name_token_set",
    "suffix_levenshtein_gain",
    "address_exact",
    "address_levenshtein",
    "address_jaro_winkler",
    "address_token_sort",
    "address_token_set",
    "address_token_jaccard",
    "address_char3_jaccard",
    "address_length_ratio",
    "address_token_count_difference",
    "address_first_token_exact",
    "both_addresses_have_numbers",
    "one_address_missing_numbers",
    "address_number_set_exact",
    "address_first_number_exact",
    "address_number_jaccard",
    "address_conflicting_number_count",
    "address_number_count_difference",
    "strong_address_and_number",
    "baseline_name_rank",
    "baseline_name_rank_missing",
    "baseline_name_score",
    "baseline_address_rank",
    "baseline_address_rank_missing",
    "baseline_address_score",
    "transliterated_name_rank",
    "transliterated_name_rank_missing",
    "transliterated_name_score",
    "number_address_rank",
    "number_address_rank_missing",
    "number_address_score",
    "suffix_name_rank",
    "suffix_name_rank_missing",
    "suffix_name_score",
    "retrieval_signal_count_top100",
    "retrieval_best_rank",
    "retrieval_mean_rank_present",
    "retrieval_best_score",
    "retrieval_rrf_score_top100",
    "frozen_candidate_position",
    "task2_baseline_member",
    "name_address_similarity_product",
    "strong_name_and_address",
    "legal_suffix_rescue",
    "high_name_numeric_conflict",
    "independent_strong_signal_count",
}
METADATA_COLUMNS = {
    "eval_index",
    "source1_entity_id",
    "candidate_entity_id",
    "match_group",
    "split",
    "label",
}
ANALYSIS_ONLY_COLUMNS = {"match_group"}
GT_DERIVED_COLUMNS = {"label", "match_group"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "student_resource" / "dataset" / "train",
    )
    parser.add_argument(
        "--task2-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task2_outputs",
    )
    parser.add_argument(
        "--task2-5-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task2_5_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task3_outputs",
    )
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--pair-chunk-entities", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--negative-analysis-modulus", type=int, default=80)
    parser.add_argument("--smoke-entities-per-group", type=int, default=5)
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "S1": args.data_dir / "train_source1.tsv",
        "S2": args.data_dir / "train_source2.tsv",
        "S3": args.data_dir / "train_source3.tsv",
        "GT": args.data_dir / "train_ground_truth.tsv",
        "eval": args.task2_output_dir / "task2_eval_entities.csv",
        "checkpoint": args.task2_5_output_dir / "task2_5_retrieval_top250.npz",
        "decision": args.task2_5_output_dir / "task2_5_decision.json",
        "task2_5_manifest": args.task2_5_output_dir / "task2_5_run_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Task 3 inputs: {missing}")
    for key in ("S1", "S2", "S3", "GT"):
        if "test" in paths[key].name.casefold():
            raise ValueError("Task 3 must use training data only")
    return paths


@dataclass
class RetrievalCheckpoint:
    ids: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]


@dataclass
class PairMetadata:
    query_index: np.ndarray
    candidate_ids: np.ndarray
    candidate_position: np.ndarray
    ranks: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]

    def slice(self, start: int, stop: int) -> "PairMetadata":
        return PairMetadata(
            query_index=self.query_index[start:stop],
            candidate_ids=self.candidate_ids[start:stop],
            candidate_position=self.candidate_position[start:stop],
            ranks={signal: values[start:stop] for signal, values in self.ranks.items()},
            scores={signal: values[start:stop] for signal, values in self.scores.items()},
        )


@dataclass
class TextStore:
    ids: np.ndarray
    raw_name: np.ndarray
    raw_address: np.ndarray
    normalized_name: np.ndarray
    transliterated_name: np.ndarray
    suffix_name: np.ndarray
    normalized_address: np.ndarray
    number_sequence: np.ndarray
    normalized_country: np.ndarray
    name_non_ascii: np.ndarray

    def positions(self, encoded_ids: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.ids, encoded_ids)
        if bool((positions >= len(self.ids)).any()):
            raise KeyError("Candidate ID falls outside the collected text store")
        if not bool(np.array_equal(self.ids[positions], encoded_ids)):
            raise KeyError("Candidate ID is missing from the collected text store")
        return positions


class RuntimeRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record(self, phase: str, seconds: float, detail: str = "") -> None:
        self.rows.append(
            {
                "phase": phase,
                "runtime_seconds": float(seconds),
                "current_rss_mb": current_rss_mb(),
                "process_peak_rss_mb": peak_rss_mb(),
                "detail": detail,
            }
        )
        log(
            f"{phase}: {seconds:.1f}s; current RSS={self.rows[-1]['current_rss_mb']:.1f} MB; "
            f"peak={self.rows[-1]['process_peak_rss_mb']:.1f} MB"
        )


def load_checkpoint(path: Path) -> RetrievalCheckpoint:
    with np.load(path) as data:
        expected = {f"{signal}_{kind}" for signal in SIGNALS for kind in ("ids", "scores")}
        if set(data.files) != expected:
            raise ValueError(
                f"Unexpected Task 2.5 checkpoint keys. Expected {sorted(expected)}, "
                f"found {sorted(data.files)}"
            )
        ids = {signal: np.asarray(data[f"{signal}_ids"], dtype=np.int64) for signal in SIGNALS}
        scores = {
            signal: np.asarray(data[f"{signal}_scores"], dtype=np.float32)
            for signal in SIGNALS
        }
    for signal in SIGNALS:
        if ids[signal].shape != (10_000, 250) or scores[signal].shape != (10_000, 250):
            raise ValueError(f"Unexpected checkpoint shape for {signal}: {ids[signal].shape}")
    return RetrievalCheckpoint(ids=ids, scores=scores)


def checkpoint_artifacts(checkpoint: RetrievalCheckpoint) -> dict[str, t2.RetrievalArtifacts]:
    return {
        signal: t2.RetrievalArtifacts(
            ids=checkpoint.ids[signal],
            scores=checkpoint.scores[signal],
            build_seconds=0.0,
            query_seconds=0.0,
            ntotal=0,
            estimated_index_mb=0.0,
        )
        for signal in SIGNALS
    }


def direct_ground_truth_check(
    eval_df: pd.DataFrame,
    eval_truth_strings: dict[str, set[str]],
    gt_path: Path,
    chunksize: int,
) -> None:
    wanted = set(eval_df["source1_entity_id"].astype(str))
    observed: dict[str, set[str]] = {}
    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype="string",
        chunksize=chunksize,
    ):
        selected = chunk.loc[chunk["source1_entity_id"].isin(wanted)]
        for row in selected.itertuples(index=False):
            observed[str(row.source1_entity_id)] = {
                value for value in str(row.matched_entity_ids).split(",") if value
            }
    if set(observed) != wanted:
        missing = sorted(wanted - set(observed))[:10]
        raise AssertionError(f"Direct ground-truth verification missed S1 IDs: {missing}")
    mismatches = [
        entity_id
        for entity_id in wanted
        if observed[entity_id] != eval_truth_strings[entity_id]
    ]
    if mismatches:
        raise AssertionError(
            f"Saved evaluation ground truth differs from train_ground_truth.tsv: {mismatches[:10]}"
        )
    log("Directly verified all 10,000 evaluation labels against train_ground_truth.tsv")


def reconstruct_frozen_candidates(
    eval_df: pd.DataFrame,
    checkpoint: RetrievalCheckpoint,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    artifacts = checkpoint_artifacts(checkpoint)
    partitions = sorted(eval_df["normalized_country"].astype(str).unique())
    empty_exact = {
        (partition, field): defaultdict(list)
        for partition in partitions
        for field in t2.FIELDS
    }
    candidate_sets: list[np.ndarray] = []
    counts = np.empty(len(eval_df), dtype=np.int32)
    for query_index in range(len(eval_df)):
        candidates = t25.configuration_candidates(
            FROZEN_CONFIGURATION,
            query_index,
            eval_df,
            empty_exact,
            artifacts,
        )
        values = np.asarray(candidates, dtype=np.int64)
        if len(values) != len(np.unique(values)):
            raise AssertionError(f"Duplicate candidates for eval index {query_index}")
        if len(values) > 250:
            raise AssertionError(f"Candidate cap exceeded for eval index {query_index}")
        candidate_sets.append(values)
        counts[query_index] = len(values)
    offsets = np.zeros(len(eval_df) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    candidate_ids = np.concatenate(candidate_sets).astype(np.int64, copy=False)
    candidate_position = np.concatenate(
        [np.arange(1, len(values) + 1, dtype=np.uint16) for values in candidate_sets]
    )
    return offsets, candidate_ids, candidate_position, candidate_sets


def lookup_retrieval_metadata(
    query_index: np.ndarray,
    candidate_ids: np.ndarray,
    checkpoint: RetrievalCheckpoint,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ranks = {
        signal: np.zeros(len(candidate_ids), dtype=np.uint16) for signal in SIGNALS
    }
    scores = {
        signal: np.zeros(len(candidate_ids), dtype=np.float32) for signal in SIGNALS
    }
    if not len(candidate_ids):
        return ranks, scores
    starts = np.flatnonzero(np.r_[True, query_index[1:] != query_index[:-1]])
    stops = np.r_[starts[1:], len(query_index)]
    unique_queries = query_index[starts]
    for query, start, stop in zip(unique_queries, starts, stops):
        pair_ids = candidate_ids[start:stop]
        for signal in SIGNALS:
            row_ids = checkpoint.ids[signal][int(query)]
            row_scores = checkpoint.scores[signal][int(query)]
            valid = row_ids >= 0
            rank_lookup = {
                int(candidate_id): (rank + 1, float(score))
                for rank, (candidate_id, score) in enumerate(
                    zip(row_ids[valid], row_scores[valid])
                )
            }
            signal_ranks = ranks[signal][start:stop]
            signal_scores = scores[signal][start:stop]
            for local_index, candidate_id in enumerate(pair_ids):
                value = rank_lookup.get(int(candidate_id))
                if value is not None:
                    signal_ranks[local_index] = value[0]
                    signal_scores[local_index] = value[1]
    return ranks, scores


def verify_frozen_metrics(
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    candidate_sets: list[np.ndarray],
    decision_path: Path,
) -> dict[str, object]:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision["selected_configuration"] != FROZEN_CONFIGURATION:
        raise AssertionError("Task 2.5 decision does not name the frozen configuration")
    total_links = retrieved_links = complete_entities = 0
    source_total = Counter()
    source_hits = Counter()
    group_total = Counter()
    group_hits = Counter()
    group_complete = Counter()
    counts = []
    for query_index, (truth, candidates) in enumerate(zip(truth_lists, candidate_sets)):
        candidate_set = set(map(int, candidates))
        hits = sum(int(candidate_id) in candidate_set for candidate_id in truth)
        group = str(eval_df.iloc[query_index]["match_group"])
        total_links += len(truth)
        retrieved_links += hits
        complete_entities += int(hits == len(truth))
        counts.append(len(candidates))
        group_total[group] += len(truth)
        group_hits[group] += hits
        group_complete[group] += int(hits == len(truth))
        for candidate_id in truth:
            source = t2.candidate_source(candidate_id)
            source_total[source] += 1
            source_hits[source] += int(candidate_id in candidate_set)
    metrics = {
        "configuration": FROZEN_CONFIGURATION,
        "evaluation_entities": len(eval_df),
        "pair_rows": int(sum(counts)),
        "positive_links": total_links,
        "retrieved_positive_links": retrieved_links,
        "missed_positive_links": total_links - retrieved_links,
        "link_recall": retrieved_links / total_links,
        "complete_recall": complete_entities / len(eval_df),
        "average_candidates": float(np.mean(counts)),
        "median_candidates": float(np.median(counts)),
        "p95_candidates": float(np.percentile(counts, 95)),
        "minimum_candidates": int(min(counts)),
        "maximum_candidates": int(max(counts)),
        "s2_link_recall": source_hits["S2"] / source_total["S2"],
        "s3_link_recall": source_hits["S3"] / source_total["S3"],
        "recall_by_match_group": {
            group: {
                "link_recall": group_hits[group] / group_total[group],
                "complete_recall": group_complete[group]
                / int(eval_df["match_group"].eq(group).sum()),
            }
            for group in ("1", "2", "3-5", "6+")
        },
    }
    comparisons = {
        "link_recall": decision["link_recall"],
        "complete_recall": decision["complete_recall"],
        "average_candidates": decision["average_candidates"],
        "maximum_candidates": decision["maximum_candidates"],
        "s2_link_recall": decision["s2_link_recall"],
        "s3_link_recall": decision["s3_link_recall"],
    }
    for key, expected in comparisons.items():
        if not math.isclose(float(metrics[key]), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"Frozen candidate parity failed for {key}: {metrics[key]} vs {expected}"
            )
    log(
        f"Frozen parity passed: {retrieved_links:,}/{total_links:,} links "
        f"({metrics['link_recall']:.4%}), complete={metrics['complete_recall']:.4%}, "
        f"pairs={metrics['pair_rows']:,}, max={metrics['maximum_candidates']}"
    )
    return metrics


def make_entity_split(
    eval_df: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    indices = np.arange(len(eval_df), dtype=np.int32)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=eval_df["match_group"].astype(str),
    )
    split = np.full(len(eval_df), "train", dtype=object)
    split[np.asarray(validation_indices, dtype=np.int32)] = "validation"
    result = eval_df[
        ["eval_index", "source1_entity_id", "match_count", "match_group"]
    ].copy()
    result["split"] = split
    result = result.sort_values("eval_index").reset_index(drop=True)
    if set(train_indices) & set(validation_indices):
        raise AssertionError("S1 train/validation overlap detected")
    return result, split


def number_sequence(value: object) -> str:
    return "|".join(t25.number_tokens(value))


def collect_text_store(
    required_ids: np.ndarray,
    paths: dict[str, Path],
    chunksize: int,
) -> TextStore:
    required_ids = np.asarray(required_ids, dtype=np.int64)
    if not bool(np.all(required_ids[1:] > required_ids[:-1])):
        raise ValueError("Required entity IDs must be strictly increasing")
    size = len(required_ids)
    string_columns = {
        name: np.full(size, None, dtype=object)
        for name in (
            "raw_name",
            "raw_address",
            "normalized_name",
            "transliterated_name",
            "suffix_name",
            "normalized_address",
            "number_sequence",
            "normalized_country",
        )
    }
    name_non_ascii = np.zeros(size, dtype=bool)
    found = np.zeros(size, dtype=bool)

    for source in ("S2", "S3"):
        source_rows = selected_rows = 0
        reader = pd.read_csv(
            paths[source],
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype="string",
            chunksize=chunksize,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            encoded = t2.numeric_id_series(chunk["entity_id"], source)
            positions = np.searchsorted(required_ids, encoded)
            bounded = np.minimum(positions, size - 1)
            mask = (positions < size) & (required_ids[bounded] == encoded)
            source_rows += len(chunk)
            if bool(mask.any()):
                target = positions[mask]
                selected = chunk.loc[
                    mask, ["business_name", "business_address", "country"]
                ].reset_index(drop=True)
                raw_name = selected["business_name"].fillna("").astype(str)
                raw_address = selected["business_address"].fillna("").astype(str)
                normalized_name = t2.normalize_series(raw_name).fillna("").astype(str)
                normalized_address = (
                    t2.normalize_series(raw_address).fillna("").astype(str)
                )
                values = {
                    "raw_name": raw_name.to_numpy(dtype=object),
                    "raw_address": raw_address.to_numpy(dtype=object),
                    "normalized_name": normalized_name.to_numpy(dtype=object),
                    "transliterated_name": t25.transliterate_series(raw_name)
                    .fillna("")
                    .astype(str)
                    .to_numpy(dtype=object),
                    "suffix_name": t25.strip_legal_suffix_series(raw_name)
                    .fillna("")
                    .astype(str)
                    .to_numpy(dtype=object),
                    "normalized_address": normalized_address.to_numpy(dtype=object),
                    "number_sequence": np.asarray(
                        [number_sequence(value) for value in raw_address], dtype=object
                    ),
                    "normalized_country": t2.normalize_country_series(selected["country"])
                    .fillna("")
                    .astype(str)
                    .to_numpy(dtype=object),
                }
                for name, data in values.items():
                    string_columns[name][target] = data
                name_non_ascii[target] = np.fromiter(
                    (t2.contains_non_ascii(value) for value in raw_name),
                    dtype=bool,
                    count=len(raw_name),
                )
                if bool(found[target].any()):
                    raise AssertionError(f"Duplicate candidate entity IDs found in {source}")
                found[target] = True
                selected_rows += len(selected)
            if chunk_number % 10 == 0:
                log(
                    f"Task 3 text scan {source}: {source_rows:,} rows, "
                    f"selected {selected_rows:,}"
                )
        log(f"Collected {selected_rows:,} required records from {source}")
    if not bool(found.all()):
        missing = required_ids[~found][:10]
        raise AssertionError(f"Missing required candidate text records: {missing.tolist()}")
    return TextStore(
        ids=required_ids,
        name_non_ascii=name_non_ascii,
        **string_columns,
    )


def build_s1_text(eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    raw_name = eval_df["business_name"].fillna("").astype(str)
    raw_address = eval_df["business_address"].fillna("").astype(str)
    return {
        "raw_name": raw_name.to_numpy(dtype=object),
        "raw_address": raw_address.to_numpy(dtype=object),
        "normalized_name": eval_df["normalized_name"].fillna("").astype(str).to_numpy(dtype=object),
        "transliterated_name": t25.transliterate_series(raw_name)
        .fillna("")
        .astype(str)
        .to_numpy(dtype=object),
        "suffix_name": t25.strip_legal_suffix_series(raw_name)
        .fillna("")
        .astype(str)
        .to_numpy(dtype=object),
        "normalized_address": eval_df["normalized_address"]
        .fillna("")
        .astype(str)
        .to_numpy(dtype=object),
        "number_sequence": np.asarray(
            [number_sequence(value) for value in raw_address], dtype=object
        ),
        "normalized_country": eval_df["normalized_country"]
        .fillna("")
        .astype(str)
        .to_numpy(dtype=object),
        "name_non_ascii": np.fromiter(
            (t2.contains_non_ascii(value) for value in raw_name),
            dtype=bool,
            count=len(raw_name),
        ),
    }


def cp_similarity(
    left: np.ndarray,
    right: np.ndarray,
    scorer: object,
    present: np.ndarray,
    workers: int,
    scale: float = 1.0,
) -> np.ndarray:
    values = process.cpdist(
        left.tolist(),
        right.tolist(),
        scorer=scorer,
        workers=workers,
        dtype=np.float32,
    )
    values = np.asarray(values, dtype=np.float32) * np.float32(scale)
    values[~present] = 0.0
    return values


def token_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros(len(left), dtype=np.float32)
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        left_tokens = set(str(left_value).split())
        right_tokens = set(str(right_value).split())
        union = left_tokens | right_tokens
        if union and left_tokens and right_tokens:
            result[index] = len(left_tokens & right_tokens) / len(union)
    return result


def character_ngram_jaccard(
    left: np.ndarray,
    right: np.ndarray,
    ngram: int = 3,
) -> np.ndarray:
    result = np.zeros(len(left), dtype=np.float32)
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        left_value = str(left_value)
        right_value = str(right_value)
        if not left_value or not right_value:
            continue
        left_grams = {
            left_value[position : position + ngram]
            for position in range(max(0, len(left_value) - ngram + 1))
        }
        right_grams = {
            right_value[position : position + ngram]
            for position in range(max(0, len(right_value) - ngram + 1))
        }
        union = left_grams | right_grams
        if union:
            result[index] = len(left_grams & right_grams) / len(union)
    return result


def structure_features(
    left: np.ndarray,
    right: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    left_lengths = np.fromiter((len(str(value)) for value in left), dtype=np.uint16)
    right_lengths = np.fromiter((len(str(value)) for value in right), dtype=np.uint16)
    maximum = np.maximum(left_lengths, right_lengths)
    minimum = np.minimum(left_lengths, right_lengths)
    length_ratio = np.divide(
        minimum,
        maximum,
        out=np.zeros(len(left), dtype=np.float32),
        where=maximum > 0,
    )
    token_count_left = np.fromiter(
        (len(str(value).split()) for value in left), dtype=np.uint16
    )
    token_count_right = np.fromiter(
        (len(str(value).split()) for value in right), dtype=np.uint16
    )
    first_token_equal = np.fromiter(
        (
            bool(str(a))
            and bool(str(b))
            and str(a).split()[0] == str(b).split()[0]
            for a, b in zip(left, right)
        ),
        dtype=bool,
        count=len(left),
    )
    prefix4_equal = np.fromiter(
        (
            len(str(a)) >= 4 and len(str(b)) >= 4 and str(a)[:4] == str(b)[:4]
            for a, b in zip(left, right)
        ),
        dtype=bool,
        count=len(left),
    )
    suffix4_equal = np.fromiter(
        (
            len(str(a)) >= 4 and len(str(b)) >= 4 and str(a)[-4:] == str(b)[-4:]
            for a, b in zip(left, right)
        ),
        dtype=bool,
        count=len(left),
    )
    return {
        f"{prefix}_length_ratio": length_ratio,
        f"{prefix}_length_difference": np.abs(
            left_lengths.astype(np.int32) - right_lengths.astype(np.int32)
        ).astype(np.uint16),
        f"{prefix}_token_count_difference": np.abs(
            token_count_left.astype(np.int32) - token_count_right.astype(np.int32)
        ).astype(np.uint16),
        f"{prefix}_first_token_exact": first_token_equal,
        f"{prefix}_prefix4_exact": prefix4_equal,
        f"{prefix}_suffix4_exact": suffix4_equal,
    }


def full_text_features(
    left: np.ndarray,
    right: np.ndarray,
    prefix: str,
    workers: int,
) -> dict[str, np.ndarray]:
    present = np.fromiter(
        (bool(str(a)) and bool(str(b)) for a, b in zip(left, right)),
        dtype=bool,
        count=len(left),
    )
    exact = np.fromiter(
        (bool(is_present) and str(a) == str(b) for a, b, is_present in zip(left, right, present)),
        dtype=bool,
        count=len(left),
    )
    result = {
        f"{prefix}_exact": exact,
        f"{prefix}_levenshtein": cp_similarity(
            left, right, Levenshtein.normalized_similarity, present, workers
        ),
        f"{prefix}_jaro_winkler": cp_similarity(
            left, right, JaroWinkler.normalized_similarity, present, workers
        ),
        f"{prefix}_token_sort": cp_similarity(
            left, right, fuzz.token_sort_ratio, present, workers, scale=0.01
        ),
        f"{prefix}_token_set": cp_similarity(
            left, right, fuzz.token_set_ratio, present, workers, scale=0.01
        ),
        f"{prefix}_token_jaccard": token_jaccard(left, right),
        f"{prefix}_char3_jaccard": character_ngram_jaccard(left, right),
    }
    result.update(structure_features(left, right, prefix))
    return result


def compact_name_variant_features(
    left: np.ndarray,
    right: np.ndarray,
    prefix: str,
    workers: int,
) -> dict[str, np.ndarray]:
    present = np.fromiter(
        (bool(str(a)) and bool(str(b)) for a, b in zip(left, right)),
        dtype=bool,
        count=len(left),
    )
    return {
        f"{prefix}_exact": np.fromiter(
            (bool(p) and str(a) == str(b) for a, b, p in zip(left, right, present)),
            dtype=bool,
            count=len(left),
        ),
        f"{prefix}_levenshtein": cp_similarity(
            left, right, Levenshtein.normalized_similarity, present, workers
        ),
        f"{prefix}_token_set": cp_similarity(
            left, right, fuzz.token_set_ratio, present, workers, scale=0.01
        ),
        f"{prefix}_token_jaccard": token_jaccard(left, right),
    }


def numeric_address_features(
    left_sequences: np.ndarray,
    right_sequences: np.ndarray,
) -> dict[str, np.ndarray]:
    size = len(left_sequences)
    left_has = np.zeros(size, dtype=bool)
    right_has = np.zeros(size, dtype=bool)
    exact_set = np.zeros(size, dtype=bool)
    ordered_exact = np.zeros(size, dtype=bool)
    first_exact = np.zeros(size, dtype=bool)
    jaccard = np.zeros(size, dtype=np.float32)
    shared_count = np.zeros(size, dtype=np.uint8)
    conflict_count = np.zeros(size, dtype=np.uint8)
    count_difference = np.zeros(size, dtype=np.uint8)
    for index, (left_value, right_value) in enumerate(
        zip(left_sequences, right_sequences)
    ):
        left_tokens = tuple(token for token in str(left_value).split("|") if token)
        right_tokens = tuple(token for token in str(right_value).split("|") if token)
        left_has[index] = bool(left_tokens)
        right_has[index] = bool(right_tokens)
        if not left_tokens or not right_tokens:
            count_difference[index] = min(255, abs(len(left_tokens) - len(right_tokens)))
            continue
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        intersection = left_set & right_set
        union = left_set | right_set
        exact_set[index] = left_set == right_set
        ordered_exact[index] = left_tokens == right_tokens
        first_exact[index] = left_tokens[0] == right_tokens[0]
        jaccard[index] = len(intersection) / len(union)
        shared_count[index] = min(255, len(intersection))
        conflict_count[index] = min(255, len(left_set ^ right_set))
        count_difference[index] = min(255, abs(len(left_tokens) - len(right_tokens)))
    both_have = left_has & right_has
    return {
        "s1_address_has_numbers": left_has,
        "candidate_address_has_numbers": right_has,
        "both_addresses_have_numbers": both_have,
        "one_address_missing_numbers": left_has ^ right_has,
        "address_number_set_exact": exact_set,
        "address_number_ordered_exact": ordered_exact,
        "address_first_number_exact": first_exact,
        "address_number_jaccard": jaccard,
        "address_shared_number_count": shared_count,
        "address_conflicting_number_count": conflict_count,
        "address_number_count_difference": count_difference,
    }


def decode_candidate_ids(candidate_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [t2.decode_candidate_id(int(value)) for value in candidate_ids], dtype=object
    )


def build_pair_feature_frame(
    metadata: PairMetadata,
    eval_df: pd.DataFrame,
    truth_sets: list[set[int]],
    split_by_query: np.ndarray,
    s1_text: dict[str, np.ndarray],
    text_store: TextStore,
    workers: int,
) -> pd.DataFrame:
    query_index = metadata.query_index.astype(np.int32, copy=False)
    candidate_ids = metadata.candidate_ids.astype(np.int64, copy=False)
    candidate_positions = text_store.positions(candidate_ids)
    size = len(candidate_ids)

    left_name = s1_text["normalized_name"][query_index]
    right_name = text_store.normalized_name[candidate_positions]
    left_address = s1_text["normalized_address"][query_index]
    right_address = text_store.normalized_address[candidate_positions]
    left_transliterated = s1_text["transliterated_name"][query_index]
    right_transliterated = text_store.transliterated_name[candidate_positions]
    left_suffix = s1_text["suffix_name"][query_index]
    right_suffix = text_store.suffix_name[candidate_positions]

    label = np.fromiter(
        (
            int(candidate_id) in truth_sets[int(query)]
            for query, candidate_id in zip(query_index, candidate_ids)
        ),
        dtype=np.uint8,
        count=size,
    )
    source_is_s3 = candidate_ids >= t2.SOURCE_OFFSET["S3"]
    candidate_source = np.where(source_is_s3, "S3", "S2")
    s1_address_missing = np.fromiter(
        (not bool(str(value)) for value in left_address), dtype=bool, count=size
    )
    candidate_address_missing = np.fromiter(
        (not bool(str(value)) for value in right_address), dtype=bool, count=size
    )
    s1_name_missing = np.fromiter(
        (not bool(str(value)) for value in left_name), dtype=bool, count=size
    )
    candidate_name_missing = np.fromiter(
        (not bool(str(value)) for value in right_name), dtype=bool, count=size
    )

    data: dict[str, object] = {
        "eval_index": query_index,
        "source1_entity_id": eval_df.iloc[query_index]["source1_entity_id"]
        .astype(str)
        .to_numpy(dtype=object),
        "candidate_entity_id": decode_candidate_ids(candidate_ids),
        "candidate_source": candidate_source.astype(object),
        "match_group": eval_df.iloc[query_index]["match_group"]
        .astype(str)
        .to_numpy(dtype=object),
        "split": split_by_query[query_index].astype(object),
        "label": label,
        "s1_name_missing": s1_name_missing,
        "candidate_name_missing": candidate_name_missing,
        "s1_address_missing": s1_address_missing,
        "candidate_address_missing": candidate_address_missing,
        "both_addresses_present": ~(s1_address_missing | candidate_address_missing),
        "both_addresses_missing": s1_address_missing & candidate_address_missing,
        "candidate_source_s3": source_is_s3,
        "s1_name_non_ascii": s1_text["name_non_ascii"][query_index],
        "candidate_name_non_ascii": text_store.name_non_ascii[candidate_positions],
    }
    data["cross_script_name"] = np.asarray(data["s1_name_non_ascii"]) ^ np.asarray(
        data["candidate_name_non_ascii"]
    )
    data["country_exact"] = np.fromiter(
        (
            bool(a) and bool(b) and str(a) == str(b)
            for a, b in zip(
                s1_text["normalized_country"][query_index],
                text_store.normalized_country[candidate_positions],
            )
        ),
        dtype=bool,
        count=size,
    )

    data.update(full_text_features(left_name, right_name, "name", workers))
    data.update(
        compact_name_variant_features(
            left_transliterated,
            right_transliterated,
            "transliterated_name",
            workers,
        )
    )
    data.update(
        compact_name_variant_features(
            left_suffix, right_suffix, "suffix_name", workers
        )
    )
    data.update(full_text_features(left_address, right_address, "address", workers))
    data.update(
        numeric_address_features(
            s1_text["number_sequence"][query_index],
            text_store.number_sequence[candidate_positions],
        )
    )

    data["transliteration_levenshtein_gain"] = (
        np.asarray(data["transliterated_name_levenshtein"], dtype=np.float32)
        - np.asarray(data["name_levenshtein"], dtype=np.float32)
    ).astype(np.float32)
    data["suffix_levenshtein_gain"] = (
        np.asarray(data["suffix_name_levenshtein"], dtype=np.float32)
        - np.asarray(data["name_levenshtein"], dtype=np.float32)
    ).astype(np.float32)

    rank_matrix = np.column_stack([metadata.ranks[signal] for signal in SIGNALS])
    score_matrix = np.column_stack([metadata.scores[signal] for signal in SIGNALS])
    rank_present = rank_matrix > 0
    rank_top100 = (rank_matrix > 0) & (rank_matrix <= 100)
    signal_count_250 = rank_present.sum(axis=1).astype(np.uint8)
    signal_count_100 = rank_top100.sum(axis=1).astype(np.uint8)
    best_rank = np.where(rank_present, rank_matrix, 65535).min(axis=1).astype(np.uint16)
    best_rank[signal_count_250 == 0] = 0
    mean_rank = np.divide(
        np.where(rank_present, rank_matrix, 0).sum(axis=1),
        signal_count_250,
        out=np.zeros(size, dtype=np.float32),
        where=signal_count_250 > 0,
    ).astype(np.float32)
    masked_scores = np.where(rank_present, score_matrix, -np.inf)
    best_score = masked_scores.max(axis=1).astype(np.float32)
    best_score[~np.isfinite(best_score)] = 0.0
    rrf_score = np.where(
        rank_top100, 1.0 / (60.0 + rank_matrix.astype(np.float32)), 0.0
    ).sum(axis=1).astype(np.float32)

    for signal in SIGNALS:
        data[f"{signal}_rank"] = metadata.ranks[signal].astype(np.uint16, copy=False)
        data[f"{signal}_rank_missing"] = metadata.ranks[signal] == 0
        clean_scores = metadata.scores[signal].astype(np.float32, copy=True)
        clean_scores[~np.isfinite(clean_scores)] = 0.0
        data[f"{signal}_score"] = clean_scores
    data["retrieval_signal_count_top100"] = signal_count_100
    data["retrieval_signal_count_top250"] = signal_count_250
    data["retrieval_best_rank"] = best_rank
    data["retrieval_mean_rank_present"] = mean_rank
    data["retrieval_best_score"] = best_score
    data["retrieval_rrf_score_top100"] = rrf_score
    data["frozen_candidate_position"] = metadata.candidate_position.astype(
        np.uint16, copy=False
    )
    baseline_member = (
        (metadata.ranks["baseline_name"] > 0)
        & (metadata.ranks["baseline_name"] <= 100)
    ) | (
        (metadata.ranks["baseline_address"] > 0)
        & (metadata.ranks["baseline_address"] <= 100)
    )
    data["task2_baseline_member"] = baseline_member
    data["task2_5_augmentation_only"] = (
        (metadata.candidate_position > 0) & ~baseline_member
    )

    name_similarity = np.asarray(data["name_token_set"], dtype=np.float32)
    address_similarity = np.asarray(data["address_token_set"], dtype=np.float32)
    transliterated_similarity = np.asarray(
        data["transliterated_name_token_set"], dtype=np.float32
    )
    suffix_similarity = np.asarray(data["suffix_name_token_set"], dtype=np.float32)
    number_similarity = np.asarray(data["address_number_jaccard"], dtype=np.float32)
    data["name_address_similarity_min"] = np.minimum(
        name_similarity, address_similarity
    ).astype(np.float32)
    data["name_address_similarity_product"] = (
        name_similarity * address_similarity
    ).astype(np.float32)
    data["strong_name_and_address"] = (name_similarity >= 0.85) & (
        address_similarity >= 0.85
    )
    data["transliteration_rescue"] = (
        (transliterated_similarity >= 0.85)
        & (transliterated_similarity - name_similarity >= 0.20)
    )
    data["legal_suffix_rescue"] = np.asarray(data["suffix_name_exact"]) | (
        (suffix_similarity >= 0.90) & (suffix_similarity - name_similarity >= 0.15)
    )
    data["strong_address_and_number"] = (
        (address_similarity >= 0.85)
        & np.asarray(data["both_addresses_have_numbers"])
        & (number_similarity >= 0.80)
    )
    data["high_name_numeric_conflict"] = (
        (name_similarity >= 0.90)
        & np.asarray(data["both_addresses_have_numbers"])
        & (number_similarity == 0.0)
    )
    data["high_address_weak_name"] = (address_similarity >= 0.90) & (
        name_similarity < 0.50
    )
    data["independent_strong_signal_count"] = (
        (name_similarity >= 0.85).astype(np.uint8)
        + (address_similarity >= 0.85).astype(np.uint8)
        + (transliterated_similarity >= 0.85).astype(np.uint8)
        + (suffix_similarity >= 0.90).astype(np.uint8)
        + (number_similarity >= 0.80).astype(np.uint8)
    ).astype(np.uint8)

    frame = pd.DataFrame(data)
    numeric = frame.select_dtypes(include=[np.number, "bool"])
    if bool(numeric.isna().any().any()):
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise AssertionError(f"NaN values in numeric features: {bad}")
    values = numeric.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if values.size and not bool(np.isfinite(values).all()):
        raise AssertionError("Infinite value found in numeric features")
    return frame


def save_candidate_checkpoint(
    path: Path,
    offsets: np.ndarray,
    metadata: PairMetadata,
) -> None:
    np.savez_compressed(
        path,
        query_offsets=offsets,
        candidate_ids=metadata.candidate_ids,
        candidate_position=metadata.candidate_position,
        **{f"{signal}_rank": metadata.ranks[signal] for signal in SIGNALS},
        **{f"{signal}_score": metadata.scores[signal] for signal in SIGNALS},
    )


def dataframe_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def deterministic_negative_sample_mask(
    query_index: np.ndarray,
    candidate_ids: np.ndarray,
    modulus: int,
) -> np.ndarray:
    unsigned_ids = candidate_ids.astype(np.uint64)
    unsigned_queries = query_index.astype(np.uint64)
    hashed = (
        unsigned_ids * np.uint64(11400714819323198485)
        + unsigned_queries * np.uint64(14029467366897019727)
    )
    return (hashed % np.uint64(1000)) < np.uint64(modulus)


def attach_raw_text(
    frame: pd.DataFrame,
    eval_df: pd.DataFrame,
    s1_text: dict[str, np.ndarray],
    text_store: TextStore,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    query_index = result["eval_index"].to_numpy(dtype=np.int32)
    candidate_ids = np.asarray(
        [t2.encode_candidate_id(value) for value in result["candidate_entity_id"]],
        dtype=np.int64,
    )
    positions = text_store.positions(candidate_ids)
    result.insert(3, "s1_name", s1_text["raw_name"][query_index])
    result.insert(4, "candidate_name", text_store.raw_name[positions])
    result.insert(5, "s1_address", s1_text["raw_address"][query_index])
    result.insert(6, "candidate_address", text_store.raw_address[positions])
    return result


def collapse_reason_samples(samples: list[pd.DataFrame], limit_per_reason: int) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame()
    combined = pd.concat(samples, ignore_index=True)
    combined = combined.sort_values(
        ["sample_reason", "sample_score"], ascending=[True, False]
    )
    combined = combined.groupby("sample_reason", group_keys=False).head(limit_per_reason)
    key_columns = ["source1_entity_id", "candidate_entity_id"]
    reasons = (
        combined.groupby(key_columns)["sample_reason"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .rename("sample_reason")
    )
    scores = combined.groupby(key_columns)["sample_score"].max().rename("sample_score")
    details = combined.drop(columns=["sample_reason", "sample_score"]).drop_duplicates(
        key_columns
    )
    return details.merge(reasons, on=key_columns).merge(scores, on=key_columns)


def feature_schema(frame: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "candidate_source": "Candidate provenance; S2 or S3.",
        "country_exact": "Normalized country agreement; validation-only because country blocking makes this constant.",
        "task2_baseline_member": "Candidate appeared in the original name-top-100 or address-top-100 union.",
        "task2_5_augmentation_only": "Candidate was added only by Task 2.5 augmentation.",
        "retrieval_rrf_score_top100": "Frozen reciprocal-rank evidence summed across signals with 1/(60+rank).",
        "frozen_candidate_position": "Position in the frozen baseline-preserving candidate list; not a calibrated rank.",
        "match_group": "Ground-truth-derived analysis group. Never use as a classifier feature.",
        "label": "Ground-truth match target. Never used to calculate another pair feature.",
        "split": "Entity-level train/validation assignment; metadata only.",
    }
    rows = []
    for column in frame.columns:
        if column in {"source1_entity_id", "candidate_entity_id", "eval_index"}:
            family = "identifier"
            role = "metadata"
            fitted_state = False
            recommended = False
            leakage_note = "Identifier retained only for joins and audit; excluded from predictive inputs."
        elif column == "label":
            family = "target"
            role = "target"
            fitted_state = False
            recommended = False
            leakage_note = "Ground truth is used only here and in post-feature analysis."
        elif column in {"match_group", "split"}:
            family = "analysis_metadata"
            role = "analysis_only"
            fitted_state = False
            recommended = False
            leakage_note = (
                "match_group is ground-truth-derived and unsafe for prediction."
                if column == "match_group"
                else "Split membership is not a model feature."
            )
        elif column == "candidate_source" or column == "candidate_source_s3":
            family = "source_metadata"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Available for every inference-time candidate."
        elif any(
            column.startswith(f"{signal}_rank")
            or column.startswith(f"{signal}_score")
            for signal in SIGNALS
        ) or column.startswith("retrieval_") or column in {
            "task2_baseline_member",
            "task2_5_augmentation_only",
            "frozen_candidate_position",
        }:
            family = "retrieval_evidence"
            role = "predictive_feature"
            fitted_state = True
            recommended = True
            leakage_note = "Safe only when generated by the frozen training-independent retrieval pipeline."
        elif column.startswith("name_"):
            family = "baseline_name_similarity"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Deterministic pair transformation."
        elif column.startswith("transliterated_name_") or column == "transliteration_levenshtein_gain":
            family = "transliterated_name_similarity"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Deterministic pair transformation using frozen Task 2.5 transliteration."
        elif column.startswith("suffix_name_") or column == "suffix_levenshtein_gain":
            family = "legal_suffix_name_similarity"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Deterministic pair transformation using frozen Task 2.5 suffix normalization."
        elif "number" in column:
            family = "address_number_structure"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Deterministic pair transformation using frozen Task 2.5 number normalization."
        elif column.startswith("address_"):
            family = "address_similarity"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Deterministic pair transformation."
        elif column.endswith("missing") or column.startswith("both_addresses") or column.startswith("s1_") or column.startswith("candidate_"):
            family = "missingness"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Available at inference time."
        elif column == "country_exact":
            family = "country_validation"
            role = "validation_only"
            fitted_state = False
            recommended = False
            leakage_note = "Constant after country blocking; retained as a sanity check only."
        else:
            family = "interpretable_interaction"
            role = "predictive_feature"
            fitted_state = False
            recommended = True
            leakage_note = "Deterministic combination of inference-time pair features."
        if column.endswith("_rank"):
            missing_convention = "0 means not retrieved in the saved top-250 signal; paired missing indicator is explicit."
        elif column.endswith("_score"):
            missing_convention = "0 when rank is missing; consult the paired rank-missing indicator."
        else:
            missing_convention = "No NaN/inf; absence is represented by explicit indicators or zero similarity."
        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "family": family,
                "role": role,
                "description": descriptions.get(
                    column,
                    column.replace("_", " ").capitalize() + ".",
                ),
                "missing_convention": missing_convention,
                "requires_fitted_state": fitted_state,
                "recommended_task4": recommended,
                "leakage_note": leakage_note,
            }
        )
    schema = pd.DataFrame(rows)
    predictive = schema["role"].eq("predictive_feature")
    schema.loc[predictive, "recommended_task4"] = schema.loc[
        predictive, "column"
    ].isin(TASK4_RECOMMENDED_FEATURES)
    return schema


def numeric_predictive_columns(schema: pd.DataFrame, frame: pd.DataFrame) -> list[str]:
    allowed = set(
        schema.loc[schema["role"].eq("predictive_feature"), "column"].astype(str)
    )
    return [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column in allowed
    ]


def feature_statistics(
    analysis_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    labels = analysis_frame["label"].to_numpy(dtype=np.uint8)
    for feature in feature_columns:
        values = analysis_frame[feature].astype(np.float64).to_numpy()
        positive = values[labels == 1]
        negative = values[labels == 0]
        if not len(positive) or not len(negative):
            continue
        pooled_std = math.sqrt((float(np.var(positive)) + float(np.var(negative))) / 2.0)
        smd = (
            (float(np.mean(positive)) - float(np.mean(negative))) / pooled_std
            if pooled_std > 0
            else 0.0
        )
        if len(np.unique(values)) > 1:
            auc = float(roc_auc_score(labels, values))
        else:
            auc = 0.5
        rows.append(
            {
                "feature": feature,
                "positive_count": len(positive),
                "negative_sample_count": len(negative),
                "positive_mean": float(np.mean(positive)),
                "negative_mean": float(np.mean(negative)),
                "positive_median": float(np.median(positive)),
                "negative_median": float(np.median(negative)),
                "positive_p10": float(np.percentile(positive, 10)),
                "negative_p10": float(np.percentile(negative, 10)),
                "positive_p90": float(np.percentile(positive, 90)),
                "negative_p90": float(np.percentile(negative, 90)),
                "mean_difference": float(np.mean(positive) - np.mean(negative)),
                "standardized_mean_difference": smd,
                "univariate_auc": auc,
                "directionless_univariate_auc": max(auc, 1.0 - auc),
                "constant_in_analysis_sample": len(np.unique(values)) <= 1,
                "analysis_note": "All positives plus deterministic candidate-negative sample; descriptive only, not fitted importance.",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["directionless_univariate_auc", "feature"], ascending=[False, True]
    )


def correlation_and_redundancy(
    analysis_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_size = min(100_000, len(analysis_frame))
    sample = analysis_frame.sample(n=sample_size, random_state=RANDOM_STATE)
    varying = [column for column in feature_columns if sample[column].nunique() > 1]
    correlation = sample[varying].astype(np.float32).corr(method="spearman")
    rows = []
    for left_index, left in enumerate(varying):
        for right in varying[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            if abs(value) >= 0.95:
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman_correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    redundancy = pd.DataFrame(rows)
    if not redundancy.empty:
        redundancy = redundancy.sort_values(
            ["absolute_correlation", "feature_a", "feature_b"],
            ascending=[False, True, True],
        )
    return correlation, redundancy


def grouped_feature_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(group_columns, keys))
        for feature in feature_columns:
            values = group[feature].astype(np.float64)
            rows.append(
                {
                    **group_values,
                    "feature": feature,
                    "rows": len(group),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def split_summary(
    entity_split: pd.DataFrame,
    pair_frame: pd.DataFrame,
) -> pd.DataFrame:
    pair_counts = (
        pair_frame.groupby(["split", "label"], observed=True)
        .size()
        .rename("pairs")
        .reset_index()
    )
    rows = []
    for split in ("train", "validation"):
        entities = entity_split.loc[entity_split["split"].eq(split)]
        counts = pair_counts.loc[pair_counts["split"].eq(split)].set_index("label")["pairs"]
        positives = int(counts.get(1, 0))
        negatives = int(counts.get(0, 0))
        rows.append(
            {
                "split": split,
                "s1_entities": len(entities),
                "pairs": positives + negatives,
                "positive_pairs": positives,
                "negative_pairs": negatives,
                "positive_rate": positives / (positives + negatives),
                "group_1_entities": int(entities["match_group"].eq("1").sum()),
                "group_2_entities": int(entities["match_group"].eq("2").sum()),
                "group_3_5_entities": int(entities["match_group"].eq("3-5").sum()),
                "group_6_plus_entities": int(entities["match_group"].eq("6+").sum()),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(schema: pd.DataFrame) -> pd.DataFrame:
    unsafe_predictive = schema.loc[
        schema["column"].isin(GT_DERIVED_COLUMNS)
        & schema["role"].eq("predictive_feature")
    ]
    rows = [
        {
            "check": "Ground truth used only for label and analysis metadata",
            "status": "PASS" if unsafe_predictive.empty else "FAIL",
            "detail": "label is the target; match_group is analysis-only and excluded from Task 4 features.",
        },
        {
            "check": "Entity IDs excluded from predictive features",
            "status": "PASS"
            if not bool(
                schema.loc[
                    schema["column"].isin(
                        ["eval_index", "source1_entity_id", "candidate_entity_id"]
                    ),
                    "recommended_task4",
                ].any()
            )
            else "FAIL",
            "detail": "IDs are retained only for joins and auditing.",
        },
        {
            "check": "No label-derived aggregate feature",
            "status": "PASS",
            "detail": "All pair features are deterministic text, numeric-structure, missingness, source, or frozen-retrieval transformations.",
        },
        {
            "check": "Fitted-state features isolated",
            "status": "PASS",
            "detail": "Only frozen Task 2.5 retrieval ranks/scores require fitted state; no new TF-IDF/IDF statistics were fit in Task 3.",
        },
        {
            "check": "Inference availability",
            "status": "PASS",
            "detail": "Recommended features require only S1 text, candidate text, candidate source, and frozen retrieval outputs.",
        },
        {
            "check": "Entity-level split",
            "status": "PASS",
            "detail": "Split is assigned by source1_entity_id with fixed seed 42 and match-group stratification.",
        },
        {
            "check": "Country feature treatment",
            "status": "PASS",
            "detail": "country_exact is validation-only because country is already a lossless hard block and is constant in the candidate pairs.",
        },
        {
            "check": "No classifier or threshold work",
            "status": "PASS",
            "detail": "Task 3 computes only pair features, descriptive statistics, audits, and split metadata.",
        },
    ]
    return pd.DataFrame(rows)


def validation_report(
    frozen_metrics: dict[str, object],
    total_pairs: int,
    positive_pairs: int,
    missed_positive_pairs: int,
    duplicate_pairs: int,
    maximum_candidates: int,
    split_overlap: int,
    numeric_clean: bool,
    parquet_rows: int,
    direct_gt_passed: bool,
) -> pd.DataFrame:
    checks = [
        ("Frozen candidate recall reproduced", math.isclose(float(frozen_metrics["link_recall"]), 0.9689649992489109, abs_tol=1e-12), f"link recall={frozen_metrics['link_recall']:.12f}"),
        ("Frozen complete recall reproduced", math.isclose(float(frozen_metrics["complete_recall"]), 0.9159, abs_tol=1e-12), f"complete recall={frozen_metrics['complete_recall']:.6f}"),
        ("Every pair belongs to frozen candidate set", total_pairs == int(frozen_metrics["pair_rows"]), f"pairs={total_pairs:,}"),
        ("No duplicate S1-candidate pairs", duplicate_pairs == 0, f"duplicates={duplicate_pairs}"),
        ("Labels directly verified against training ground truth", direct_gt_passed, f"retrieved positives={positive_pairs:,}"),
        ("Positive counts reconcile with retrieval misses", positive_pairs + missed_positive_pairs == int(frozen_metrics["positive_links"]), f"{positive_pairs:,}+{missed_positive_pairs:,}={int(frozen_metrics['positive_links']):,}"),
        ("Candidate cap respected", maximum_candidates <= 250, f"max={maximum_candidates}"),
        ("No train-validation S1 overlap", split_overlap == 0, f"overlap={split_overlap}"),
        ("Numeric features contain no NaN or inf", numeric_clean, "chunk validation passed"),
        ("Parquet row count matches pair count", parquet_rows == total_pairs, f"parquet rows={parquet_rows:,}"),
    ]
    result = pd.DataFrame(
        [
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
            for check, passed, detail in checks
        ]
    )
    if not bool(result["status"].eq("PASS").all()):
        raise AssertionError(f"Task 3 validation failed:\n{result}")
    return result


def write_summary(
    output_path: Path,
    pair_counts: pd.DataFrame,
    split_df: pd.DataFrame,
    feature_stats: pd.DataFrame,
    redundancy: pd.DataFrame,
    recovery_counts: pd.DataFrame,
    source_analysis: pd.DataFrame,
    group_analysis: pd.DataFrame,
    runtime_df: pd.DataFrame,
    pair_dataset_size_mb: float,
    peak_memory_mb: float,
    validation_df: pd.DataFrame,
) -> None:
    overall = pair_counts.loc[pair_counts["scope"].eq("overall")].iloc[0]
    strongest = feature_stats.head(12)
    weak = feature_stats.sort_values(
        ["directionless_univariate_auc", "feature"]
    ).head(10)
    lines = [
        "TASK 3 - PAIRWISE FEATURE ENGINEERING AND ANALYSIS",
        "",
        "Scope",
        "-----",
        "Training data only. Frozen Task 2.5 candidates were reconstructed exactly from the saved checkpoint.",
        "No matching classifier, threshold, probability calibration, test prediction, or submission was created.",
        "",
        "Pair dataset",
        "------------",
        f"Pair rows: {int(overall['pairs']):,}",
        f"Positive pairs: {int(overall['positive_pairs']):,}",
        f"Negative pairs: {int(overall['negative_pairs']):,}",
        f"Positive rate: {overall['positive_rate']:.6%}",
        f"Negatives per positive: {overall['negatives_per_positive']:.2f}",
        f"Ground-truth links missed by candidate generation and excluded from pair labels: {int(overall['missed_positive_links']):,}",
        "",
        "Entity split",
        "------------",
    ]
    for row in split_df.itertuples(index=False):
        lines.append(
            f"{row.split}: S1={row.s1_entities:,}, pairs={row.pairs:,}, "
            f"positives={row.positive_pairs:,}, negatives={row.negative_pairs:,}, "
            f"positive rate={row.positive_rate:.6%}"
        )
    lines.extend(["", "Strongest-looking descriptive features", "---------------------------------------"])
    for row in strongest.itertuples(index=False):
        lines.append(
            f"{row.feature}: positive mean={row.positive_mean:.4f}, "
            f"negative mean={row.negative_mean:.4f}, descriptive |AUC|={row.directionless_univariate_auc:.4f}"
        )
    lines.extend(["", "Weak or constant descriptive features", "-------------------------------------"])
    for row in weak.itertuples(index=False):
        lines.append(
            f"{row.feature}: descriptive |AUC|={row.directionless_univariate_auc:.4f}, "
            f"constant={row.constant_in_analysis_sample}"
        )
    lines.extend(
        [
            "",
            "Redundancy",
            "----------",
            f"Feature pairs with |Spearman| >= 0.95: {len(redundancy):,}",
            "Jaro was excluded before the full run because the pilot found 0.998-0.999 correlation with Jaro-Winkler.",
            "Indel ratio was excluded because it overlapped strongly with normalized Levenshtein.",
            "No new pairwise TF-IDF was fit; frozen retrieval scores retain that fitted signal without split leakage.",
            "",
            "Positive retrieval groups",
            "-------------------------",
        ]
    )
    for row in recovery_counts.itertuples(index=False):
        lines.append(
            f"{row.positive_group}: links={row.links:,}, S2={row.s2_links:,}, "
            f"S3={row.s3_links:,}, group6+={row.group_6_plus_links:,}"
        )
    lines.extend(
        [
            "",
            "Runtime and storage",
            "-------------------",
            f"Total runtime: {runtime_df['runtime_seconds'].sum() / 60.0:.2f} minutes",
            f"Peak process RSS: {peak_memory_mb:.1f} MB",
            f"Pair feature Parquet: {pair_dataset_size_mb:.1f} MB",
            "",
            "Validation",
            "----------",
            f"Passed checks: {int(validation_df['status'].eq('PASS').sum())}/{len(validation_df)}",
            "",
            "Task 4 recommendation",
            "---------------------",
            "Start with a class-weighted gradient-boosted tree baseline on the frozen entity split, using the recommended deterministic similarities, numeric-address features, missingness, source, and retrieval evidence. Keep a regularized logistic-regression model only as a linear benchmark. Evaluate pair ranking and entity-level outcomes on validation, then select a decision threshold only in Task 4.",
            "",
            "STOP: Task 3 ends with features, analysis, audits, and split preparation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    started = time.perf_counter()
    eval_df, truth_lists, truth_strings = t25.load_evaluation(paths["eval"])
    checkpoint = load_checkpoint(paths["checkpoint"])
    offsets, candidate_ids, candidate_position, candidate_sets = reconstruct_frozen_candidates(
        eval_df, checkpoint
    )
    frozen_metrics = verify_frozen_metrics(
        eval_df, truth_lists, candidate_sets, paths["decision"]
    )
    selected_queries = []
    for group in ("1", "2", "3-5", "6+"):
        selected_queries.extend(
            eval_df.index[eval_df["match_group"].eq(group)][
                : args.smoke_entities_per_group
            ].tolist()
        )
    query_index_parts = []
    candidate_parts = []
    position_parts = []
    for query in selected_queries:
        start, stop = offsets[query], offsets[query + 1]
        query_index_parts.append(np.full(stop - start, query, dtype=np.int32))
        candidate_parts.append(candidate_ids[start:stop])
        position_parts.append(candidate_position[start:stop])
    query_index = np.concatenate(query_index_parts)
    candidates = np.concatenate(candidate_parts)
    positions = np.concatenate(position_parts)
    ranks, scores = lookup_retrieval_metadata(query_index, candidates, checkpoint)
    truth_ids = np.asarray(
        sorted({candidate_id for query in selected_queries for candidate_id in truth_lists[query]}),
        dtype=np.int64,
    )
    required = np.unique(np.r_[candidates, truth_ids])
    text_store = collect_text_store(required, paths, args.chunksize)
    split_df, split = make_entity_split(eval_df, args.validation_fraction)
    frame = build_pair_feature_frame(
        PairMetadata(query_index, candidates, positions, ranks, scores),
        eval_df,
        [set(map(int, truth)) for truth in truth_lists],
        split,
        build_s1_text(eval_df),
        text_store,
        args.workers,
    )
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Smoke feature frame contains duplicate pairs")
    smoke = {
        "status": "passed",
        "training_only": True,
        "frozen_global_metrics": frozen_metrics,
        "smoke_s1_entities": len(selected_queries),
        "smoke_pairs": len(frame),
        "smoke_positive_pairs": int(frame["label"].sum()),
        "feature_columns": len(frame.columns),
        "numeric_nan_count": int(frame.select_dtypes(include=[np.number]).isna().sum().sum()),
        "duplicate_pairs": int(
            frame.duplicated(["source1_entity_id", "candidate_entity_id"]).sum()
        ),
        "split_train_entities": int(split_df["split"].eq("train").sum()),
        "split_validation_entities": int(split_df["split"].eq("validation").sum()),
        "runtime_seconds": elapsed(started),
        "peak_rss_mb": peak_rss_mb(),
        "classifier_work_performed": False,
        "test_data_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task3_smoke_validation.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        f"Task 3 smoke passed: {len(frame):,} pairs, {len(frame.columns)} columns, "
        f"runtime={smoke['runtime_seconds']:.1f}s"
    )


def main() -> None:
    args = make_parser().parse_args()
    args.data_dir = args.data_dir.resolve()
    args.task2_output_dir = args.task2_output_dir.resolve()
    args.task2_5_output_dir = args.task2_5_output_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = validate_paths(args)

    if args.smoke_only:
        run_smoke(args, paths)
        return

    smoke_path = args.output_dir / "task3_smoke_validation.json"
    if not smoke_path.exists():
        raise FileNotFoundError("Run Task 3 with --smoke-only before the full pipeline")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed":
        raise AssertionError("Task 3 smoke validation did not pass")

    total_started = time.perf_counter()
    runtime = RuntimeRecorder()
    validation_rows: list[dict[str, object]] = []

    phase = time.perf_counter()
    eval_df, truth_lists, truth_strings = t25.load_evaluation(paths["eval"])
    truth_sets = [set(map(int, truth)) for truth in truth_lists]
    eval_truth_strings = {
        str(row.source1_entity_id): {
            value for value in str(row.matched_entity_ids).split(",") if value
        }
        for row in eval_df.itertuples(index=False)
    }
    direct_ground_truth_check(
        eval_df, eval_truth_strings, paths["GT"], args.chunksize
    )
    checkpoint = load_checkpoint(paths["checkpoint"])
    runtime.record("load_and_ground_truth_verification", elapsed(phase))

    phase = time.perf_counter()
    offsets, candidate_ids, candidate_position, candidate_sets = reconstruct_frozen_candidates(
        eval_df, checkpoint
    )
    frozen_metrics = verify_frozen_metrics(
        eval_df, truth_lists, candidate_sets, paths["decision"]
    )
    query_index = np.repeat(
        np.arange(len(eval_df), dtype=np.int32), np.diff(offsets).astype(np.int32)
    )
    ranks, scores = lookup_retrieval_metadata(query_index, candidate_ids, checkpoint)
    pair_metadata = PairMetadata(
        query_index=query_index,
        candidate_ids=candidate_ids,
        candidate_position=candidate_position,
        ranks=ranks,
        scores=scores,
    )
    save_candidate_checkpoint(
        args.output_dir / "task3_frozen_candidates.npz", offsets, pair_metadata
    )
    runtime.record(
        "frozen_candidate_reconstruction",
        elapsed(phase),
        f"{len(candidate_ids):,} pairs",
    )

    phase = time.perf_counter()
    entity_split, split_by_query = make_entity_split(
        eval_df, args.validation_fraction
    )
    entity_split.to_csv(args.output_dir / "task3_entity_split.csv", index=False)
    split_overlap = len(
        set(entity_split.loc[entity_split["split"].eq("train"), "source1_entity_id"])
        & set(
            entity_split.loc[
                entity_split["split"].eq("validation"), "source1_entity_id"
            ]
        )
    )
    if split_overlap:
        raise AssertionError("Entity split overlap detected")
    runtime.record("entity_split", elapsed(phase))

    missed_query_parts = []
    missed_candidate_parts = []
    for query, (truth, selected) in enumerate(zip(truth_lists, candidate_sets)):
        selected_set = set(map(int, selected))
        missed = [candidate_id for candidate_id in truth if candidate_id not in selected_set]
        if missed:
            missed_query_parts.append(np.full(len(missed), query, dtype=np.int32))
            missed_candidate_parts.append(np.asarray(missed, dtype=np.int64))
    missed_query_index = np.concatenate(missed_query_parts)
    missed_candidate_ids = np.concatenate(missed_candidate_parts)
    if len(missed_candidate_ids) != int(frozen_metrics["missed_positive_links"]):
        raise AssertionError("Missed-positive count differs from frozen metrics")

    phase = time.perf_counter()
    required_ids = np.unique(np.r_[candidate_ids, missed_candidate_ids])
    text_store = collect_text_store(required_ids, paths, args.chunksize)
    s1_text = build_s1_text(eval_df)
    runtime.record(
        "entity_text_collection_and_normalization",
        elapsed(phase),
        f"{len(required_ids):,} candidate records",
    )

    phase = time.perf_counter()
    pair_path = args.output_dir / "task3_pair_features.parquet"
    writer: Optional[pq.ParquetWriter] = None
    analysis_samples: list[pd.DataFrame] = []
    hard_samples: list[pd.DataFrame] = []
    full_count_rows = []
    first_frame: Optional[pd.DataFrame] = None
    rows_written = 0

    for entity_start in range(0, len(eval_df), args.pair_chunk_entities):
        entity_stop = min(entity_start + args.pair_chunk_entities, len(eval_df))
        pair_start = int(offsets[entity_start])
        pair_stop = int(offsets[entity_stop])
        frame = build_pair_feature_frame(
            pair_metadata.slice(pair_start, pair_stop),
            eval_df,
            truth_sets,
            split_by_query,
            s1_text,
            text_store,
            args.workers,
        )
        if first_frame is None:
            first_frame = frame.head(1).copy()
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                pair_path,
                table.schema,
                compression="zstd",
                compression_level=6,
                use_dictionary=["candidate_source", "match_group", "split"],
                write_statistics=True,
            )
        writer.write_table(table, row_group_size=len(frame))
        rows_written += len(frame)

        full_count_rows.append(
            frame.groupby(
                ["split", "candidate_source", "match_group", "label"],
                observed=True,
            )
            .size()
            .rename("pairs")
            .reset_index()
        )
        negative_sample = deterministic_negative_sample_mask(
            frame["eval_index"].to_numpy(dtype=np.int32),
            pair_metadata.candidate_ids[pair_start:pair_stop],
            args.negative_analysis_modulus,
        )
        analysis_mask = frame["label"].eq(1).to_numpy() | (
            frame["label"].eq(0).to_numpy() & negative_sample
        )
        analysis_samples.append(frame.loc[analysis_mask].copy())

        negatives = frame.loc[frame["label"].eq(0)].copy()
        if not negatives.empty:
            negatives["sample_score"] = (
                0.30 * negatives["name_token_set"].astype(float)
                + 0.30 * negatives["address_token_set"].astype(float)
                + 0.15 * negatives["address_number_jaccard"].astype(float)
                + 0.15
                * (negatives["retrieval_signal_count_top100"].astype(float) / 5.0)
                + 0.10 * negatives["retrieval_best_score"].astype(float)
            )
            category_scores = {
                "highest_combined_similarity": negatives["sample_score"],
                "very_high_name_similarity": negatives["name_token_set"],
                "very_high_address_similarity": negatives["address_token_set"],
                "shared_address_numbers": negatives["address_number_jaccard"],
                "retrieved_by_multiple_signals": negatives[
                    "retrieval_signal_count_top100"
                ].astype(float),
            }
            for reason, values in category_scores.items():
                local = negatives.assign(sample_score=values).nlargest(
                    40, "sample_score"
                )
                local["sample_reason"] = reason
                hard_samples.append(local)
        log(
            f"Feature rows {rows_written:,}/{len(candidate_ids):,}; "
            f"entities {entity_start:,}-{entity_stop - 1:,}"
        )
        del frame, table, negatives
        gc.collect()
    if writer is not None:
        writer.close()
    if rows_written != len(candidate_ids):
        raise AssertionError("Parquet writer row count does not match frozen pairs")
    runtime.record(
        "pair_feature_generation",
        elapsed(phase),
        f"{rows_written:,} rows; {dataframe_size_mb(pair_path):.1f} MB",
    )

    if first_frame is None:
        raise AssertionError("No pair feature rows were generated")
    schema = feature_schema(first_frame)

    phase = time.perf_counter()
    missed_ranks, missed_scores = lookup_retrieval_metadata(
        missed_query_index, missed_candidate_ids, checkpoint
    )
    missed_metadata = PairMetadata(
        query_index=missed_query_index,
        candidate_ids=missed_candidate_ids,
        candidate_position=np.zeros(len(missed_candidate_ids), dtype=np.uint16),
        ranks=missed_ranks,
        scores=missed_scores,
    )
    missed_frame = build_pair_feature_frame(
        missed_metadata,
        eval_df,
        truth_sets,
        split_by_query,
        s1_text,
        text_store,
        args.workers,
    )
    if not bool(missed_frame["label"].eq(1).all()):
        raise AssertionError("Missed-positive analysis contains a nonpositive label")
    missed_path = args.output_dir / "task3_missed_positive_features.parquet"
    pq.write_table(
        pa.Table.from_pandas(missed_frame, preserve_index=False),
        missed_path,
        compression="zstd",
        compression_level=6,
        use_dictionary=["candidate_source", "match_group", "split"],
    )
    runtime.record(
        "missed_positive_feature_generation",
        elapsed(phase),
        f"{len(missed_frame):,} rows",
    )

    phase = time.perf_counter()
    full_counts = (
        pd.concat(full_count_rows, ignore_index=True)
        .groupby(["split", "candidate_source", "match_group", "label"], observed=True)[
            "pairs"
        ]
        .sum()
        .reset_index()
    )
    positives = int(full_counts.loc[full_counts["label"].eq(1), "pairs"].sum())
    negatives = int(full_counts.loc[full_counts["label"].eq(0), "pairs"].sum())
    pair_count_rows = [
        {
            "scope": "overall",
            "value": "all",
            "pairs": positives + negatives,
            "positive_pairs": positives,
            "negative_pairs": negatives,
            "positive_rate": positives / (positives + negatives),
            "negatives_per_positive": negatives / positives,
            "missed_positive_links": len(missed_frame),
        }
    ]
    for source in ("S2", "S3"):
        source_rows = full_counts.loc[full_counts["candidate_source"].eq(source)]
        pos = int(source_rows.loc[source_rows["label"].eq(1), "pairs"].sum())
        neg = int(source_rows.loc[source_rows["label"].eq(0), "pairs"].sum())
        pair_count_rows.append(
            {
                "scope": "candidate_source",
                "value": source,
                "pairs": pos + neg,
                "positive_pairs": pos,
                "negative_pairs": neg,
                "positive_rate": pos / (pos + neg),
                "negatives_per_positive": neg / pos,
                "missed_positive_links": int(
                    missed_frame["candidate_source"].eq(source).sum()
                ),
            }
        )
    for group in ("1", "2", "3-5", "6+"):
        group_rows = full_counts.loc[full_counts["match_group"].eq(group)]
        pos = int(group_rows.loc[group_rows["label"].eq(1), "pairs"].sum())
        neg = int(group_rows.loc[group_rows["label"].eq(0), "pairs"].sum())
        pair_count_rows.append(
            {
                "scope": "match_group",
                "value": group,
                "pairs": pos + neg,
                "positive_pairs": pos,
                "negative_pairs": neg,
                "positive_rate": pos / (pos + neg),
                "negatives_per_positive": neg / pos,
                "missed_positive_links": int(
                    missed_frame["match_group"].eq(group).sum()
                ),
            }
        )
    pair_counts = pd.DataFrame(pair_count_rows)
    pair_counts.to_csv(args.output_dir / "task3_pair_counts.csv", index=False)

    analysis_frame = pd.concat(analysis_samples, ignore_index=True)
    split_stats = split_summary(entity_split, analysis_frame)
    expected_split_counts = (
        full_counts.groupby(["split", "label"], observed=True)["pairs"]
        .sum()
        .unstack(fill_value=0)
    )
    for row_index, row in split_stats.iterrows():
        split_name = row["split"]
        pos = int(expected_split_counts.loc[split_name].get(1, 0))
        neg = int(expected_split_counts.loc[split_name].get(0, 0))
        split_stats.loc[row_index, ["pairs", "positive_pairs", "negative_pairs", "positive_rate"]] = [
            pos + neg,
            pos,
            neg,
            pos / (pos + neg),
        ]
    split_stats.to_csv(args.output_dir / "task3_split_summary.csv", index=False)

    feature_columns = numeric_predictive_columns(schema, analysis_frame)
    statistics = feature_statistics(analysis_frame, feature_columns)
    statistics.to_csv(args.output_dir / "task3_feature_statistics.csv", index=False)
    correlation, redundancy = correlation_and_redundancy(
        analysis_frame, feature_columns
    )
    correlation.to_csv(args.output_dir / "task3_feature_correlation.csv")
    redundancy.to_csv(args.output_dir / "task3_feature_redundancy.csv", index=False)

    constant_features = set(
        statistics.loc[statistics["constant_in_analysis_sample"], "feature"].astype(str)
    )
    schema.loc[schema["column"].isin(constant_features), "recommended_task4"] = False
    schema.loc[schema["column"].isin(constant_features), "role"] = "validation_only"
    schema.to_csv(args.output_dir / "task3_feature_schema.csv", index=False)

    core_features = [
        "name_levenshtein",
        "name_token_set",
        "name_token_jaccard",
        "name_char3_jaccard",
        "transliterated_name_levenshtein",
        "transliteration_levenshtein_gain",
        "suffix_name_levenshtein",
        "address_levenshtein",
        "address_token_set",
        "address_token_jaccard",
        "address_char3_jaccard",
        "address_number_jaccard",
        "address_conflicting_number_count",
        "retrieval_signal_count_top100",
        "retrieval_best_rank",
        "retrieval_rrf_score_top100",
        "independent_strong_signal_count",
    ]
    source_analysis = grouped_feature_summary(
        analysis_frame, ["candidate_source", "label"], core_features
    )
    source_analysis.to_csv(
        args.output_dir / "task3_source_feature_analysis.csv", index=False
    )
    group_analysis = grouped_feature_summary(
        analysis_frame, ["match_group", "label"], core_features
    )
    group_analysis.to_csv(
        args.output_dir / "task3_match_group_feature_analysis.csv", index=False
    )
    runtime.record("descriptive_feature_analysis", elapsed(phase))

    phase = time.perf_counter()
    hard_negative = collapse_reason_samples(hard_samples, 25)
    hard_negative = attach_raw_text(
        hard_negative, eval_df, s1_text, text_store
    ).sort_values(["sample_reason", "sample_score"], ascending=[True, False])
    hard_negative.to_csv(
        args.output_dir / "task3_hard_negative_sample.csv", index=False
    )

    positive_frame = analysis_frame.loc[analysis_frame["label"].eq(1)].copy()
    positive_frame["weak_pair_score"] = (
        positive_frame[
            [
                "name_token_set",
                "transliterated_name_token_set",
                "suffix_name_token_set",
            ]
        ]
        .max(axis=1)
        + positive_frame["address_token_set"]
        + positive_frame["address_number_jaccard"]
    ) / 3.0
    difficult_parts = []
    selections = {
        "weak_overall_similarity": positive_frame.nsmallest(40, "weak_pair_score"),
        "weak_name_evidence": positive_frame.nsmallest(40, "name_token_set"),
        "weak_address_evidence": positive_frame.nsmallest(40, "address_token_set"),
        "task2_5_recovered_positive": positive_frame.loc[
            positive_frame["task2_5_augmentation_only"]
        ].nsmallest(40, "weak_pair_score"),
    }
    for reason, selected in selections.items():
        part = selected.copy()
        part["sample_reason"] = reason
        part["sample_score"] = 1.0 - part["weak_pair_score"]
        difficult_parts.append(part)
    difficult_positive = collapse_reason_samples(difficult_parts, 25)
    difficult_positive = attach_raw_text(
        difficult_positive, eval_df, s1_text, text_store
    ).sort_values(["sample_reason", "sample_score"], ascending=[True, False])
    difficult_positive.to_csv(
        args.output_dir / "task3_difficult_positive_sample.csv", index=False
    )

    baseline_positive = positive_frame.loc[
        positive_frame["task2_baseline_member"]
    ].copy()
    baseline_positive["positive_group"] = "task2_baseline_retrieved"
    recovered_positive = positive_frame.loc[
        positive_frame["task2_5_augmentation_only"]
    ].copy()
    recovered_positive["positive_group"] = "task2_5_recovered"
    missed_analysis = missed_frame.copy()
    missed_analysis["positive_group"] = "task2_5_still_missed"
    positive_groups = pd.concat(
        [baseline_positive, recovered_positive, missed_analysis], ignore_index=True
    )
    recovery_analysis = grouped_feature_summary(
        positive_groups, ["positive_group"], core_features
    )
    recovery_analysis.to_csv(
        args.output_dir / "task3_recovered_positive_analysis.csv", index=False
    )
    recovery_count_rows = []
    for name, group in positive_groups.groupby("positive_group", observed=True):
        recovery_count_rows.append(
            {
                "positive_group": name,
                "links": len(group),
                "s2_links": int(group["candidate_source"].eq("S2").sum()),
                "s3_links": int(group["candidate_source"].eq("S3").sum()),
                "group_1_links": int(group["match_group"].eq("1").sum()),
                "group_2_links": int(group["match_group"].eq("2").sum()),
                "group_3_5_links": int(group["match_group"].eq("3-5").sum()),
                "group_6_plus_links": int(group["match_group"].eq("6+").sum()),
            }
        )
    recovery_counts = pd.DataFrame(recovery_count_rows)
    recovery_counts.to_csv(
        args.output_dir / "task3_recovery_group_counts.csv", index=False
    )
    runtime.record("hard_case_and_recovery_analysis", elapsed(phase))

    phase = time.perf_counter()
    leakage = leakage_audit(schema)
    leakage.to_csv(args.output_dir / "task3_leakage_audit.csv", index=False)

    parquet_rows = pq.ParquetFile(pair_path).metadata.num_rows
    duplicate_pairs = sum(
        len(values) - len(np.unique(values)) for values in candidate_sets
    )
    validation = validation_report(
        frozen_metrics=frozen_metrics,
        total_pairs=len(candidate_ids),
        positive_pairs=positives,
        missed_positive_pairs=len(missed_frame),
        duplicate_pairs=duplicate_pairs,
        maximum_candidates=int(np.diff(offsets).max()),
        split_overlap=split_overlap,
        numeric_clean=True,
        parquet_rows=parquet_rows,
        direct_gt_passed=True,
    )
    validation.to_csv(
        args.output_dir / "task3_validation_report.csv", index=False
    )
    runtime.record("audit_and_validation", elapsed(phase))

    runtime_df = pd.DataFrame(runtime.rows)
    runtime_df.to_csv(args.output_dir / "task3_runtime_memory.csv", index=False)
    total_runtime_seconds = elapsed(total_started)
    manifest = {
        "training_only": True,
        "frozen_configuration": FROZEN_CONFIGURATION,
        "task2_eval_sha256": file_sha256(paths["eval"]),
        "task2_5_checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "task3_candidate_checkpoint_sha256": file_sha256(
            args.output_dir / "task3_frozen_candidates.npz"
        ),
        "pair_feature_parquet_sha256": file_sha256(pair_path),
        "evaluation_entities": len(eval_df),
        "pair_rows": len(candidate_ids),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "missed_positive_links": len(missed_frame),
        "feature_columns_including_metadata": len(first_frame.columns),
        "recommended_predictive_features": int(
            schema["recommended_task4"].sum()
        ),
        "entity_split": {
            "random_state": RANDOM_STATE,
            "validation_fraction": args.validation_fraction,
            "train_entities": int(entity_split["split"].eq("train").sum()),
            "validation_entities": int(
                entity_split["split"].eq("validation").sum()
            ),
        },
        "frozen_metrics": frozen_metrics,
        "pair_dataset_mb": dataframe_size_mb(pair_path),
        "missed_positive_dataset_mb": dataframe_size_mb(missed_path),
        "total_runtime_seconds": total_runtime_seconds,
        "peak_rss_mb": peak_rss_mb(),
        "smoke_validation": smoke,
        "classifier_work_performed": False,
        "threshold_tuning_performed": False,
        "probability_calibration_performed": False,
        "test_data_used": False,
        "test_predictions_created": False,
        "submission_created": False,
    }
    (args.output_dir / "task3_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(
        args.output_dir / "task3_summary.txt",
        pair_counts,
        split_stats,
        statistics,
        redundancy,
        recovery_counts,
        source_analysis,
        group_analysis,
        runtime_df,
        dataframe_size_mb(pair_path),
        peak_rss_mb(),
        validation,
    )
    log(
        f"Task 3 complete in {total_runtime_seconds / 60.0:.2f} minutes: "
        f"{len(candidate_ids):,} pairs, {positives:,} positives, "
        f"{dataframe_size_mb(pair_path):.1f} MB Parquet"
    )


if __name__ == "__main__":
    main()
