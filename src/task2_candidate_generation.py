#!/usr/bin/env python3
"""Task 2: scalable candidate generation and recall evaluation on training data only."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import resource
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection


RANDOM_STATE = 42
SOURCE_OFFSET = {"S2": 0, "S3": 1_000_000_000}
SOURCE_FILES = {
    "S2": "train_source2.tsv",
    "S3": "train_source3.tsv",
}
FIELDS = ("name", "address")
FIELD_COLUMNS = {"name": "business_name", "address": "business_address"}
LEGAL_SUFFIXES = {
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "plc",
    "llp",
    "private",
    "pvt",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def normalize_series(series: pd.Series) -> pd.Series:
    result = series.astype("string").fillna("")
    result = result.str.normalize("NFKC").str.casefold()
    result = result.str.replace("_", " ", regex=False)
    result = result.str.replace(r"[^\w\s]", " ", regex=True)
    result = result.str.replace(r"\s+", " ", regex=True).str.strip()
    return result


def normalize_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_country_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


class CountryEncoder:
    def __init__(self) -> None:
        self.mapping: dict[str, int] = {}
        self.reverse: dict[int, str] = {0: ""}

    def encode(self, normalized: pd.Series) -> np.ndarray:
        for value in pd.unique(normalized):
            value = str(value)
            if value and value not in self.mapping:
                code = len(self.mapping) + 1
                self.mapping[value] = code
                self.reverse[code] = value
        return normalized.map(self.mapping).fillna(0).to_numpy(dtype=np.uint16)


def numeric_id_series(entity_ids: pd.Series, expected_source: str) -> np.ndarray:
    prefix = f"{expected_source}-"
    valid = entity_ids.astype("string").str.startswith(prefix)
    if not bool(valid.all()):
        bad = entity_ids.loc[~valid].head(5).tolist()
        raise ValueError(f"Unexpected {expected_source} entity IDs: {bad}")
    values = entity_ids.astype("string").str.slice(len(prefix)).astype("int64").to_numpy()
    if expected_source in SOURCE_OFFSET:
        if values.size and int(values.max()) >= 1_000_000_000:
            raise ValueError("Numeric entity ID exceeds the configured source offset")
        values = values + SOURCE_OFFSET[expected_source]
    return values


def encode_candidate_id(entity_id: str) -> int:
    source = entity_id[:2]
    if source not in SOURCE_OFFSET or not entity_id.startswith(f"{source}-"):
        raise ValueError(f"Unexpected candidate entity ID: {entity_id}")
    return SOURCE_OFFSET[source] + int(entity_id[3:])


def decode_candidate_id(encoded_id: int) -> str:
    if encoded_id >= SOURCE_OFFSET["S3"]:
        return f"S3-{encoded_id - SOURCE_OFFSET['S3']}"
    return f"S2-{encoded_id}"


def candidate_source(encoded_id: int) -> str:
    return "S3" if encoded_id >= SOURCE_OFFSET["S3"] else "S2"


def match_group(count: int) -> str:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if 3 <= count <= 5:
        return "3-5"
    if count >= 6:
        return "6+"
    return "0"


def parse_truth_ids(value: object) -> list[int]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    return [encode_candidate_id(item.strip()) for item in str(value).split(",") if item.strip()]


def current_peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


@dataclass
class TextModel:
    vectorizer: TfidfVectorizer
    projector: SparseRandomProjection
    train_texts: list[str]
    vocabulary_size: int


@dataclass
class RetrievalArtifacts:
    ids: np.ndarray
    scores: np.ndarray
    build_seconds: float
    query_seconds: float
    ntotal: int
    estimated_index_mb: float


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2_outputs"))
    parser.add_argument("--eval-size", type=int, default=10_000)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--sample-modulus", type=int, default=30)
    parser.add_argument("--train-sample-per-source-country", type=int, default=50_000)
    parser.add_argument("--max-features", type=int, default=32_768)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--nlist", type=int, default=2_048)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--search-k", type=int, default=1_000)
    parser.add_argument("--transform-batch-size", type=int, default=50_000)
    parser.add_argument("--faiss-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--failure-sample-size", type=int, default=50)
    return parser


def validate_input_files(data_dir: Path) -> dict[str, Path]:
    paths = {
        "S1": data_dir / "train_source1.tsv",
        "S2": data_dir / "train_source2.tsv",
        "S3": data_dir / "train_source3.tsv",
        "GT": data_dir / "train_ground_truth.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training files: {missing}")
    if any("test" in path.name.casefold() for path in paths.values()):
        raise ValueError("Task 2 must use training data only")
    return paths


def load_ground_truth(gt_path: Path, eval_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    gt = pd.read_csv(
        gt_path,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype="string",
    )
    matched = gt["matched_entity_ids"].fillna("")
    gt["match_count"] = matched.str.count(",") + matched.ne("").astype(np.int16)
    gt["match_group"] = gt["match_count"].map(match_group)

    groups = ["1", "2", "3-5", "6+"]
    base = eval_size // len(groups)
    allocations = {group: base for group in groups}
    for group in groups[: eval_size - base * len(groups)]:
        allocations[group] += 1

    samples = []
    for group in groups:
        pool = gt.loc[gt["match_group"].eq(group)]
        requested = allocations[group]
        if len(pool) < requested:
            raise ValueError(f"Only {len(pool):,} entities available in match group {group}")
        samples.append(pool.sample(n=requested, random_state=RANDOM_STATE))

    eval_gt = (
        pd.concat(samples, ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    eval_gt.insert(0, "eval_index", np.arange(len(eval_gt), dtype=np.int32))
    log(
        f"Loaded GT {gt.shape}; positive links={int(gt['match_count'].sum()):,}; "
        f"evaluation entities={len(eval_gt):,} in {elapsed(started):.1f}s"
    )
    return gt, eval_gt


def scan_source1(
    s1_path: Path,
    eval_gt: pd.DataFrame,
    encoder: CountryEncoder,
    chunksize: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    eval_ids = set(eval_gt["source1_entity_id"].astype(str))
    sample_chunks: list[pd.DataFrame] = []
    all_ids: list[np.ndarray] = []
    all_countries: list[np.ndarray] = []

    reader = pd.read_csv(
        s1_path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype="string",
        chunksize=chunksize,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        country_norm = normalize_country_series(chunk["country"])
        all_ids.append(numeric_id_series(chunk["entity_id"], "S1"))
        all_countries.append(encoder.encode(country_norm))
        selected = chunk.loc[chunk["entity_id"].isin(eval_ids)].copy()
        if not selected.empty:
            sample_chunks.append(selected)
        if chunk_number % 5 == 0:
            log(f"S1 scan: {chunk_number * chunksize:,} rows processed")

    s1_ids = np.concatenate(all_ids)
    s1_country_codes = np.concatenate(all_countries)
    sample_records = pd.concat(sample_chunks, ignore_index=True)
    if sample_records["entity_id"].nunique() != len(eval_gt):
        missing = eval_ids - set(sample_records["entity_id"].astype(str))
        raise ValueError(f"Missing {len(missing)} sampled S1 records")

    eval_df = eval_gt.merge(
        sample_records,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["entity_id"])
    eval_df["normalized_name"] = normalize_series(eval_df["business_name"])
    eval_df["normalized_address"] = normalize_series(eval_df["business_address"])
    eval_df["normalized_country"] = normalize_country_series(eval_df["country"])
    log(f"Scanned S1 and collected evaluation records in {elapsed(started):.1f}s")
    return eval_df, s1_ids, s1_country_codes


def map_gt_to_s1_countries(
    gt: pd.DataFrame,
    s1_ids: np.ndarray,
    s1_country_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    order = np.argsort(s1_ids)
    sorted_ids = s1_ids[order]
    sorted_countries = s1_country_codes[order]
    gt_s1_ids = numeric_id_series(gt["source1_entity_id"], "S1")
    positions = np.searchsorted(sorted_ids, gt_s1_ids)
    found = (positions < len(sorted_ids)) & (
        sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == gt_s1_ids
    )
    if not bool(found.all()):
        raise ValueError(f"Could not locate {int((~found).sum()):,} GT S1 IDs in source 1")
    gt_country_codes = sorted_countries[positions]

    counts = gt["match_count"].to_numpy(dtype=np.int16)
    link_s1_ids = np.repeat(gt_s1_ids, counts)
    link_s1_countries = np.repeat(gt_country_codes, counts)
    exploded = gt["matched_entity_ids"].str.split(",").explode(ignore_index=True).dropna()
    exploded = exploded.loc[exploded.astype("string").str.strip().ne("")]
    source_digit = exploded.astype("string").str.slice(1, 2).astype("int8").to_numpy()
    numeric = exploded.astype("string").str.slice(3).astype("int64").to_numpy()
    link_candidate_ids = numeric + np.where(source_digit == 3, SOURCE_OFFSET["S3"], 0)
    link_sources = np.where(source_digit == 3, 3, 2).astype(np.uint8)
    if len(link_candidate_ids) != int(counts.sum()):
        raise ValueError("Exploded ground-truth link count does not match match_count")
    log(f"Expanded {len(link_candidate_ids):,} positive links in {elapsed(started):.1f}s")
    return link_candidate_ids, link_s1_ids, link_s1_countries, link_sources


def preliminary_candidate_scan(
    paths: dict[str, Path],
    eval_truth_strings: set[str],
    encoder: CountryEncoder,
    chunksize: int,
    sample_modulus: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[int, dict[str, str]], dict[str, int]]:
    started = time.perf_counter()
    candidate_ids: list[np.ndarray] = []
    candidate_countries: list[np.ndarray] = []
    sample_chunks: list[pd.DataFrame] = []
    true_records: dict[int, dict[str, str]] = {}
    source_counts: dict[str, int] = {}

    for source in ("S2", "S3"):
        source_started = time.perf_counter()
        row_count = 0
        reader = pd.read_csv(
            paths[source],
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype="string",
            chunksize=chunksize,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            ids = numeric_id_series(chunk["entity_id"], source)
            country_norm = normalize_country_series(chunk["country"])
            country_codes = encoder.encode(country_norm)
            candidate_ids.append(ids)
            candidate_countries.append(country_codes)
            row_count += len(chunk)

            hashes = pd.util.hash_pandas_object(chunk["entity_id"], index=False).to_numpy(
                dtype=np.uint64
            )
            sample_mask = hashes % np.uint64(sample_modulus) == 0
            if bool(sample_mask.any()):
                sampled = chunk.loc[sample_mask].copy()
                sampled["_source"] = source
                sampled["_global_id"] = ids[sample_mask]
                sampled["_hash"] = hashes[sample_mask]
                sampled["normalized_country"] = country_norm.loc[sample_mask].to_numpy()
                sample_chunks.append(sampled)

            truth_mask = chunk["entity_id"].isin(eval_truth_strings).to_numpy()
            if bool(truth_mask.any()):
                for row, global_id in zip(
                    chunk.loc[truth_mask].itertuples(index=False), ids[truth_mask]
                ):
                    true_records[int(global_id)] = {
                        "entity_id": str(row.entity_id),
                        "business_name": "" if pd.isna(row.business_name) else str(row.business_name),
                        "business_address": ""
                        if pd.isna(row.business_address)
                        else str(row.business_address),
                        "country": "" if pd.isna(row.country) else str(row.country),
                        "source": source,
                    }

            if chunk_number % 5 == 0:
                log(f"Preliminary {source} scan: {row_count:,} rows processed")

        source_counts[source] = row_count
        log(f"Preliminary {source} scan complete in {elapsed(source_started):.1f}s")

    sample_df = pd.concat(sample_chunks, ignore_index=True)
    all_candidate_ids = np.concatenate(candidate_ids)
    all_candidate_countries = np.concatenate(candidate_countries)
    log(
        f"Preliminary candidate scan complete: {len(all_candidate_ids):,} records, "
        f"{len(sample_df):,} deterministic sample rows in {elapsed(started):.1f}s"
    )
    return (
        all_candidate_ids,
        all_candidate_countries,
        sample_df,
        true_records,
        source_counts,
    )


def validate_country_links(
    link_candidate_ids: np.ndarray,
    link_s1_ids: np.ndarray,
    link_s1_countries: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_countries: np.ndarray,
    encoder: CountryEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    started = time.perf_counter()
    order = np.argsort(candidate_ids)
    sorted_ids = candidate_ids[order]
    sorted_countries = candidate_countries[order]
    duplicate_candidate_records = int(np.count_nonzero(sorted_ids[1:] == sorted_ids[:-1]))

    positions = np.searchsorted(sorted_ids, link_candidate_ids)
    safe_positions = np.minimum(positions, len(sorted_ids) - 1)
    found = (positions < len(sorted_ids)) & (sorted_ids[safe_positions] == link_candidate_ids)
    matched_country = np.zeros(len(link_candidate_ids), dtype=np.uint16)
    matched_country[found] = sorted_countries[positions[found]]

    missing_country = found & (
        (link_s1_countries == 0) | (matched_country == 0)
    )
    comparable = found & (link_s1_countries != 0) & (matched_country != 0)
    same_country = comparable & (link_s1_countries == matched_country)
    different_country = comparable & (link_s1_countries != matched_country)
    lookup_missing = ~found
    total = len(link_candidate_ids)

    summary = pd.DataFrame(
        [
            {
                "total_positive_links": total,
                "same_country_links": int(same_country.sum()),
                "different_country_links": int(different_country.sum()),
                "missing_country_links": int(missing_country.sum()),
                "missing_candidate_lookup_links": int(lookup_missing.sum()),
                "same_country_percentage": float(same_country.mean() * 100.0),
                "different_country_percentage": float(different_country.mean() * 100.0),
                "missing_country_percentage": float(missing_country.mean() * 100.0),
                "missing_candidate_lookup_percentage": float(lookup_missing.mean() * 100.0),
                "duplicate_candidate_record_ids": duplicate_candidate_records,
            }
        ]
    )

    issue_indices = np.flatnonzero(different_country | missing_country | lookup_missing)[:50]
    examples = []
    for index in issue_indices:
        examples.append(
            {
                "s1_entity_id": f"S1-{int(link_s1_ids[index])}",
                "matched_entity_id": decode_candidate_id(int(link_candidate_ids[index])),
                "s1_country_normalized": encoder.reverse.get(int(link_s1_countries[index]), ""),
                "matched_country_normalized": encoder.reverse.get(int(matched_country[index]), ""),
                "issue": "candidate_lookup_missing"
                if lookup_missing[index]
                else "country_missing"
                if missing_country[index]
                else "country_different",
            }
        )
    examples_df = pd.DataFrame(
        examples,
        columns=[
            "s1_entity_id",
            "matched_entity_id",
            "s1_country_normalized",
            "matched_country_normalized",
            "issue",
        ],
    )
    country_block_validated = (
        int(different_country.sum()) == 0
        and int(missing_country.sum()) == 0
        and int(lookup_missing.sum()) == 0
    )
    log(
        f"Country validation complete in {elapsed(started):.1f}s: "
        f"same={int(same_country.sum()):,}, different={int(different_country.sum()):,}, "
        f"missing={int(missing_country.sum()):,}, lookup-missing={int(lookup_missing.sum()):,}"
    )
    return summary, examples_df, country_block_validated


def trim_training_sample(
    sample_df: pd.DataFrame,
    per_source_country: int,
) -> pd.DataFrame:
    parts = []
    for (_, _), group in sample_df.groupby(["_source", "normalized_country"], sort=True):
        parts.append(group.nsmallest(min(per_source_country, len(group)), "_hash"))
    trimmed = pd.concat(parts, ignore_index=True)
    trimmed["normalized_name"] = normalize_series(trimmed["business_name"])
    trimmed["normalized_address"] = normalize_series(trimmed["business_address"])
    return trimmed


def partition_series(country_series: pd.Series, country_block: bool) -> pd.Series:
    if country_block:
        return country_series
    return pd.Series("__all__", index=country_series.index, dtype="string")


def build_text_models(
    sample_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    country_block: bool,
    max_features: int,
    projection_dim: int,
) -> tuple[dict[tuple[str, str], TextModel], list[str]]:
    started = time.perf_counter()
    sample_df = sample_df.copy()
    eval_df = eval_df.copy()
    sample_df["partition"] = partition_series(sample_df["normalized_country"], country_block)
    eval_df["partition"] = partition_series(eval_df["normalized_country"], country_block)
    partitions = sorted(eval_df["partition"].dropna().astype(str).unique().tolist())
    models: dict[tuple[str, str], TextModel] = {}

    for partition in partitions:
        partition_sample = sample_df.loc[sample_df["partition"].eq(partition)]
        if partition_sample.empty:
            raise ValueError(f"No training sample available for partition {partition}")
        for field in FIELDS:
            values = partition_sample[f"normalized_{field}"].astype(str)
            values = values.loc[values.ne("")]
            if len(values) < 1_000:
                raise ValueError(f"Too few nonempty {field} samples in partition {partition}")
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=False,
                min_df=2,
                max_features=max_features,
                sublinear_tf=True,
                norm="l2",
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform(values.tolist())
            projector = SparseRandomProjection(
                n_components=projection_dim,
                dense_output=True,
                random_state=RANDOM_STATE,
            )
            projector.fit(matrix[: min(len(values), 5_000)])
            models[(partition, field)] = TextModel(
                vectorizer=vectorizer,
                projector=projector,
                train_texts=values.tolist(),
                vocabulary_size=len(vectorizer.vocabulary_),
            )
            log(
                f"Fitted {partition}/{field} TF-IDF: {len(values):,} sample strings, "
                f"{len(vectorizer.vocabulary_):,} features"
            )
            del matrix
            gc.collect()

    log(f"Fitted {len(models)} text models in {elapsed(started):.1f}s")
    return models, partitions


def project_texts(model: TextModel, texts: Iterable[str]) -> np.ndarray:
    matrix = model.vectorizer.transform(list(texts))
    projected = model.projector.transform(matrix)
    if sparse.issparse(projected):
        projected = projected.toarray()
    projected = np.ascontiguousarray(projected, dtype=np.float32)
    norms = np.linalg.norm(projected, axis=1)
    nonzero = norms > 0
    projected[nonzero] /= norms[nonzero, None]
    return projected


def create_faiss_index(
    model: TextModel,
    projection_dim: int,
    requested_nlist: int,
    pq_m: int,
    nprobe: int,
) -> tuple[faiss.IndexIVFPQ, float]:
    started = time.perf_counter()
    if projection_dim % pq_m != 0:
        raise ValueError("projection_dim must be divisible by pq_m")
    train_texts = model.train_texts
    train_vectors = project_texts(model, train_texts)
    train_vectors = train_vectors[np.linalg.norm(train_vectors, axis=1) > 0]
    max_nlist = max(1, len(train_vectors) // 39)
    nlist = min(requested_nlist, max_nlist)
    quantizer = faiss.IndexFlatIP(projection_dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        projection_dim,
        nlist,
        pq_m,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.cp.niter = 12
    index.pq.cp.niter = 12
    index.train(train_vectors)
    index.nprobe = min(nprobe, nlist)
    duration = elapsed(started)
    log(
        f"Trained FAISS IVFPQ: train={len(train_vectors):,}, nlist={nlist}, "
        f"nprobe={index.nprobe}, m={pq_m} in {duration:.1f}s"
    )
    del train_vectors
    gc.collect()
    return index, duration


def prepare_exact_query_keys(
    eval_df: pd.DataFrame,
    country_block: bool,
) -> dict[tuple[str, str], set[str]]:
    partitions = partition_series(eval_df["normalized_country"], country_block)
    keys: dict[tuple[str, str], set[str]] = {}
    for field in FIELDS:
        for partition, group in eval_df.assign(partition=partitions).groupby("partition"):
            values = set(group[f"normalized_{field}"].astype(str))
            values.discard("")
            keys[(str(partition), field)] = values
    return keys


def add_to_exact_map(
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    partition: str,
    field: str,
    normalized_values: pd.Series,
    ids: np.ndarray,
    query_keys: dict[tuple[str, str], set[str]],
) -> None:
    keys = query_keys.get((partition, field), set())
    if not keys:
        return
    mask = normalized_values.isin(keys).to_numpy()
    if not bool(mask.any()):
        return
    selected = pd.DataFrame(
        {
            "value": normalized_values.loc[mask].astype(str).to_numpy(),
            "global_id": ids[mask],
        }
    )
    target = exact_map[(partition, field)]
    for value, group in selected.groupby("value", sort=False):
        target[value].extend(group["global_id"].astype("int64").tolist())


def build_field_index_and_retrieve(
    field: str,
    paths: dict[str, Path],
    eval_df: pd.DataFrame,
    models: dict[tuple[str, str], TextModel],
    partitions: list[str],
    country_block: bool,
    query_keys: dict[tuple[str, str], set[str]],
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    chunksize: int,
    transform_batch_size: int,
    projection_dim: int,
    nlist: int,
    pq_m: int,
    nprobe: int,
    search_k: int,
    collect_exact: bool,
) -> RetrievalArtifacts:
    total_started = time.perf_counter()
    indexes: dict[str, faiss.IndexIVFPQ] = {}
    training_seconds = 0.0
    for partition in partitions:
        index, duration = create_faiss_index(
            models[(partition, field)], projection_dim, nlist, pq_m, nprobe
        )
        indexes[partition] = index
        training_seconds += duration

    indexing_started = time.perf_counter()
    for source in ("S2", "S3"):
        source_rows = 0
        reader = pd.read_csv(
            paths[source],
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype="string",
            chunksize=chunksize,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            ids = numeric_id_series(chunk["entity_id"], source)
            country_norm = normalize_country_series(chunk["country"])
            partition_values = partition_series(country_norm, country_block)
            normalized_field = normalize_series(chunk[FIELD_COLUMNS[field]])

            if collect_exact:
                other_field = "address" if field == "name" else "name"
                normalized_other = normalize_series(chunk[FIELD_COLUMNS[other_field]])
            else:
                normalized_other = None

            for partition in partitions:
                partition_mask = partition_values.eq(partition).to_numpy()
                if not bool(partition_mask.any()):
                    continue
                partition_ids = ids[partition_mask]
                partition_values_field = normalized_field.loc[partition_mask].reset_index(drop=True)

                if collect_exact:
                    add_to_exact_map(
                        exact_map,
                        partition,
                        field,
                        partition_values_field,
                        partition_ids,
                        query_keys,
                    )
                    other_values = normalized_other.loc[partition_mask].reset_index(drop=True)
                    add_to_exact_map(
                        exact_map,
                        partition,
                        other_field,
                        other_values,
                        partition_ids,
                        query_keys,
                    )

                nonempty = partition_values_field.ne("").to_numpy()
                if not bool(nonempty.any()):
                    continue
                add_ids = partition_ids[nonempty]
                add_values = partition_values_field.loc[nonempty].tolist()
                model = models[(partition, field)]
                index = indexes[partition]
                for start in range(0, len(add_values), transform_batch_size):
                    stop = min(start + transform_batch_size, len(add_values))
                    vectors = project_texts(model, add_values[start:stop])
                    vector_nonzero = np.linalg.norm(vectors, axis=1) > 0
                    if bool(vector_nonzero.any()):
                        index.add_with_ids(
                            vectors[vector_nonzero],
                            np.ascontiguousarray(add_ids[start:stop][vector_nonzero], dtype=np.int64),
                        )
                    del vectors

            source_rows += len(chunk)
            if chunk_number % 5 == 0:
                totals = ", ".join(
                    f"{partition}:{indexes[partition].ntotal:,}" for partition in partitions
                )
                log(f"{field} index {source}: {source_rows:,} rows; indexed [{totals}]")
            del chunk, normalized_field
            if normalized_other is not None:
                del normalized_other
            gc.collect()

    indexing_seconds = elapsed(indexing_started)
    build_seconds = training_seconds + indexing_seconds
    log(f"Built {field} indexes in {build_seconds:.1f}s")

    query_started = time.perf_counter()
    result_ids = np.full((len(eval_df), search_k), -1, dtype=np.int64)
    result_scores = np.full((len(eval_df), search_k), -np.inf, dtype=np.float32)
    eval_partitions = partition_series(eval_df["normalized_country"], country_block)
    for partition in partitions:
        query_positions = np.flatnonzero(eval_partitions.eq(partition).to_numpy())
        query_values = eval_df.loc[query_positions, f"normalized_{field}"].astype(str)
        nonempty = query_values.ne("").to_numpy()
        if not bool(nonempty.any()):
            continue
        vectors = project_texts(models[(partition, field)], query_values.loc[nonempty].tolist())
        distances, labels = indexes[partition].search(vectors, search_k)
        destination = query_positions[nonempty]
        result_ids[destination, :] = labels
        result_scores[destination, :] = distances
        del vectors, distances, labels

    query_seconds = elapsed(query_started)
    ntotal = sum(index.ntotal for index in indexes.values())
    estimated_index_mb = ntotal * (pq_m + 8) / (1024.0 * 1024.0)
    log(
        f"Queried {field} indexes for {len(eval_df):,} entities at top-{search_k} "
        f"in {query_seconds:.1f}s"
    )
    del indexes
    gc.collect()
    return RetrievalArtifacts(
        ids=result_ids,
        scores=result_scores,
        build_seconds=build_seconds,
        query_seconds=query_seconds,
        ntotal=ntotal,
        estimated_index_mb=estimated_index_mb,
    )


def exact_candidates_for_query(
    row: pd.Series,
    field: str,
    country_block: bool,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
) -> list[int]:
    partition = str(row["normalized_country"]) if country_block else "__all__"
    value = str(row[f"normalized_{field}"])
    if not value:
        return []
    return exact_map[(partition, field)].get(value, [])


def clean_ranked_ids(values: np.ndarray, k: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values[:k]:
        value = int(value)
        if value < 0 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def all_signal_ranked(
    name_ids: list[int],
    address_ids: list[int],
    exact_name: list[int],
    exact_address: list[int],
    k: int,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for rank, candidate_id in enumerate(name_ids, start=1):
        scores[candidate_id] += 1.0 / (60.0 + rank)
    for rank, candidate_id in enumerate(address_ids, start=1):
        scores[candidate_id] += 1.0 / (60.0 + rank)
    for candidate_id in exact_name:
        scores[int(candidate_id)] += 2.0
    for candidate_id in exact_address:
        scores[int(candidate_id)] += 2.0
    ranked = sorted(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id))
    return ranked[:k]


def strategy_candidates(
    strategy: str,
    k: Optional[int],
    query_index: int,
    eval_df: pd.DataFrame,
    country_block: bool,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    name_ids: np.ndarray,
    address_ids: np.ndarray,
) -> list[int]:
    row = eval_df.iloc[query_index]
    exact_name = exact_candidates_for_query(row, "name", country_block, exact_map)
    exact_address = exact_candidates_for_query(row, "address", country_block, exact_map)
    if strategy == "Exact normalized name":
        return list(dict.fromkeys(exact_name))
    if strategy == "Exact normalized address":
        return list(dict.fromkeys(exact_address))
    if strategy == "Exact union":
        return list(dict.fromkeys([*exact_name, *exact_address]))
    if k is None:
        raise ValueError(f"Strategy {strategy} requires K")
    ranked_name = clean_ranked_ids(name_ids[query_index], k)
    ranked_address = clean_ranked_ids(address_ids[query_index], k)
    if strategy == "Name TF-IDF":
        return ranked_name
    if strategy == "Address TF-IDF":
        return ranked_address
    if strategy == "Name + address union":
        return list(dict.fromkeys([*ranked_name, *ranked_address]))
    if strategy == "All signals union":
        return list(
            dict.fromkeys([*ranked_name, *ranked_address, *exact_name, *exact_address])
        )
    if strategy == "All signals ranked":
        return all_signal_ranked(
            ranked_name,
            ranked_address,
            exact_name,
            exact_address,
            k,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_configuration(
    strategy: str,
    k: Optional[int],
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    country_block: bool,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    name_ids: np.ndarray,
    address_ids: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    started = time.perf_counter()
    total_links = 0
    total_hits = 0
    complete_entities = 0
    candidate_counts: list[int] = []
    source_totals = Counter()
    source_hits = Counter()
    group_totals = Counter()
    group_hits = Counter()
    group_entities = Counter()
    group_complete = Counter()

    for index, truth in enumerate(truth_lists):
        candidates = strategy_candidates(
            strategy,
            k,
            index,
            eval_df,
            country_block,
            exact_map,
            name_ids,
            address_ids,
        )
        candidate_set = set(candidates)
        hits = sum(candidate_id in candidate_set for candidate_id in truth)
        group = str(eval_df.iloc[index]["match_group"])
        total_links += len(truth)
        total_hits += hits
        candidate_counts.append(len(candidate_set))
        complete = hits == len(truth)
        complete_entities += int(complete)
        group_totals[group] += len(truth)
        group_hits[group] += hits
        group_entities[group] += 1
        group_complete[group] += int(complete)
        for candidate_id in truth:
            source = candidate_source(candidate_id)
            source_totals[source] += 1
            source_hits[source] += int(candidate_id in candidate_set)

    evaluation_seconds = elapsed(started)
    global_row = {
        "strategy": strategy,
        "k": "" if k is None else k,
        "positive_links": total_links,
        "retrieved_positive_links": total_hits,
        "link_recall": total_hits / total_links if total_links else 0.0,
        "complete_recall": complete_entities / len(truth_lists) if truth_lists else 0.0,
        "average_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "p95_candidates": percentile(candidate_counts, 95),
        "max_candidates": max(candidate_counts, default=0),
        "evaluation_seconds": evaluation_seconds,
    }
    source_rows = [
        {
            "strategy": strategy,
            "k": "" if k is None else k,
            "source": source,
            "positive_links": source_totals[source],
            "retrieved_positive_links": source_hits[source],
            "link_recall": source_hits[source] / source_totals[source]
            if source_totals[source]
            else 0.0,
        }
        for source in ("S2", "S3")
    ]
    group_rows = [
        {
            "strategy": strategy,
            "k": "" if k is None else k,
            "match_group": group,
            "entities": group_entities[group],
            "positive_links": group_totals[group],
            "retrieved_positive_links": group_hits[group],
            "link_recall": group_hits[group] / group_totals[group]
            if group_totals[group]
            else 0.0,
            "complete_entities": group_complete[group],
            "complete_recall": group_complete[group] / group_entities[group]
            if group_entities[group]
            else 0.0,
        }
        for group in ("1", "2", "3-5", "6+")
    ]
    return global_row, source_rows, group_rows


def evaluate_strategies(
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    country_block: bool,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    name_artifacts: RetrievalArtifacts,
    address_artifacts: RetrievalArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configurations: list[tuple[str, Optional[int]]] = [
        ("Exact normalized name", None),
        ("Exact normalized address", None),
        ("Exact union", None),
    ]
    for k in (5, 10, 20, 50, 100):
        configurations.extend(
            [
                ("Name TF-IDF", k),
                ("Address TF-IDF", k),
                ("Name + address union", k),
                ("All signals ranked", k),
                ("All signals union", k),
            ]
        )

    global_rows = []
    source_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for strategy, k in configurations:
        row, source_part, group_part = evaluate_configuration(
            strategy,
            k,
            eval_df,
            truth_lists,
            country_block,
            exact_map,
            name_artifacts.ids,
            address_artifacts.ids,
        )
        if strategy == "Name TF-IDF":
            row["retrieval_precompute_seconds"] = (
                name_artifacts.build_seconds + name_artifacts.query_seconds
            )
        elif strategy == "Address TF-IDF":
            row["retrieval_precompute_seconds"] = (
                address_artifacts.build_seconds + address_artifacts.query_seconds
            )
        elif strategy in {
            "Name + address union",
            "All signals ranked",
            "All signals union",
        }:
            row["retrieval_precompute_seconds"] = (
                name_artifacts.build_seconds
                + name_artifacts.query_seconds
                + address_artifacts.build_seconds
                + address_artifacts.query_seconds
            )
        else:
            row["retrieval_precompute_seconds"] = 0.0
        global_rows.append(row)
        source_rows.extend(source_part)
        group_rows.extend(group_part)
        log(
            f"Evaluated {strategy} K={k}: recall={row['link_recall']:.4f}, "
            f"complete={row['complete_recall']:.4f}, avg candidates={row['average_candidates']:.1f}"
        )

    return pd.DataFrame(global_rows), pd.DataFrame(group_rows), pd.DataFrame(source_rows)


def sparse_pair_cosine(model: TextModel, left: str, right: str) -> Optional[float]:
    if not left or not right:
        return None
    matrix = model.vectorizer.transform([left, right])
    return float(matrix[0].multiply(matrix[1]).sum())


def rank_in_results(results: np.ndarray, candidate_id: int) -> Optional[int]:
    positions = np.flatnonzero(results == candidate_id)
    return int(positions[0] + 1) if len(positions) else None


def strip_legal_suffix(text: str) -> str:
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def contains_non_ascii(text: str) -> bool:
    return any(ord(character) > 127 for character in text)


def digit_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def failure_tags(
    s1_name: str,
    true_name: str,
    s1_address: str,
    true_address: str,
    name_similarity: Optional[float],
    address_similarity: Optional[float],
    exact_name_count: int,
    exact_address_count: int,
) -> str:
    tags: list[str] = []
    if not s1_address or not true_address:
        tags.append("missing_address")
    if contains_non_ascii(s1_name) != contains_non_ascii(true_name):
        tags.append("script_or_transliteration_difference")
    if digit_tokens(s1_name) != digit_tokens(true_name):
        tags.append("name_number_difference")
    if digit_tokens(s1_address) != digit_tokens(true_address):
        tags.append("address_number_difference")
    if s1_name != true_name and sorted(s1_name.split()) == sorted(true_name.split()):
        tags.append("name_word_reordering")
    if s1_name != true_name and strip_legal_suffix(s1_name) == strip_legal_suffix(true_name):
        tags.append("legal_suffix_variation")
    if name_similarity is not None and name_similarity < 0.20:
        tags.append("very_different_name")
    if address_similarity is not None and address_similarity < 0.20:
        tags.append("very_different_address")
    if exact_name_count > 100:
        tags.append("common_exact_name")
    if exact_address_count > 100:
        tags.append("common_exact_address")
    if not tags:
        tags.append("lexical_rank_beyond_budget")
    return ";".join(tags)


def choose_failure_configuration(strategy_df: pd.DataFrame) -> tuple[str, Optional[int]]:
    preferred = strategy_df.loc[
        strategy_df["strategy"].isin(["All signals union", "All signals ranked"])
    ].copy()
    preferred["k_numeric"] = pd.to_numeric(preferred["k"], errors="coerce").fillna(-1)
    preferred = preferred.sort_values(
        ["link_recall", "complete_recall", "average_candidates", "k_numeric"],
        ascending=[False, False, True, True],
    )
    row = preferred.iloc[0]
    return str(row["strategy"]), int(row["k_numeric"])


def build_failure_analysis(
    strategy: str,
    k: Optional[int],
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    true_records: dict[int, dict[str, str]],
    country_block: bool,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    models: dict[tuple[str, str], TextModel],
    name_artifacts: RetrievalArtifacts,
    address_artifacts: RetrievalArtifacts,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    misses: list[tuple[int, int]] = []
    for query_index, truth in enumerate(truth_lists):
        candidates = set(
            strategy_candidates(
                strategy,
                k,
                query_index,
                eval_df,
                country_block,
                exact_map,
                name_artifacts.ids,
                address_artifacts.ids,
            )
        )
        misses.extend(
            (query_index, candidate_id)
            for candidate_id in truth
            if candidate_id not in candidates
        )

    columns = [
        "strategy",
        "k",
        "s1_entity_id",
        "true_matched_entity_id",
        "matched_source",
        "s1_name",
        "true_name",
        "s1_address",
        "true_address",
        "s1_country",
        "true_country",
        "match_count",
        "match_group",
        "name_similarity",
        "address_similarity",
        "name_rank_up_to_1000",
        "address_rank_up_to_1000",
        "exact_name_candidate_count",
        "exact_address_candidate_count",
        "failure_patterns",
    ]
    if not misses:
        return pd.DataFrame(columns=columns), pd.DataFrame(
            columns=["failure_pattern", "sampled_missed_positive_links"]
        )

    rng = np.random.default_rng(RANDOM_STATE)
    selected_positions = rng.choice(
        len(misses), size=min(sample_size, len(misses)), replace=False
    )
    rows = []
    for position in selected_positions:
        query_index, candidate_id = misses[int(position)]
        s1 = eval_df.iloc[query_index]
        true_record = true_records.get(candidate_id)
        if true_record is None:
            continue
        partition = str(s1["normalized_country"]) if country_block else "__all__"
        s1_name_norm = str(s1["normalized_name"])
        s1_address_norm = str(s1["normalized_address"])
        true_name_norm = normalize_value(true_record["business_name"])
        true_address_norm = normalize_value(true_record["business_address"])
        name_similarity = sparse_pair_cosine(
            models[(partition, "name")], s1_name_norm, true_name_norm
        )
        address_similarity = sparse_pair_cosine(
            models[(partition, "address")], s1_address_norm, true_address_norm
        )
        exact_name = exact_candidates_for_query(s1, "name", country_block, exact_map)
        exact_address = exact_candidates_for_query(s1, "address", country_block, exact_map)
        tags = failure_tags(
            s1_name_norm,
            true_name_norm,
            s1_address_norm,
            true_address_norm,
            name_similarity,
            address_similarity,
            len(set(exact_name)),
            len(set(exact_address)),
        )
        rows.append(
            {
                "strategy": strategy,
                "k": "" if k is None else k,
                "s1_entity_id": str(s1["source1_entity_id"]),
                "true_matched_entity_id": decode_candidate_id(candidate_id),
                "matched_source": candidate_source(candidate_id),
                "s1_name": "" if pd.isna(s1["business_name"]) else str(s1["business_name"]),
                "true_name": true_record["business_name"],
                "s1_address": ""
                if pd.isna(s1["business_address"])
                else str(s1["business_address"]),
                "true_address": true_record["business_address"],
                "s1_country": "" if pd.isna(s1["country"]) else str(s1["country"]),
                "true_country": true_record["country"],
                "match_count": int(s1["match_count"]),
                "match_group": str(s1["match_group"]),
                "name_similarity": name_similarity,
                "address_similarity": address_similarity,
                "name_rank_up_to_1000": rank_in_results(
                    name_artifacts.ids[query_index], candidate_id
                ),
                "address_rank_up_to_1000": rank_in_results(
                    address_artifacts.ids[query_index], candidate_id
                ),
                "exact_name_candidate_count": len(set(exact_name)),
                "exact_address_candidate_count": len(set(exact_address)),
                "failure_patterns": tags,
            }
        )

    failures = pd.DataFrame(rows, columns=columns)
    pattern_counter = Counter()
    for value in failures["failure_patterns"].fillna(""):
        for tag in str(value).split(";"):
            if tag:
                pattern_counter[tag] += 1
    patterns = pd.DataFrame(
        [
            {"failure_pattern": tag, "sampled_missed_positive_links": count}
            for tag, count in pattern_counter.most_common()
        ]
    )
    return failures, patterns


def attach_runtime_columns(
    strategy_df: pd.DataFrame,
    name_artifacts: RetrievalArtifacts,
    address_artifacts: RetrievalArtifacts,
) -> pd.DataFrame:
    result = strategy_df.copy()
    result["name_index_build_seconds"] = name_artifacts.build_seconds
    result["name_query_seconds"] = name_artifacts.query_seconds
    result["address_index_build_seconds"] = address_artifacts.build_seconds
    result["address_query_seconds"] = address_artifacts.query_seconds
    return result


def make_runtime_table(
    total_seconds: float,
    source_counts: dict[str, int],
    models: dict[tuple[str, str], TextModel],
    name_artifacts: RetrievalArtifacts,
    address_artifacts: RetrievalArtifacts,
    args: argparse.Namespace,
) -> pd.DataFrame:
    vocabulary = {f"{partition}_{field}": model.vocabulary_size for (partition, field), model in models.items()}
    return pd.DataFrame(
        [
            {
                "evaluation_entities": args.eval_size,
                "s2_candidate_records": source_counts["S2"],
                "s3_candidate_records": source_counts["S3"],
                "projection_dimension": args.projection_dim,
                "pq_bytes_per_vector": args.pq_m,
                "faiss_nlist": args.nlist,
                "faiss_nprobe": args.nprobe,
                "stored_rank_depth": args.search_k,
                "name_indexed_records": name_artifacts.ntotal,
                "address_indexed_records": address_artifacts.ntotal,
                "name_index_estimated_mb": name_artifacts.estimated_index_mb,
                "address_index_estimated_mb": address_artifacts.estimated_index_mb,
                "name_index_build_seconds": name_artifacts.build_seconds,
                "name_query_seconds": name_artifacts.query_seconds,
                "address_index_build_seconds": address_artifacts.build_seconds,
                "address_query_seconds": address_artifacts.query_seconds,
                "peak_process_rss_mb": current_peak_rss_mb(),
                "total_pipeline_seconds": total_seconds,
                "vocabulary_sizes_json": json.dumps(vocabulary, sort_keys=True),
            }
        ]
    )


def write_summary(
    output_path: Path,
    paths: dict[str, Path],
    country_summary: pd.DataFrame,
    country_block: bool,
    eval_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    source_df: pd.DataFrame,
    group_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    failure_strategy: tuple[str, Optional[int]],
    failures: pd.DataFrame,
    failure_patterns_df: pd.DataFrame,
) -> None:
    best = strategy_df.sort_values(
        ["link_recall", "complete_recall", "average_candidates"],
        ascending=[False, False, True],
    ).iloc[0]
    important = strategy_df.loc[
        strategy_df["strategy"].isin(
            [
                "Exact normalized name",
                "Exact normalized address",
                "Exact union",
                "Name TF-IDF",
                "Address TF-IDF",
                "All signals ranked",
                "All signals union",
            ]
        )
    ]
    country = country_summary.iloc[0]
    runtime = runtime_df.iloc[0]
    lines = [
        "TASK 2 - CANDIDATE GENERATION AND RECALL EVALUATION",
        "",
        "Scope",
        "-----",
        "Training data only. No classifier, threshold selection, test labels, or test predictions were used.",
        f"Source files: {', '.join(path.name for path in paths.values())}",
        "",
        "Normalization",
        "-------------",
        "Names and addresses: Unicode NFKC, casefold, underscore/punctuation to spaces, whitespace collapse, trim.",
        "Country: Unicode NFKC, casefold, whitespace collapse, trim.",
        "Original values were retained in evaluation and failure-analysis outputs.",
        "",
        "Full-ground-truth country validation",
        "------------------------------------",
        f"Total positive links: {int(country['total_positive_links']):,}",
        f"Same-country links: {int(country['same_country_links']):,} ({country['same_country_percentage']:.6f}%)",
        f"Different-country links: {int(country['different_country_links']):,} ({country['different_country_percentage']:.6f}%)",
        f"Missing-country links: {int(country['missing_country_links']):,} ({country['missing_country_percentage']:.6f}%)",
        f"Missing candidate lookups: {int(country['missing_candidate_lookup_links']):,}",
        f"Country hard block used: {country_block}",
        "",
        "Evaluation set",
        "--------------",
        f"S1 entities: {len(eval_df):,}, random_state={RANDOM_STATE}",
        "Match-count groups: "
        + ", ".join(
            f"{group}={count:,}"
            for group, count in eval_df["match_group"].value_counts().sort_index().items()
        ),
        "",
        "Retrieval method",
        "----------------",
        "Approximate retrieval uses character 3-5 gram TF-IDF, sparse random projection, and FAISS IVFPQ cosine-style search.",
        "S2 and S3 share each partition index; source-prefixed IDs remain distinct.",
        "Name and address retrieval are evaluated independently and in unions/rank fusion.",
        "",
        "Selected results",
        "----------------",
    ]
    for row in important.itertuples(index=False):
        k_label = "-" if row.k == "" else str(row.k)
        lines.append(
            f"{row.strategy} K={k_label}: link recall={row.link_recall:.4%}, "
            f"complete recall={row.complete_recall:.4%}, avg candidates={row.average_candidates:.2f}, "
            f"p95={row.p95_candidates:.0f}, max={int(row.max_candidates)}"
        )
    lines.extend(
        [
            "",
            "Best observed configuration",
            "---------------------------",
            f"{best['strategy']} K={best['k'] if best['k'] != '' else '-'}: "
            f"link recall={best['link_recall']:.4%}, complete recall={best['complete_recall']:.4%}, "
            f"average candidates={best['average_candidates']:.2f}.",
            "",
            "S2 vs S3 for the best configuration",
            "------------------------------------",
        ]
    )
    best_source = source_df.loc[
        source_df["strategy"].eq(best["strategy"])
        & source_df["k"].astype(str).eq(str(best["k"]))
    ]
    for row in best_source.itertuples(index=False):
        lines.append(f"{row.source}: {row.link_recall:.4%} ({row.retrieved_positive_links:,}/{row.positive_links:,})")
    lines.extend(["", "Recall by match-count group", "---------------------------"])
    best_group = group_df.loc[
        group_df["strategy"].eq(best["strategy"])
        & group_df["k"].astype(str).eq(str(best["k"]))
    ]
    for row in best_group.itertuples(index=False):
        lines.append(
            f"{row.match_group}: link recall={row.link_recall:.4%}, complete recall={row.complete_recall:.4%}"
        )
    lines.extend(
        [
            "",
            "Runtime and memory",
            "------------------",
            f"Total pipeline runtime: {runtime['total_pipeline_seconds'] / 60.0:.2f} minutes",
            f"Peak process RSS: {runtime['peak_process_rss_mb']:.1f} MB",
            f"Estimated compressed name index: {runtime['name_index_estimated_mb']:.1f} MB",
            f"Estimated compressed address index: {runtime['address_index_estimated_mb']:.1f} MB",
            "No dense 10,000 x 10.3M similarity matrix was created.",
            "",
            "Failure analysis",
            "----------------",
            f"Configuration analyzed: {failure_strategy[0]} K={failure_strategy[1]}",
            f"Sampled missed positive links: {len(failures):,}",
        ]
    )
    if not failure_patterns_df.empty:
        for row in failure_patterns_df.itertuples(index=False):
            lines.append(f"{row.failure_pattern}: {row.sampled_missed_positive_links}")
    else:
        lines.append("No missed positives for the analyzed configuration.")
    lines.extend(
        [
            "",
            "Conclusion",
            "----------",
            "Use the strategy comparison to select the smallest candidate budget that preserves acceptable link and complete recall.",
            "Exact name/address rules are useful recall supplements but are not sufficient as hard blocking rules.",
            "The approximate name and address signals should remain separate inputs to candidate generation because their union recovers complementary positives.",
            "Task 2 stops here. No pairwise classifier or final match threshold was built.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = make_parser().parse_args()
    if args.eval_size < 4:
        raise ValueError("eval-size must be at least 4")
    if args.search_k < 100:
        raise ValueError("search-k must be at least 100")
    if args.projection_dim % args.pq_m != 0:
        raise ValueError("projection-dim must be divisible by pq-m")
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    faiss.omp_set_num_threads(args.faiss_threads)
    total_started = time.perf_counter()
    paths = validate_input_files(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Using training data only from {args.data_dir.resolve()}")

    gt, eval_gt = load_ground_truth(paths["GT"], args.eval_size)
    encoder = CountryEncoder()
    eval_df, s1_ids, s1_country_codes = scan_source1(
        paths["S1"], eval_gt, encoder, args.chunksize
    )
    link_candidate_ids, link_s1_ids, link_s1_countries, _ = map_gt_to_s1_countries(
        gt, s1_ids, s1_country_codes
    )

    truth_lists = [parse_truth_ids(value) for value in eval_df["matched_entity_ids"]]
    eval_truth_strings = {
        decode_candidate_id(candidate_id)
        for truth in truth_lists
        for candidate_id in truth
    }
    (
        candidate_ids,
        candidate_countries,
        training_sample,
        true_records,
        source_counts,
    ) = preliminary_candidate_scan(
        paths,
        eval_truth_strings,
        encoder,
        args.chunksize,
        args.sample_modulus,
    )
    if len(true_records) != len(eval_truth_strings):
        missing = eval_truth_strings - {record["entity_id"] for record in true_records.values()}
        raise ValueError(f"Missing {len(missing):,} evaluation true-match records")

    country_summary, country_examples, country_block = validate_country_links(
        link_candidate_ids,
        link_s1_ids,
        link_s1_countries,
        candidate_ids,
        candidate_countries,
        encoder,
    )
    del candidate_ids, candidate_countries, s1_ids, s1_country_codes
    del link_candidate_ids, link_s1_ids, link_s1_countries
    del gt, eval_gt
    gc.collect()

    training_sample = trim_training_sample(
        training_sample, args.train_sample_per_source_country
    )
    models, partitions = build_text_models(
        training_sample,
        eval_df,
        country_block,
        args.max_features,
        args.projection_dim,
    )
    query_keys = prepare_exact_query_keys(eval_df, country_block)
    exact_map: dict[tuple[str, str], dict[str, list[int]]] = {
        (partition, field): defaultdict(list)
        for partition in partitions
        for field in FIELDS
    }

    name_artifacts = build_field_index_and_retrieve(
        "name",
        paths,
        eval_df,
        models,
        partitions,
        country_block,
        query_keys,
        exact_map,
        args.chunksize,
        args.transform_batch_size,
        args.projection_dim,
        args.nlist,
        args.pq_m,
        args.nprobe,
        args.search_k,
        collect_exact=True,
    )
    address_artifacts = build_field_index_and_retrieve(
        "address",
        paths,
        eval_df,
        models,
        partitions,
        country_block,
        query_keys,
        exact_map,
        args.chunksize,
        args.transform_batch_size,
        args.projection_dim,
        args.nlist,
        args.pq_m,
        args.nprobe,
        args.search_k,
        collect_exact=False,
    )

    strategy_df, group_df, source_df = evaluate_strategies(
        eval_df,
        truth_lists,
        country_block,
        exact_map,
        name_artifacts,
        address_artifacts,
    )
    strategy_df = attach_runtime_columns(strategy_df, name_artifacts, address_artifacts)
    failure_strategy = choose_failure_configuration(strategy_df)
    failures_df, failure_patterns_df = build_failure_analysis(
        failure_strategy[0],
        failure_strategy[1],
        eval_df,
        truth_lists,
        true_records,
        country_block,
        exact_map,
        models,
        name_artifacts,
        address_artifacts,
        args.failure_sample_size,
    )
    runtime_df = make_runtime_table(
        elapsed(total_started),
        source_counts,
        models,
        name_artifacts,
        address_artifacts,
        args,
    )

    eval_output_columns = [
        "eval_index",
        "source1_entity_id",
        "match_count",
        "match_group",
        "matched_entity_ids",
        "business_name",
        "business_address",
        "country",
        "normalized_name",
        "normalized_address",
        "normalized_country",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_df.loc[:, eval_output_columns].to_csv(
        args.output_dir / "task2_eval_entities.csv", index=False
    )
    country_summary.to_csv(args.output_dir / "task2_country_validation.csv", index=False)
    country_examples.to_csv(
        args.output_dir / "task2_country_mismatch_examples.csv", index=False
    )
    strategy_df.to_csv(args.output_dir / "task2_strategy_comparison.csv", index=False)
    group_df.to_csv(args.output_dir / "task2_recall_by_match_count.csv", index=False)
    source_df.to_csv(args.output_dir / "task2_recall_by_source.csv", index=False)
    failures_df.to_csv(args.output_dir / "task2_retrieval_failures.csv", index=False)
    failure_patterns_df.to_csv(
        args.output_dir / "task2_failure_patterns.csv", index=False
    )
    runtime_df.to_csv(args.output_dir / "task2_runtime_memory.csv", index=False)
    write_summary(
        args.output_dir / "task2_summary.txt",
        paths,
        country_summary,
        country_block,
        eval_df,
        strategy_df,
        source_df,
        group_df,
        runtime_df,
        failure_strategy,
        failures_df,
        failure_patterns_df,
    )
    log(
        f"Task 2 complete in {elapsed(total_started) / 60.0:.2f} minutes. "
        f"Outputs written to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
