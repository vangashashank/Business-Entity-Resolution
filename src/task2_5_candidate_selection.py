#!/usr/bin/env python3
"""Task 2.5: candidate-generator ablation, selection, and freeze decision."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / ".task2_5_vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from unidecode import unidecode

import task2_candidate_generation as t2


RANDOM_STATE = 42
BASELINE_UNION = "Existing name + address union, K=100"
BASELINE_RANKED = "Existing all signals ranked, K=100"
TRANSLITERATION = "Baseline + transliteration union"
ADDRESS_NUMBER = "Baseline + address-number union"
MISSING_ADDRESS = "Baseline + missing-address handling"
LEGAL_SUFFIX = "Baseline + legal-suffix union"
COMBINED_UNION = "Combined improved union"
COMBINED_RANKED_100 = "Combined improved ranked, cap=100"
COMBINED_RANKED_200 = "Combined improved ranked, cap=200"
COMBINED_AUGMENTED_250 = "Combined baseline-preserving augmentation, cap=250"

CONFIGURATIONS = [
    BASELINE_RANKED,
    BASELINE_UNION,
    TRANSLITERATION,
    ADDRESS_NUMBER,
    MISSING_ADDRESS,
    LEGAL_SUFFIX,
    COMBINED_UNION,
    COMBINED_RANKED_100,
    COMBINED_RANKED_200,
    COMBINED_AUGMENTED_250,
]

ORDINAL_WORDS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
}

LEGAL_SUFFIX_PATTERN = re.compile(
    r"(?:\b(?:private|pvt|limited|ltd|llp|llc|incorporated|inc|corporation|corp|"
    r"company|co|plc|enterprises|enterprise)\b\s*)+$",
    flags=re.IGNORECASE,
)
ORDINAL_TOKEN_PATTERN = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
NUMBER_TOKEN_PATTERN = re.compile(r"\d+[a-z]?")
SEPARATED_NUMBER_PATTERN = re.compile(r"\d+(?:\s*[-/]\s*\d+)+")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def elapsed(start: float) -> float:
    return time.perf_counter() - start


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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task2_5_outputs",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--sample-modulus", type=int, default=30)
    parser.add_argument("--train-sample-per-source-country", type=int, default=50_000)
    parser.add_argument("--max-features", type=int, default=32_768)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--nlist", type=int, default=2_048)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--baseline-search-k", type=int, default=1_000)
    parser.add_argument("--search-k", type=int, default=250)
    parser.add_argument("--transform-batch-size", type=int, default=50_000)
    parser.add_argument("--faiss-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--failure-sample-size", type=int, default=50)
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "S1": args.data_dir / "train_source1.tsv",
        "S2": args.data_dir / "train_source2.tsv",
        "S3": args.data_dir / "train_source3.tsv",
        "GT": args.data_dir / "train_ground_truth.tsv",
        "eval": args.task2_output_dir / "task2_eval_entities.csv",
        "task2_strategies": args.task2_output_dir / "task2_strategy_comparison.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Task 2.5 inputs: {missing}")
    if any("test" in path.name.casefold() for key, path in paths.items() if key in {"S1", "S2", "S3", "GT"}):
        raise ValueError("Task 2.5 must use training data only")
    return paths


def collapse_repeated_ascii(text: str) -> str:
    return re.sub(r"([a-z])\1+", r"\1", text)


def transliterate_value(value: object) -> str:
    base = t2.normalize_value(value)
    if not base:
        return ""
    transliterated = t2.normalize_value(unidecode(base))
    collapsed = collapse_repeated_ascii(transliterated)
    if collapsed and collapsed != transliterated:
        return f"{transliterated} {collapsed}"
    return transliterated


def transliterate_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        [transliterate_value(value) for value in series],
        index=series.index,
        dtype="string",
    )


def strip_legal_suffix_value(value: object) -> str:
    base = t2.normalize_value(value)
    if not base:
        return ""
    stripped = LEGAL_SUFFIX_PATTERN.sub("", base).strip()
    return stripped if stripped else base


def strip_legal_suffix_series(series: pd.Series) -> pd.Series:
    base = t2.normalize_series(series)
    stripped = base.str.replace(LEGAL_SUFFIX_PATTERN, "", regex=True).str.strip()
    return stripped.where(stripped.ne(""), base)


def canonical_number_token(token: str) -> str:
    match = ORDINAL_TOKEN_PATTERN.match(token)
    if match:
        token = match.group(1)
    if token.isdigit():
        return str(int(token))
    number = re.match(r"^(\d+)([a-z])$", token)
    if number:
        return f"{int(number.group(1))}{number.group(2)}"
    return token


def number_tokens(value: object) -> list[str]:
    base = t2.normalize_value(value)
    tokens = []
    for token in base.split():
        token = ORDINAL_WORDS.get(token, token)
        if NUMBER_TOKEN_PATTERN.fullmatch(token) or ORDINAL_TOKEN_PATTERN.fullmatch(token):
            tokens.append(canonical_number_token(token))
    return tokens


def address_number_value(value: object) -> str:
    base = t2.normalize_value(value)
    if not base:
        return ""
    canonical_tokens = []
    for token in base.split():
        token = ORDINAL_WORDS.get(token, token)
        canonical_tokens.append(canonical_number_token(token))
    canonical = " ".join(canonical_tokens)
    features = [f"num{token}" for token in number_tokens(canonical)]

    raw = "" if value is None or pd.isna(value) else str(value).casefold()
    for separated in SEPARATED_NUMBER_PATTERN.findall(raw):
        parts = [str(int(part)) for part in re.findall(r"\d+", separated)]
        if parts:
            features.append("numseq" + "_".join(parts))
            features.append("numcompact" + "".join(parts))
    if len(features) > 1:
        plain_numbers = [feature[3:] for feature in features if feature.startswith("num") and not feature.startswith("numseq") and not feature.startswith("numcompact")]
        if len(plain_numbers) > 1:
            features.append("numordered" + "_".join(plain_numbers))
    return " ".join([canonical, *features]).strip()


def address_number_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        [address_number_value(value) for value in series],
        index=series.index,
        dtype="string",
    )


@dataclass(frozen=True)
class SignalSpec:
    name: str
    raw_column: str
    transform: Callable[[pd.Series], pd.Series]


def dataframe_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def run_smoke_validation(paths: dict[str, Path], output_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    frames = []
    for source in ("S2", "S3"):
        frame = pd.read_csv(
            paths[source],
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype="string",
            nrows=20_000,
        )
        frame["source"] = source
        frames.append(frame)
    sample = pd.concat(frames, ignore_index=True)

    transform_checks = {
        "transliteration_nonempty": bool(transliterate_value("गुड इम्पेक्स प्रा. लि.")),
        "transliteration_ascii": transliterate_value("गुड इम्पेक्स प्रा. लि.").isascii(),
        "zero_padding": "num1301" in address_number_value("001301 Eventide Drive"),
        "separator_compaction": "numcompact1827" in address_number_value("House 18-27 Main Road"),
        "ordinal_normalization": "num2" in address_number_value("Second Floor"),
        "suffix_normalization": strip_legal_suffix_value("Acme Private Limited") == "acme",
    }
    if not all(transform_checks.values()):
        raise AssertionError(f"Task 2.5 transform smoke checks failed: {transform_checks}")

    specs = [
        SignalSpec("transliterated_name", "business_name", transliterate_series),
        SignalSpec("suffix_name", "business_name", strip_legal_suffix_series),
        SignalSpec("number_address", "business_address", address_number_series),
    ]
    retrieval_checks = {}
    for spec in specs:
        values = spec.transform(sample[spec.raw_column]).fillna("").astype(str)
        nonempty = values.ne("")
        values = values.loc[nonempty].head(12_000)
        ids = np.flatnonzero(nonempty.to_numpy())[: len(values)].astype(np.int64)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=False,
            min_df=2,
            max_features=4_096,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(values.tolist())
        projector = SparseRandomProjection(
            n_components=64,
            dense_output=True,
            random_state=RANDOM_STATE,
        ).fit(matrix[: min(2_000, len(values))])
        projected = projector.transform(matrix)
        if sparse.issparse(projected):
            projected = projected.toarray()
        projected = np.ascontiguousarray(projected, dtype=np.float32)
        faiss.normalize_L2(projected)
        index = faiss.IndexFlatIP(64)
        index.add(projected)
        _, labels = index.search(projected[:100], 100)
        self_retrieval = float(np.mean([query_index in labels[query_index] for query_index in range(100)]))
        retrieval_checks[spec.name] = {
            "records": int(len(values)),
            "features": int(len(vectorizer.vocabulary_)),
            "self_retrieval_at_100": self_retrieval,
        }
        if self_retrieval < 0.95:
            raise AssertionError(f"Low smoke self-retrieval for {spec.name}: {self_retrieval}")
        del matrix, projected, index
        gc.collect()

    result = {
        "status": "passed",
        "training_only": True,
        "candidate_rows_tested": int(len(sample)),
        "transform_checks": transform_checks,
        "retrieval_checks": retrieval_checks,
        "runtime_seconds": elapsed(started),
        "peak_rss_mb": t2.current_peak_rss_mb(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task2_5_smoke_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(f"Smoke validation passed in {result['runtime_seconds']:.1f}s")
    return result


def load_evaluation(eval_path: Path) -> tuple[pd.DataFrame, list[list[int]], set[str]]:
    eval_df = pd.read_csv(eval_path, dtype="string")
    numeric_columns = ["eval_index", "match_count"]
    for column in numeric_columns:
        eval_df[column] = pd.to_numeric(eval_df[column], errors="raise")
    expected_groups = {"1": 2_500, "2": 2_500, "3-5": 2_500, "6+": 2_500}
    if len(eval_df) != 10_000:
        raise ValueError(f"Expected 10,000 Task 2 evaluation entities, found {len(eval_df):,}")
    if eval_df["match_group"].value_counts().to_dict() != expected_groups:
        raise ValueError("Task 2 evaluation match groups changed")
    if not np.array_equal(eval_df["eval_index"].to_numpy(), np.arange(len(eval_df))):
        raise ValueError("Task 2 evaluation order changed")
    truth_lists = [t2.parse_truth_ids(value) for value in eval_df["matched_entity_ids"]]
    truth_strings = {
        t2.decode_candidate_id(candidate_id)
        for truth in truth_lists
        for candidate_id in truth
    }
    log(
        f"Reused exact Task 2 evaluation set: {len(eval_df):,} S1 entities, "
        f"{sum(map(len, truth_lists)):,} positive links"
    )
    return eval_df, truth_lists, truth_strings


def collect_training_sample_and_truth_records(
    paths: dict[str, Path],
    truth_strings: set[str],
    chunksize: int,
    sample_modulus: int,
) -> tuple[pd.DataFrame, dict[int, dict[str, str]], dict[str, int]]:
    started = time.perf_counter()
    sample_chunks = []
    true_records: dict[int, dict[str, str]] = {}
    source_counts = {}

    for source in ("S2", "S3"):
        row_count = 0
        reader = pd.read_csv(
            paths[source],
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype="string",
            chunksize=chunksize,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            ids = t2.numeric_id_series(chunk["entity_id"], source)
            countries = t2.normalize_country_series(chunk["country"])
            hashes = pd.util.hash_pandas_object(chunk["entity_id"], index=False).to_numpy(
                dtype=np.uint64
            )
            sample_mask = hashes % np.uint64(sample_modulus) == 0
            if bool(sample_mask.any()):
                sampled = chunk.loc[sample_mask].copy()
                sampled["_source"] = source
                sampled["_global_id"] = ids[sample_mask]
                sampled["_hash"] = hashes[sample_mask]
                sampled["normalized_country"] = countries.loc[sample_mask].to_numpy()
                sample_chunks.append(sampled)

            truth_mask = chunk["entity_id"].isin(truth_strings).to_numpy()
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
            row_count += len(chunk)
            if chunk_number % 5 == 0:
                log(f"Task 2.5 preliminary {source}: {row_count:,} rows")
        source_counts[source] = row_count

    found_truth_strings = {record["entity_id"] for record in true_records.values()}
    missing = truth_strings - found_truth_strings
    if missing:
        raise ValueError(f"Missing {len(missing):,} Task 2 evaluation truth records")
    sample_df = pd.concat(sample_chunks, ignore_index=True)
    log(
        f"Collected {len(sample_df):,} deterministic training-sample rows and "
        f"{len(true_records):,} true records in {elapsed(started):.1f}s"
    )
    return sample_df, true_records, source_counts


def build_variant_models(
    training_sample: pd.DataFrame,
    eval_df: pd.DataFrame,
    specs: list[SignalSpec],
    max_features: int,
    projection_dim: int,
) -> tuple[dict[tuple[str, str], t2.TextModel], list[str]]:
    started = time.perf_counter()
    partitions = sorted(eval_df["normalized_country"].dropna().astype(str).unique().tolist())
    models: dict[tuple[str, str], t2.TextModel] = {}
    for partition in partitions:
        partition_sample = training_sample.loc[
            training_sample["normalized_country"].eq(partition)
        ]
        for spec in specs:
            values = spec.transform(partition_sample[spec.raw_column]).fillna("").astype(str)
            values = values.loc[values.ne("")]
            if len(values) < 1_000:
                raise ValueError(f"Too few {spec.name} samples in {partition}")
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
            models[(partition, spec.name)] = t2.TextModel(
                vectorizer=vectorizer,
                projector=projector,
                train_texts=values.tolist(),
                vocabulary_size=len(vectorizer.vocabulary_),
            )
            log(
                f"Fitted {partition}/{spec.name}: {len(values):,} strings, "
                f"{len(vectorizer.vocabulary_):,} features"
            )
            del matrix
            gc.collect()
    log(f"Fitted {len(models)} Task 2.5 variant models in {elapsed(started):.1f}s")
    return models, partitions


def build_variant_indexes_and_retrieve(
    specs: list[SignalSpec],
    paths: dict[str, Path],
    eval_df: pd.DataFrame,
    models: dict[tuple[str, str], t2.TextModel],
    partitions: list[str],
    args: argparse.Namespace,
) -> dict[str, t2.RetrievalArtifacts]:
    indexes: dict[tuple[str, str], faiss.IndexIVFPQ] = {}
    train_seconds = Counter()
    add_seconds = Counter()
    for spec in specs:
        for partition in partitions:
            index, duration = t2.create_faiss_index(
                models[(partition, spec.name)],
                args.projection_dim,
                args.nlist,
                args.pq_m,
                args.nprobe,
            )
            indexes[(partition, spec.name)] = index
            train_seconds[spec.name] += duration

    for source in ("S2", "S3"):
        row_count = 0
        reader = pd.read_csv(
            paths[source],
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype="string",
            chunksize=args.chunksize,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            ids = t2.numeric_id_series(chunk["entity_id"], source)
            countries = t2.normalize_country_series(chunk["country"])
            for spec in specs:
                signal_started = time.perf_counter()
                transformed = spec.transform(chunk[spec.raw_column]).fillna("").astype(str)
                for partition in partitions:
                    partition_mask = countries.eq(partition).to_numpy()
                    if not bool(partition_mask.any()):
                        continue
                    partition_values = transformed.loc[partition_mask].reset_index(drop=True)
                    partition_ids = ids[partition_mask]
                    nonempty = partition_values.ne("").to_numpy()
                    if not bool(nonempty.any()):
                        continue
                    add_values = partition_values.loc[nonempty].tolist()
                    add_ids = partition_ids[nonempty]
                    model = models[(partition, spec.name)]
                    index = indexes[(partition, spec.name)]
                    for start in range(0, len(add_values), args.transform_batch_size):
                        stop = min(start + args.transform_batch_size, len(add_values))
                        vectors = t2.project_texts(model, add_values[start:stop])
                        vector_nonzero = np.linalg.norm(vectors, axis=1) > 0
                        if bool(vector_nonzero.any()):
                            index.add_with_ids(
                                vectors[vector_nonzero],
                                np.ascontiguousarray(
                                    add_ids[start:stop][vector_nonzero], dtype=np.int64
                                ),
                            )
                        del vectors
                add_seconds[spec.name] += elapsed(signal_started)
                del transformed
                gc.collect()
            row_count += len(chunk)
            if chunk_number % 5 == 0:
                totals = ", ".join(
                    f"{spec.name}={sum(indexes[(partition, spec.name)].ntotal for partition in partitions):,}"
                    for spec in specs
                )
                log(f"Variant indexes {source}: {row_count:,} rows; {totals}")

    artifacts = {}
    for spec in specs:
        query_started = time.perf_counter()
        result_ids = np.full((len(eval_df), args.search_k), -1, dtype=np.int64)
        result_scores = np.full((len(eval_df), args.search_k), -np.inf, dtype=np.float32)
        for partition in partitions:
            positions = np.flatnonzero(eval_df["normalized_country"].eq(partition).to_numpy())
            values = spec.transform(eval_df.loc[positions, spec.raw_column]).fillna("").astype(str)
            nonempty = values.ne("").to_numpy()
            if not bool(nonempty.any()):
                continue
            vectors = t2.project_texts(
                models[(partition, spec.name)], values.loc[nonempty].tolist()
            )
            distances, labels = indexes[(partition, spec.name)].search(vectors, args.search_k)
            destination = positions[nonempty]
            result_ids[destination, :] = labels
            result_scores[destination, :] = distances
            del vectors, distances, labels
        query_seconds = elapsed(query_started)
        ntotal = sum(indexes[(partition, spec.name)].ntotal for partition in partitions)
        estimated_mb = ntotal * (args.pq_m + 8) / (1024.0 * 1024.0)
        artifacts[spec.name] = t2.RetrievalArtifacts(
            ids=result_ids,
            scores=result_scores,
            build_seconds=float(train_seconds[spec.name] + add_seconds[spec.name]),
            query_seconds=query_seconds,
            ntotal=ntotal,
            estimated_index_mb=estimated_mb,
        )
        log(
            f"Queried {spec.name} top-{args.search_k}: ntotal={ntotal:,}, "
            f"build={artifacts[spec.name].build_seconds:.1f}s, query={query_seconds:.1f}s"
        )
    del indexes
    gc.collect()
    return artifacts


def clean_ids(values: np.ndarray, k: int) -> list[int]:
    return t2.clean_ranked_ids(values, k)


def ordered_union(*candidate_lists: Iterable[int]) -> list[int]:
    seen = set()
    result = []
    for candidates in candidate_lists:
        for candidate_id in candidates:
            candidate_id = int(candidate_id)
            if candidate_id < 0 or candidate_id in seen:
                continue
            seen.add(candidate_id)
            result.append(candidate_id)
    return result


def reciprocal_rank_candidates(
    signal_lists: list[tuple[list[int], float]],
    exact_name: list[int],
    exact_address: list[int],
    limit: int,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for candidates, weight in signal_lists:
        for rank, candidate_id in enumerate(candidates, start=1):
            scores[int(candidate_id)] += weight / (60.0 + rank)
    for candidate_id in exact_name:
        scores[int(candidate_id)] += 2.0
    for candidate_id in exact_address:
        scores[int(candidate_id)] += 2.0
    return sorted(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id))[:limit]


def exact_lists(
    query_index: int,
    eval_df: pd.DataFrame,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
) -> tuple[list[int], list[int]]:
    row = eval_df.iloc[query_index]
    return (
        t2.exact_candidates_for_query(row, "name", True, exact_map),
        t2.exact_candidates_for_query(row, "address", True, exact_map),
    )


def configuration_candidates(
    configuration: str,
    query_index: int,
    eval_df: pd.DataFrame,
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    artifacts: dict[str, t2.RetrievalArtifacts],
) -> list[int]:
    baseline_name_100 = clean_ids(artifacts["baseline_name"].ids[query_index], 100)
    baseline_address_100 = clean_ids(artifacts["baseline_address"].ids[query_index], 100)
    transliterated_100 = clean_ids(artifacts["transliterated_name"].ids[query_index], 100)
    suffix_100 = clean_ids(artifacts["suffix_name"].ids[query_index], 100)
    number_address_100 = clean_ids(artifacts["number_address"].ids[query_index], 100)
    address_missing = not str(eval_df.iloc[query_index]["normalized_address"]).strip()
    exact_name, exact_address = exact_lists(query_index, eval_df, exact_map)

    if configuration == BASELINE_RANKED:
        return t2.all_signal_ranked(
            baseline_name_100,
            baseline_address_100,
            exact_name,
            exact_address,
            100,
        )
    baseline_union = ordered_union(baseline_name_100, baseline_address_100)
    if configuration == BASELINE_UNION:
        return baseline_union
    if configuration == TRANSLITERATION:
        return ordered_union(baseline_union, transliterated_100)
    if configuration == ADDRESS_NUMBER:
        return ordered_union(baseline_union, number_address_100)
    if configuration == MISSING_ADDRESS:
        if address_missing:
            return clean_ids(artifacts["baseline_name"].ids[query_index], 200)
        return baseline_union
    if configuration == LEGAL_SUFFIX:
        return ordered_union(baseline_union, suffix_100)

    baseline_for_combined = (
        clean_ids(artifacts["baseline_name"].ids[query_index], 200)
        if address_missing
        else baseline_union
    )
    active_signals: list[tuple[list[int], float]] = [
        (clean_ids(artifacts["baseline_name"].ids[query_index], 200 if address_missing else 100), 1.0),
        (transliterated_100, 1.0),
        (suffix_100, 1.0),
    ]
    if not address_missing:
        active_signals.extend(
            [
                (baseline_address_100, 1.0),
                (number_address_100, 1.0),
            ]
        )
    if configuration == COMBINED_UNION:
        return ordered_union(
            baseline_for_combined,
            transliterated_100,
            suffix_100,
            [] if address_missing else number_address_100,
        )
    if configuration == COMBINED_RANKED_100:
        return reciprocal_rank_candidates(
            active_signals, exact_name, exact_address, 100
        )
    if configuration == COMBINED_RANKED_200:
        return reciprocal_rank_candidates(
            active_signals, exact_name, exact_address, 200
        )
    if configuration == COMBINED_AUGMENTED_250:
        result = list(baseline_for_combined)
        seen = set(result)
        ranked_new = reciprocal_rank_candidates(active_signals, [], [], 500)
        for candidate_id in ranked_new:
            if candidate_id in seen:
                continue
            result.append(candidate_id)
            seen.add(candidate_id)
            if len(result) >= 250:
                break
        return result
    raise ValueError(f"Unknown Task 2.5 configuration: {configuration}")


def configuration_required_signals(configuration: str) -> list[str]:
    baseline = ["baseline_name", "baseline_address"]
    if configuration in {BASELINE_RANKED, BASELINE_UNION, MISSING_ADDRESS}:
        return baseline
    if configuration == TRANSLITERATION:
        return [*baseline, "transliterated_name"]
    if configuration == ADDRESS_NUMBER:
        return [*baseline, "number_address"]
    if configuration == LEGAL_SUFFIX:
        return [*baseline, "suffix_name"]
    return [*baseline, "transliterated_name", "number_address", "suffix_name"]


def address_availability_group(s1_missing: bool, true_missing: bool) -> str:
    if s1_missing and true_missing:
        return "both_missing"
    if s1_missing:
        return "s1_missing_only"
    if true_missing:
        return "true_candidate_missing_only"
    return "neither_missing"


def evaluate_configuration(
    configuration: str,
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    true_records: dict[int, dict[str, str]],
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    artifacts: dict[str, t2.RetrievalArtifacts],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    started = time.perf_counter()
    total_links = total_hits = complete_entities = 0
    candidate_counts = []
    source_totals = Counter()
    source_hits = Counter()
    group_totals = Counter()
    group_hits = Counter()
    group_entities = Counter()
    group_complete = Counter()
    availability_totals = Counter()
    availability_hits = Counter()

    for query_index, truth in enumerate(truth_lists):
        candidate_set = set(
            configuration_candidates(
                configuration, query_index, eval_df, exact_map, artifacts
            )
        )
        hits = sum(candidate_id in candidate_set for candidate_id in truth)
        group = str(eval_df.iloc[query_index]["match_group"])
        total_links += len(truth)
        total_hits += hits
        candidate_counts.append(len(candidate_set))
        complete = hits == len(truth)
        complete_entities += int(complete)
        group_totals[group] += len(truth)
        group_hits[group] += hits
        group_entities[group] += 1
        group_complete[group] += int(complete)
        s1_missing = not str(eval_df.iloc[query_index]["normalized_address"]).strip()
        for candidate_id in truth:
            hit = candidate_id in candidate_set
            source = t2.candidate_source(candidate_id)
            source_totals[source] += 1
            source_hits[source] += int(hit)
            true_missing = not t2.normalize_value(true_records[candidate_id]["business_address"])
            availability = address_availability_group(s1_missing, true_missing)
            availability_totals[availability] += 1
            availability_hits[availability] += int(hit)

    required = configuration_required_signals(configuration)
    retrieval_seconds = sum(
        artifacts[signal].build_seconds + artifacts[signal].query_seconds
        for signal in required
    )
    artifact_bytes = sum(
        artifacts[signal].ids.nbytes + artifacts[signal].scores.nbytes
        for signal in required
    )
    max_index_mb = max(artifacts[signal].estimated_index_mb for signal in required)
    evaluation_seconds = elapsed(started)
    global_row = {
        "configuration": configuration,
        "positive_links": total_links,
        "retrieved_positive_links": total_hits,
        "link_recall": total_hits / total_links,
        "complete_entities": complete_entities,
        "complete_recall": complete_entities / len(truth_lists),
        "average_candidates": float(np.mean(candidate_counts)),
        "median_candidates": float(np.median(candidate_counts)),
        "p95_candidates": float(np.percentile(candidate_counts, 95)),
        "max_candidates": max(candidate_counts),
        "retrieval_precompute_seconds": retrieval_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_runtime_seconds": retrieval_seconds + evaluation_seconds,
        "approx_retrieval_artifacts_mb": artifact_bytes / (1024.0 * 1024.0),
        "approx_peak_index_mb": max_index_mb,
        "signals": ",".join(required),
    }
    source_rows = [
        {
            "configuration": configuration,
            "source": source,
            "positive_links": source_totals[source],
            "retrieved_positive_links": source_hits[source],
            "link_recall": source_hits[source] / source_totals[source],
        }
        for source in ("S2", "S3")
    ]
    group_rows = [
        {
            "configuration": configuration,
            "match_group": group,
            "entities": group_entities[group],
            "positive_links": group_totals[group],
            "retrieved_positive_links": group_hits[group],
            "link_recall": group_hits[group] / group_totals[group],
            "complete_entities": group_complete[group],
            "complete_recall": group_complete[group] / group_entities[group],
        }
        for group in ("1", "2", "3-5", "6+")
    ]
    availability_rows = [
        {
            "configuration": configuration,
            "address_availability": group,
            "positive_links": availability_totals[group],
            "retrieved_positive_links": availability_hits[group],
            "link_recall": availability_hits[group] / availability_totals[group]
            if availability_totals[group]
            else np.nan,
        }
        for group in (
            "both_missing",
            "s1_missing_only",
            "true_candidate_missing_only",
            "neither_missing",
        )
    ]
    log(
        f"Evaluated {configuration}: recall={global_row['link_recall']:.4%}, "
        f"complete={global_row['complete_recall']:.4%}, "
        f"avg={global_row['average_candidates']:.1f}, max={global_row['max_candidates']}"
    )
    return global_row, source_rows, group_rows, availability_rows


def evaluate_all(
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    true_records: dict[int, dict[str, str]],
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    artifacts: dict[str, t2.RetrievalArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows = []
    source_rows = []
    group_rows = []
    availability_rows = []
    for configuration in CONFIGURATIONS:
        global_row, source_part, group_part, availability_part = evaluate_configuration(
            configuration,
            eval_df,
            truth_lists,
            true_records,
            exact_map,
            artifacts,
        )
        global_rows.append(global_row)
        source_rows.extend(source_part)
        group_rows.extend(group_part)
        availability_rows.extend(availability_part)
    return (
        pd.DataFrame(global_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(availability_rows),
    )


def verify_baseline_parity(
    ablation_df: pd.DataFrame,
    task2_strategy_path: Path,
) -> pd.DataFrame:
    task2 = pd.read_csv(task2_strategy_path)
    comparisons = [
        (BASELINE_UNION, "Name + address union"),
        (BASELINE_RANKED, "All signals ranked"),
    ]
    rows = []
    for task2_5_name, task2_name in comparisons:
        current = ablation_df.loc[ablation_df["configuration"].eq(task2_5_name)].iloc[0]
        previous = task2.loc[
            task2["strategy"].eq(task2_name)
            & pd.to_numeric(task2["k"], errors="coerce").eq(100)
        ].iloc[0]
        recall_delta = float(current["link_recall"] - previous["link_recall"])
        complete_delta = float(current["complete_recall"] - previous["complete_recall"])
        rows.append(
            {
                "configuration": task2_5_name,
                "task2_link_recall": previous["link_recall"],
                "task2_5_link_recall": current["link_recall"],
                "link_recall_delta": recall_delta,
                "task2_complete_recall": previous["complete_recall"],
                "task2_5_complete_recall": current["complete_recall"],
                "complete_recall_delta": complete_delta,
                "parity_passed": abs(recall_delta) < 1e-12
                and abs(complete_delta) < 1e-12,
            }
        )
    parity = pd.DataFrame(rows)
    if not bool(parity["parity_passed"].all()):
        raise AssertionError(f"Task 2 baseline parity failed:\n{parity}")
    log("Task 2 baseline parity reproduced exactly")
    return parity


def rank_for_signal(
    artifacts: dict[str, t2.RetrievalArtifacts],
    signal: str,
    query_index: int,
    candidate_id: int,
) -> Optional[int]:
    return t2.rank_in_results(artifacts[signal].ids[query_index], candidate_id)


def recovery_flags(s1: pd.Series, true_record: dict[str, str]) -> str:
    flags = []
    s1_name = "" if pd.isna(s1["business_name"]) else str(s1["business_name"])
    true_name = true_record["business_name"]
    s1_address = "" if pd.isna(s1["business_address"]) else str(s1["business_address"])
    true_address = true_record["business_address"]
    if t2.contains_non_ascii(s1_name) != t2.contains_non_ascii(true_name):
        flags.append("cross_script_name")
    if not t2.normalize_value(s1_address) or not t2.normalize_value(true_address):
        flags.append("missing_address")
    if set(number_tokens(s1_address)) != set(number_tokens(true_address)):
        flags.append("address_number_difference")
    if (
        t2.normalize_value(s1_name) != t2.normalize_value(true_name)
        and strip_legal_suffix_value(s1_name) == strip_legal_suffix_value(true_name)
    ):
        flags.append("legal_suffix_variation")
    return ";".join(flags) if flags else "rank_or_lexical_improvement"


def build_recovery_analysis(
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    true_records: dict[int, dict[str, str]],
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    artifacts: dict[str, t2.RetrievalArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_sets = [
        set(configuration_candidates(BASELINE_UNION, index, eval_df, exact_map, artifacts))
        for index in range(len(eval_df))
    ]
    baseline_hits = sum(
        candidate_id in baseline_sets[index]
        for index, truth in enumerate(truth_lists)
        for candidate_id in truth
    )
    baseline_misses = sum(map(len, truth_lists)) - baseline_hits
    detail_rows = []
    summary_rows = []
    for configuration in CONFIGURATIONS:
        if configuration == BASELINE_UNION:
            continue
        recovered = lost = 0
        for query_index, truth in enumerate(truth_lists):
            candidate_set = set(
                configuration_candidates(
                    configuration, query_index, eval_df, exact_map, artifacts
                )
            )
            for candidate_id in truth:
                baseline_hit = candidate_id in baseline_sets[query_index]
                current_hit = candidate_id in candidate_set
                if not baseline_hit and current_hit:
                    recovered += 1
                    s1 = eval_df.iloc[query_index]
                    true_record = true_records[candidate_id]
                    detail_rows.append(
                        {
                            "configuration": configuration,
                            "s1_entity_id": str(s1["source1_entity_id"]),
                            "true_matched_entity_id": t2.decode_candidate_id(candidate_id),
                            "matched_source": t2.candidate_source(candidate_id),
                            "match_group": str(s1["match_group"]),
                            "s1_name": "" if pd.isna(s1["business_name"]) else str(s1["business_name"]),
                            "true_name": true_record["business_name"],
                            "s1_address": "" if pd.isna(s1["business_address"]) else str(s1["business_address"]),
                            "true_address": true_record["business_address"],
                            "recovery_flags": recovery_flags(s1, true_record),
                            "baseline_name_rank": rank_for_signal(
                                artifacts, "baseline_name", query_index, candidate_id
                            ),
                            "baseline_address_rank": rank_for_signal(
                                artifacts, "baseline_address", query_index, candidate_id
                            ),
                            "transliterated_name_rank": rank_for_signal(
                                artifacts, "transliterated_name", query_index, candidate_id
                            ),
                            "number_address_rank": rank_for_signal(
                                artifacts, "number_address", query_index, candidate_id
                            ),
                            "suffix_name_rank": rank_for_signal(
                                artifacts, "suffix_name", query_index, candidate_id
                            ),
                        }
                    )
                elif baseline_hit and not current_hit:
                    lost += 1
        summary_rows.append(
            {
                "configuration": configuration,
                "baseline_positive_hits": baseline_hits,
                "baseline_positive_misses": baseline_misses,
                "recovered_baseline_misses": recovered,
                "recovery_rate_of_baseline_misses": recovered / baseline_misses,
                "lost_baseline_hits": lost,
                "net_positive_link_gain": recovered - lost,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def get_metric(
    table: pd.DataFrame,
    configuration: str,
    filter_column: str,
    filter_value: str,
    metric: str,
) -> float:
    row = table.loc[
        table["configuration"].eq(configuration)
        & table[filter_column].astype(str).eq(str(filter_value))
    ]
    if len(row) != 1:
        raise ValueError(f"Could not resolve {configuration}/{filter_column}={filter_value}")
    return float(row.iloc[0][metric])


def select_configuration(
    ablation_df: pd.DataFrame,
    source_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> dict[str, object]:
    baseline = ablation_df.loc[ablation_df["configuration"].eq(BASELINE_UNION)].iloc[0]
    baseline_s3 = get_metric(source_df, BASELINE_UNION, "source", "S3", "link_recall")
    baseline_6_complete = get_metric(
        group_df, BASELINE_UNION, "match_group", "6+", "complete_recall"
    )
    operational = ablation_df.loc[ablation_df["max_candidates"].le(250)].copy()
    eligible_rows = []
    for row in operational.itertuples(index=False):
        if row.configuration in {BASELINE_RANKED, BASELINE_UNION}:
            continue
        s3_recall = get_metric(source_df, row.configuration, "source", "S3", "link_recall")
        six_complete = get_metric(
            group_df, row.configuration, "match_group", "6+", "complete_recall"
        )
        meaningful = row.link_recall - baseline["link_recall"] >= 0.005
        no_global_complete_harm = row.complete_recall >= baseline["complete_recall"]
        no_s3_harm = s3_recall >= baseline_s3 - 0.001
        no_high_match_harm = six_complete >= baseline_6_complete - 0.005
        if meaningful and no_global_complete_harm and no_s3_harm and no_high_match_harm:
            eligible_rows.append(row)

    if eligible_rows:
        selected = sorted(
            eligible_rows,
            key=lambda row: (
                -row.link_recall,
                -row.complete_recall,
                row.average_candidates,
            ),
        )[0]
        reason = (
            "Selected because it improved link recall by at least 0.5 percentage points "
            "within a 250-candidate cap without degrading global complete recall, S3 recall, "
            "or 6+-match complete recall beyond the allowed tolerances."
        )
    else:
        selected = next(
            row for row in ablation_df.itertuples(index=False) if row.configuration == BASELINE_UNION
        )
        reason = (
            "The existing baseline was retained because no operational configuration under "
            "the 250-candidate cap produced a meaningful recall improvement without material trade-offs."
        )

    selected_name = selected.configuration
    source_metrics = {
        source: get_metric(source_df, selected_name, "source", source, "link_recall")
        for source in ("S2", "S3")
    }
    group_metrics = {
        group: {
            "link_recall": get_metric(
                group_df, selected_name, "match_group", group, "link_recall"
            ),
            "complete_recall": get_metric(
                group_df, selected_name, "match_group", group, "complete_recall"
            ),
        }
        for group in ("1", "2", "3-5", "6+")
    }
    return {
        "selected_configuration": selected_name,
        "signals": configuration_required_signals(selected_name),
        "candidate_budget": int(selected.max_candidates),
        "link_recall": float(selected.link_recall),
        "complete_recall": float(selected.complete_recall),
        "average_candidates": float(selected.average_candidates),
        "median_candidates": float(selected.median_candidates),
        "p95_candidates": float(selected.p95_candidates),
        "maximum_candidates": int(selected.max_candidates),
        "s2_link_recall": source_metrics["S2"],
        "s3_link_recall": source_metrics["S3"],
        "recall_by_match_group": group_metrics,
        "selection_rule_minimum_link_recall_gain": 0.005,
        "reason": reason,
        "task3_classifier_work_performed": False,
        "test_data_used": False,
    }


def pair_similarity(model: t2.TextModel, left: str, right: str) -> Optional[float]:
    if not left or not right:
        return None
    matrix = model.vectorizer.transform([left, right])
    return float(matrix[0].multiply(matrix[1]).sum())


def remaining_failure_tags(
    s1: pd.Series,
    true_record: dict[str, str],
    similarities: dict[str, Optional[float]],
    ranks: dict[str, Optional[int]],
) -> str:
    tags = []
    s1_name = "" if pd.isna(s1["business_name"]) else str(s1["business_name"])
    true_name = true_record["business_name"]
    s1_address = "" if pd.isna(s1["business_address"]) else str(s1["business_address"])
    true_address = true_record["business_address"]
    if not t2.normalize_value(s1_address) or not t2.normalize_value(true_address):
        tags.append("missing_address")
    if t2.contains_non_ascii(s1_name) != t2.contains_non_ascii(true_name):
        tags.append("cross_script_name")
    if set(number_tokens(s1_address)) != set(number_tokens(true_address)):
        tags.append("address_number_difference")
    if (
        t2.normalize_value(s1_name) != t2.normalize_value(true_name)
        and strip_legal_suffix_value(s1_name) == strip_legal_suffix_value(true_name)
    ):
        tags.append("legal_suffix_variation")
    if similarities.get("transliterated_name") is not None and similarities["transliterated_name"] < 0.2:
        tags.append("very_different_transliterated_name")
    if similarities.get("number_address") is not None and similarities["number_address"] < 0.2:
        tags.append("very_different_number_address")
    if all(rank is None or rank > 100 for rank in ranks.values()):
        tags.append("all_signal_ranks_beyond_100")
    if all(rank is None for rank in ranks.values()):
        tags.append("not_in_top250_any_signal")
    return ";".join(tags) if tags else "rank_fusion_or_budget_limit"


def build_remaining_failure_analysis(
    selected_configuration: str,
    eval_df: pd.DataFrame,
    truth_lists: list[list[int]],
    true_records: dict[int, dict[str, str]],
    exact_map: dict[tuple[str, str], dict[str, list[int]]],
    artifacts: dict[str, t2.RetrievalArtifacts],
    baseline_models: dict[tuple[str, str], t2.TextModel],
    variant_models: dict[tuple[str, str], t2.TextModel],
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    misses = []
    for query_index, truth in enumerate(truth_lists):
        candidate_set = set(
            configuration_candidates(
                selected_configuration, query_index, eval_df, exact_map, artifacts
            )
        )
        misses.extend(
            (query_index, candidate_id)
            for candidate_id in truth
            if candidate_id not in candidate_set
        )
    rng = np.random.default_rng(RANDOM_STATE)
    positions = rng.choice(len(misses), size=min(sample_size, len(misses)), replace=False)
    rows = []
    for position in positions:
        query_index, candidate_id = misses[int(position)]
        s1 = eval_df.iloc[query_index]
        true_record = true_records[candidate_id]
        partition = str(s1["normalized_country"])
        s1_values = {
            "baseline_name": str(s1["normalized_name"]),
            "baseline_address": str(s1["normalized_address"]),
            "transliterated_name": transliterate_value(s1["business_name"]),
            "suffix_name": strip_legal_suffix_value(s1["business_name"]),
            "number_address": address_number_value(s1["business_address"]),
        }
        true_values = {
            "baseline_name": t2.normalize_value(true_record["business_name"]),
            "baseline_address": t2.normalize_value(true_record["business_address"]),
            "transliterated_name": transliterate_value(true_record["business_name"]),
            "suffix_name": strip_legal_suffix_value(true_record["business_name"]),
            "number_address": address_number_value(true_record["business_address"]),
        }
        similarities = {}
        ranks = {}
        for signal in (
            "baseline_name",
            "baseline_address",
            "transliterated_name",
            "number_address",
            "suffix_name",
        ):
            field = "name" if signal == "baseline_name" else "address"
            model = (
                baseline_models[(partition, field)]
                if signal.startswith("baseline_")
                else variant_models[(partition, signal)]
            )
            similarities[signal] = pair_similarity(
                model, s1_values[signal], true_values[signal]
            )
            ranks[signal] = rank_for_signal(artifacts, signal, query_index, candidate_id)
        tags = remaining_failure_tags(s1, true_record, similarities, ranks)
        rows.append(
            {
                "selected_configuration": selected_configuration,
                "s1_entity_id": str(s1["source1_entity_id"]),
                "true_matched_entity_id": t2.decode_candidate_id(candidate_id),
                "matched_source": t2.candidate_source(candidate_id),
                "match_count": int(s1["match_count"]),
                "match_group": str(s1["match_group"]),
                "s1_name": "" if pd.isna(s1["business_name"]) else str(s1["business_name"]),
                "true_name": true_record["business_name"],
                "s1_address": "" if pd.isna(s1["business_address"]) else str(s1["business_address"]),
                "true_address": true_record["business_address"],
                **{f"{signal}_similarity": similarities[signal] for signal in similarities},
                **{f"{signal}_rank_up_to_250": ranks[signal] for signal in ranks},
                "failure_patterns": tags,
            }
        )
    failures = pd.DataFrame(rows)
    counter = Counter()
    for value in failures["failure_patterns"].fillna(""):
        for tag in str(value).split(";"):
            if tag:
                counter[tag] += 1
    patterns = pd.DataFrame(
        [
            {"failure_pattern": tag, "sampled_remaining_failures": count}
            for tag, count in counter.most_common()
        ]
    )
    return failures, patterns


def signal_runtime_table(
    artifacts: dict[str, t2.RetrievalArtifacts],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for signal, artifact in artifacts.items():
        rows.append(
            {
                "signal": signal,
                "indexed_records": artifact.ntotal,
                "build_seconds": artifact.build_seconds,
                "query_seconds": artifact.query_seconds,
                "total_seconds": artifact.build_seconds + artifact.query_seconds,
                "estimated_compressed_index_mb": artifact.estimated_index_mb,
                "retrieval_ids_mb": artifact.ids.nbytes / (1024.0 * 1024.0),
                "retrieval_scores_mb": artifact.scores.nbytes / (1024.0 * 1024.0),
                "search_depth": (
                    args.baseline_search_k
                    if signal.startswith("baseline_")
                    else args.search_k
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    output_path: Path,
    ablation_df: pd.DataFrame,
    source_df: pd.DataFrame,
    group_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    failure_patterns_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    decision: dict[str, object],
    total_runtime_seconds: float,
    peak_rss_mb: float,
) -> None:
    lines = [
        "TASK 2.5 - CANDIDATE GENERATOR SELECTION",
        "",
        "Scope",
        "-----",
        "Training data only. The exact saved Task 2 evaluation set was reused.",
        "No pairwise classifier, threshold, calibration, test prediction, or submission was created.",
        "",
        "Ablation results",
        "----------------",
    ]
    for row in ablation_df.itertuples(index=False):
        lines.append(
            f"{row.configuration}: link recall={row.link_recall:.4%}, "
            f"complete recall={row.complete_recall:.4%}, avg candidates={row.average_candidates:.2f}, "
            f"p95={row.p95_candidates:.0f}, max={int(row.max_candidates)}"
        )
    lines.extend(["", "Recovered Task 2 baseline misses", "--------------------------------"])
    for row in recovery_df.itertuples(index=False):
        lines.append(
            f"{row.configuration}: recovered={row.recovered_baseline_misses:,}, "
            f"lost={row.lost_baseline_hits:,}, net={row.net_positive_link_gain:,}"
        )
    selected = decision["selected_configuration"]
    lines.extend(
        [
            "",
            "Frozen candidate generator",
            "--------------------------",
            f"Selected configuration: {selected}",
            f"Signals: {', '.join(decision['signals'])}",
            f"Candidate budget: {decision['candidate_budget']}",
            f"Link recall: {decision['link_recall']:.4%}",
            f"Complete recall: {decision['complete_recall']:.4%}",
            f"Average candidates: {decision['average_candidates']:.2f}",
            f"Median candidates: {decision['median_candidates']:.2f}",
            f"P95 candidates: {decision['p95_candidates']:.0f}",
            f"Maximum candidates: {decision['maximum_candidates']}",
            f"S2 link recall: {decision['s2_link_recall']:.4%}",
            f"S3 link recall: {decision['s3_link_recall']:.4%}",
            f"Reason: {decision['reason']}",
            "",
            "Selected recall by match-count group",
            "------------------------------------",
        ]
    )
    for group in ("1", "2", "3-5", "6+"):
        metrics = decision["recall_by_match_group"][group]
        lines.append(
            f"{group}: link recall={metrics['link_recall']:.4%}, "
            f"complete recall={metrics['complete_recall']:.4%}"
        )
    lines.extend(["", "Remaining failure patterns", "--------------------------"])
    if failure_patterns_df.empty:
        lines.append("No remaining failures in the sampled evaluation links.")
    else:
        for row in failure_patterns_df.itertuples(index=False):
            lines.append(f"{row.failure_pattern}: {row.sampled_remaining_failures}")
    lines.extend(
        [
            "",
            "Runtime and memory",
            "------------------",
            f"Total Task 2.5 runtime: {total_runtime_seconds / 60.0:.2f} minutes",
            f"Observed peak process RSS: {peak_rss_mb:.1f} MB",
        ]
    )
    for row in runtime_df.itertuples(index=False):
        lines.append(
            f"{row.signal}: build={row.build_seconds:.1f}s, query={row.query_seconds:.1f}s, "
            f"estimated index={row.estimated_compressed_index_mb:.1f} MB"
        )
    lines.extend(
        [
            "",
            "Known limitations",
            "-----------------",
            "Transliteration is approximate and may not preserve all language-specific phonetics.",
            "Address-number normalization improves formatting tolerance but cannot resolve genuinely conflicting or missing numbers.",
            "Lexically unrelated aliases and records with both weak names and missing addresses may remain outside the candidate budget.",
            "The frozen choice is based on the fixed 10,000-entity training evaluation set and should be monitored during Task 3 validation.",
            "",
            "STOP: Task 2.5 ends with candidate-generator selection.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = make_parser().parse_args()
    args.data_dir = args.data_dir.resolve()
    args.task2_output_dir = args.task2_output_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = validate_paths(args)
    faiss.omp_set_num_threads(args.faiss_threads)

    if args.smoke_only:
        run_smoke_validation(paths, args.output_dir)
        return

    total_started = time.perf_counter()
    smoke_path = args.output_dir / "task2_5_smoke_validation.json"
    if not smoke_path.exists():
        raise FileNotFoundError(
            "Run Task 2.5 with --smoke-only before the full experiment"
        )
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed":
        raise ValueError("Task 2.5 smoke validation did not pass")

    eval_df, truth_lists, truth_strings = load_evaluation(paths["eval"])
    training_sample, true_records, source_counts = collect_training_sample_and_truth_records(
        paths,
        truth_strings,
        args.chunksize,
        args.sample_modulus,
    )
    training_sample = t2.trim_training_sample(
        training_sample, args.train_sample_per_source_country
    )

    baseline_models, partitions = t2.build_text_models(
        training_sample,
        eval_df,
        True,
        args.max_features,
        args.projection_dim,
    )
    query_keys = t2.prepare_exact_query_keys(eval_df, True)
    exact_map: dict[tuple[str, str], dict[str, list[int]]] = {
        (partition, field): defaultdict(list)
        for partition in partitions
        for field in t2.FIELDS
    }
    baseline_name = t2.build_field_index_and_retrieve(
        "name",
        paths,
        eval_df,
        baseline_models,
        partitions,
        True,
        query_keys,
        exact_map,
        args.chunksize,
        args.transform_batch_size,
        args.projection_dim,
        args.nlist,
        args.pq_m,
        args.nprobe,
        args.baseline_search_k,
        collect_exact=True,
    )
    baseline_address = t2.build_field_index_and_retrieve(
        "address",
        paths,
        eval_df,
        baseline_models,
        partitions,
        True,
        query_keys,
        exact_map,
        args.chunksize,
        args.transform_batch_size,
        args.projection_dim,
        args.nlist,
        args.pq_m,
        args.nprobe,
        args.baseline_search_k,
        collect_exact=False,
    )

    variant_specs = [
        SignalSpec("transliterated_name", "business_name", transliterate_series),
        SignalSpec("suffix_name", "business_name", strip_legal_suffix_series),
        SignalSpec("number_address", "business_address", address_number_series),
    ]
    variant_models, variant_partitions = build_variant_models(
        training_sample,
        eval_df,
        variant_specs,
        args.max_features,
        args.projection_dim,
    )
    if variant_partitions != partitions:
        raise ValueError("Task 2 and Task 2.5 country partitions differ")
    name_variant_artifacts = build_variant_indexes_and_retrieve(
        variant_specs[:2],
        paths,
        eval_df,
        variant_models,
        partitions,
        args,
    )
    number_variant_artifacts = build_variant_indexes_and_retrieve(
        variant_specs[2:],
        paths,
        eval_df,
        variant_models,
        partitions,
        args,
    )
    artifacts = {
        "baseline_name": baseline_name,
        "baseline_address": baseline_address,
        **name_variant_artifacts,
        **number_variant_artifacts,
    }

    # Persist the expensive retrieval stage before report generation. Only the
    # first 250 neighbors are needed by every Task 2.5 configuration.
    retrieval_checkpoint = args.output_dir / "task2_5_retrieval_top250.npz"
    np.savez_compressed(
        retrieval_checkpoint,
        **{
            f"{signal}_{kind}": getattr(artifact, kind)[:, :250]
            for signal, artifact in artifacts.items()
            for kind in ("ids", "scores")
        },
    )
    log(f"Saved retrieval checkpoint: {retrieval_checkpoint.name}")

    ablation_df, source_df, group_df, availability_df = evaluate_all(
        eval_df,
        truth_lists,
        true_records,
        exact_map,
        artifacts,
    )
    parity_df = verify_baseline_parity(ablation_df, paths["task2_strategies"])
    recovery_df, recovered_links_df = build_recovery_analysis(
        eval_df,
        truth_lists,
        true_records,
        exact_map,
        artifacts,
    )
    decision = select_configuration(ablation_df, source_df, group_df)
    failures_df, failure_patterns_df = build_remaining_failure_analysis(
        str(decision["selected_configuration"]),
        eval_df,
        truth_lists,
        true_records,
        exact_map,
        artifacts,
        baseline_models,
        variant_models,
        args.failure_sample_size,
    )
    runtime_df = signal_runtime_table(artifacts, args)
    peak_rss_mb = t2.current_peak_rss_mb()
    ablation_df["observed_pipeline_peak_rss_mb"] = peak_rss_mb

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(args.output_dir / "task2_5_ablation_comparison.csv", index=False)
    source_df.to_csv(args.output_dir / "task2_5_recall_by_source.csv", index=False)
    group_df.to_csv(args.output_dir / "task2_5_recall_by_match_count.csv", index=False)
    availability_df.to_csv(
        args.output_dir / "task2_5_recall_by_address_availability.csv", index=False
    )
    parity_df.to_csv(args.output_dir / "task2_5_baseline_parity.csv", index=False)
    recovery_df.to_csv(args.output_dir / "task2_5_recovery_summary.csv", index=False)
    recovered_links_df.to_csv(
        args.output_dir / "task2_5_recovered_links.csv", index=False
    )
    failures_df.to_csv(args.output_dir / "task2_5_remaining_failures.csv", index=False)
    failure_patterns_df.to_csv(
        args.output_dir / "task2_5_failure_patterns.csv", index=False
    )
    runtime_df.to_csv(args.output_dir / "task2_5_signal_runtime_memory.csv", index=False)
    (args.output_dir / "task2_5_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total_runtime_seconds = elapsed(total_started)
    manifest = {
        "training_only": True,
        "task2_eval_sha256": dataframe_sha256(paths["eval"]),
        "evaluation_entities": len(eval_df),
        "evaluation_positive_links": sum(map(len, truth_lists)),
        "source_candidate_counts": source_counts,
        "random_state": RANDOM_STATE,
        "parameters": {
            "max_features": args.max_features,
            "projection_dim": args.projection_dim,
            "nlist": args.nlist,
            "pq_m": args.pq_m,
            "nprobe": args.nprobe,
            "baseline_search_k": args.baseline_search_k,
            "new_signal_search_k": args.search_k,
        },
        "smoke_validation": smoke,
        "baseline_parity_passed": bool(parity_df["parity_passed"].all()),
        "total_runtime_seconds": total_runtime_seconds,
        "observed_peak_rss_mb": peak_rss_mb,
        "classifier_work_performed": False,
        "test_data_used": False,
        "test_predictions_created": False,
        "submission_created": False,
    }
    (args.output_dir / "task2_5_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(
        args.output_dir / "task2_5_summary.txt",
        ablation_df,
        source_df,
        group_df,
        recovery_df,
        failure_patterns_df,
        runtime_df,
        decision,
        total_runtime_seconds,
        peak_rss_mb,
    )
    log(
        f"Task 2.5 complete in {total_runtime_seconds / 60.0:.2f} minutes. "
        f"Selected: {decision['selected_configuration']}"
    )


if __name__ == "__main__":
    main()
