"""Pairwise feature extraction for business entity resolution."""

from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_COLUMNS_V1 = [
    "name_levenshtein_ratio",
    "name_jaccard",
    "name_tfidf_cosine",
    "address_levenshtein_ratio",
    "address_jaccard",
    "address_tfidf_cosine",
    "country_match",
    "name_length_ratio",
    "token_overlap_count",
]
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS_V1 + [
    "name_first_token_match",
    "numeric_token_match",
    "abbreviation_normalized_match",
    "address_component_count_diff",
]
FEATURE_COLUMNS = FEATURE_COLUMNS_V2

_DIGIT_RE = re.compile(r"\d+")
_LEGAL_SUFFIXES = (
    ("pvt", "ltd"), ("llc",), ("llp",), ("pllc",), ("corp",),
    ("inc",), ("co",), ("plc",), ("pc",), ("sa",), ("sas",),
    ("sarl",), ("eurl",), ("sci",), ("jsc",), ("ltd",),
)


def _as_text(values: Iterable[object]) -> list[str]:
    return ["" if pd.isna(value) else str(value) for value in values]


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _length_ratio(left: str, right: str) -> float:
    a, b = len(left), len(right)
    return min(a, b) / max(a, b) if max(a, b) else 1.0


def _strip_legal_suffix(name: str) -> str:
    tokens = name.split()
    changed = True
    while tokens and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            if len(tokens) >= len(suffix) and tuple(tokens[-len(suffix):]) == suffix:
                del tokens[-len(suffix):]
                changed = True
                break
    return " ".join(tokens)


def _component_count(raw_address: object) -> int:
    if pd.isna(raw_address):
        return 0
    return sum(bool(part.strip()) for part in re.split(r"[,;|]+", str(raw_address)))


def _cosine_rows(vectorizer: TfidfVectorizer, left: list[str], right: list[str]) -> np.ndarray:
    left_matrix = vectorizer.transform(left)
    right_matrix = vectorizer.transform(right)
    return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel().astype(np.float32)


class PairFeatureExtractor:
    """Fits name/address TF-IDF vocabularies and builds the v1 pair features.

    ``pairs`` must contain ``source1_entity_id`` and ``candidate_entity_id``.
    Record frames must contain ``entity_id``, raw name/address, country, and the
    normalized name/address columns produced by ``preprocess_dataframe``.
    """

    def __init__(self, max_features: int = 50000, feature_set: str = "v1"):
        if feature_set not in {"v1", "v2"}:
            raise ValueError("feature_set must be 'v1' or 'v2'")
        self.max_features = max_features
        self.feature_set = feature_set
        self.name_vectorizer: Optional[TfidfVectorizer] = None
        self.address_vectorizer: Optional[TfidfVectorizer] = None

    @staticmethod
    def _pair_texts(s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame):
        s1_index = s1.drop_duplicates("entity_id").set_index("entity_id")
        pool_index = pool.drop_duplicates("entity_id").set_index("entity_id")
        left = s1_index.reindex(pairs["source1_entity_id"].tolist())
        right = pool_index.reindex(pairs["candidate_entity_id"].tolist())
        if left["business_name_norm"].isna().any() or right["business_name_norm"].isna().any():
            raise ValueError("Pair references an entity absent from its source record frame")
        return left.reset_index(drop=True), right.reset_index(drop=True)

    def fit(self, s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame):
        left, right = self._pair_texts(s1, pool, pairs)
        name_docs = list(dict.fromkeys(
            _as_text(left.business_name_norm) + _as_text(right.business_name_norm)
        ))
        address_docs = list(dict.fromkeys(
            _as_text(left.business_address_norm) + _as_text(right.business_address_norm)
        ))
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=self.max_features,
            dtype=np.float32,
        ).fit(name_docs)
        self.address_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=self.max_features,
            dtype=np.float32,
        ).fit(address_docs)
        return self

    def transform(self, s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.name_vectorizer is None or self.address_vectorizer is None:
            raise RuntimeError("PairFeatureExtractor must be fit before transform")
        left, right = self._pair_texts(s1, pool, pairs)
        ln = _as_text(left.business_name_norm)
        rn = _as_text(right.business_name_norm)
        la = _as_text(left.business_address_norm)
        ra = _as_text(right.business_address_norm)

        name_lev = np.fromiter((Levenshtein.normalized_similarity(a, b) for a, b in zip(ln, rn)), dtype=np.float32)
        address_lev = np.fromiter((Levenshtein.normalized_similarity(a, b) for a, b in zip(la, ra)), dtype=np.float32)
        name_jaccard = np.fromiter((_jaccard(a, b) for a, b in zip(ln, rn)), dtype=np.float32)
        address_jaccard = np.fromiter((_jaccard(a, b) for a, b in zip(la, ra)), dtype=np.float32)
        overlaps = np.fromiter((len(_tokens(a) & _tokens(b)) for a, b in zip(ln, rn)), dtype=np.float32)
        country = np.fromiter(
            (int(pd.notna(a) and pd.notna(b) and a == b)
             for a, b in zip(left.country, right.country)), dtype=np.float32,
        )

        values = {
            "name_levenshtein_ratio": name_lev,
            "name_jaccard": name_jaccard,
            "name_tfidf_cosine": _cosine_rows(self.name_vectorizer, ln, rn),
            "address_levenshtein_ratio": address_lev,
            "address_jaccard": address_jaccard,
            "address_tfidf_cosine": _cosine_rows(self.address_vectorizer, la, ra),
            "country_match": country,
            "name_length_ratio": np.fromiter((_length_ratio(a, b) for a, b in zip(ln, rn)), dtype=np.float32),
            "token_overlap_count": overlaps,
        }
        if getattr(self, "feature_set", "v1") == "v2":
            first_tokens_left = [a.split() for a in ln]
            first_tokens_right = [b.split() for b in rn]
            first_token_match = np.fromiter(
                (int(bool(a) and bool(b) and a[0] == b[0])
                 for a, b in zip(first_tokens_left, first_tokens_right)), dtype=np.float32,
            )
            numeric_left = [_DIGIT_RE.findall(a) for a in la]
            numeric_right = [_DIGIT_RE.findall(b) for b in ra]
            numeric_match = np.fromiter(
                (int(bool(a) and bool(set(a) & set(b)))
                 for a, b in zip(numeric_left, numeric_right)), dtype=np.float32,
            )
            suffix_left = [_strip_legal_suffix(a) for a in ln]
            suffix_right = [_strip_legal_suffix(b) for b in rn]
            suffix_match = np.fromiter(
                (int(bool(a) and a == b) for a, b in zip(suffix_left, suffix_right)), dtype=np.float32,
            )
            component_left = [_component_count(a) for a in left.business_address]
            component_right = [_component_count(b) for b in right.business_address]
            component_diff = np.fromiter(
                (abs(a - b) for a, b in zip(component_left, component_right)), dtype=np.float32,
            )
            values.update({
                "name_first_token_match": first_token_match,
                "numeric_token_match": numeric_match,
                "abbreviation_normalized_match": suffix_match,
                "address_component_count_diff": component_diff,
            })
        columns = FEATURE_COLUMNS_V2 if getattr(self, "feature_set", "v1") == "v2" else FEATURE_COLUMNS_V1
        return pd.DataFrame(values, columns=columns)
