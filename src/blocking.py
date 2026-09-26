"""
ML Challenge 2026 — Business Entity Resolution
Module: src/blocking.py

Implements candidate generation / blocking strategies:
1. TF-IDF char n-grams (analyzer='char_wb', ngram_range=(2,4)) on business_name
   using NearestNeighbors with cosine distance, top_k=20.
2. Token / Address blocking: sorted significant address tokens (dropping stopwords
   like 'road', 'street', etc.) and numeric address keys.
3. Phonetic blocking: jellyfish.metaphone on significant name tokens.

Unions and dedupes the candidate sets per S1 entity into candidate_pairs.tsv.
"""

import sys
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from src.preprocessing import normalize_name, normalize_address, preprocess_dataframe
except ModuleNotFoundError:
    from preprocessing import normalize_name, normalize_address, preprocess_dataframe

# -----------------------------------------------------------------------------
# Stopwords for Blocking Keys
# -----------------------------------------------------------------------------
ADDR_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "court", "circle", "floor",
    "suite", "near", "opposite", "apartment", "building", "sector", "plot",
    "house", "number", "flr", "ste", "apt", "bldg", "opp", "dist", "st", "rd",
    "ave", "dr", "ln", "ct", "cir", "rue", "boulevard", "allee", "chemin",
}

NAME_STOPWORDS = {
    "the", "and", "co", "corp", "corporation", "inc", "incorporated", "ltd",
    "limited", "pvt", "private", "llc", "llp", "pllc", "holding", "holdings",
    "enterprise", "enterprises", "service", "services", "consultancy",
    "assoc", "associates", "partner", "partners", "sa", "sas", "sarl",
}


# -----------------------------------------------------------------------------
# Strategy 1: TF-IDF Char n-grams + NearestNeighbors
# -----------------------------------------------------------------------------
def generate_tfidf_candidates(
    s1_names: List[str],
    s1_ids: List[str],
    pool_names: List[str],
    pool_ids: List[str],
    top_k: int = 20,
    ngram_range: Tuple[int, int] = (2, 4),
    max_features: int = 50000,
    batch_size: int = 5000,
) -> Dict[str, Set[str]]:
    """
    Candidate generation via TF-IDF character n-grams and NearestNeighbors cosine distance.
    Batches kneighbors queries to maintain bounded memory footprint.
    """
    if len(s1_names) == 0 or len(pool_names) == 0:
        return {sid: set() for sid in s1_ids}
    
    print(f"  [TF-IDF] Fitting vectorizer (analyzer='char_wb', ngrams={ngram_range}, max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    # Fit vocabulary on combined S1 + Pool names
    vectorizer.fit(s1_names + pool_names[:min(len(pool_names), 500000)])
    
    print(f"  [TF-IDF] Transforming pool ({len(pool_names):,} records)...")
    X_pool = vectorizer.transform(pool_names)
    
    print(f"  [TF-IDF] Transforming S1 ({len(s1_names):,} records)...")
    X_s1 = vectorizer.transform(s1_names)
    
    print(f"  [TF-IDF] Fitting NearestNeighbors (top_k={top_k}, metric='cosine')...")
    nn = NearestNeighbors(n_neighbors=min(top_k, len(pool_names)), metric="cosine", n_jobs=-1)
    nn.fit(X_pool)
    
    pool_ids_arr = np.array(pool_ids)
    candidates: Dict[str, Set[str]] = defaultdict(set)
    
    print(f"  [TF-IDF] Querying nearest neighbors in batches of {batch_size}...")
    n_queries = len(s1_ids)
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        batch_X = X_s1[start_idx:end_idx]
        _, indices = nn.kneighbors(batch_X)
        for i, global_i in enumerate(range(start_idx, end_idx)):
            sid = s1_ids[global_i]
            matched_pool_ids = pool_ids_arr[indices[i]]
            candidates[sid].update(matched_pool_ids)
            
    return candidates


# -----------------------------------------------------------------------------
# Strategy 2: Address / Token Blocking
# -----------------------------------------------------------------------------
def get_address_blocking_keys(addr_norm: str) -> List[str]:
    """Generates significant token combinations and numeric keys from normalized address."""
    if not addr_norm:
        return []
    words = [w for w in addr_norm.split() if len(w) >= 3 and w not in ADDR_STOPWORDS]
    nums = [w for w in addr_norm.split() if w.isdigit() and len(w) >= 2]
    
    keys = []
    # Sorted significant token combinations
    if len(words) >= 2:
        keys.append(" ".join(sorted(words[:4])))
        if len(words) >= 4:
            keys.append(" ".join(sorted(words[-3:])))
    if words:
        keys.append(words[0])
    # Numeric identifier keys (PIN / building / plot numbers)
    for num in nums[:2]:
        keys.append(f"num_{num}")
    return keys


def generate_address_candidates(
    s1_addrs: List[str],
    s1_ids: List[str],
    pool_addrs: List[str],
    pool_ids: List[str],
    max_bucket_size: int = 150,
) -> Dict[str, Set[str]]:
    """Builds an inverted index on address keys and retrieves candidate IDs."""
    print("  [Address Blocking] Building inverted index on address keys...")
    index: Dict[str, List[str]] = defaultdict(list)
    for pid, addr in zip(pool_ids, pool_addrs):
        for key in get_address_blocking_keys(addr):
            index[key].append(pid)
            
    print(f"  [Address Blocking] Querying {len(s1_ids):,} S1 entities against {len(index):,} keys...")
    candidates: Dict[str, Set[str]] = defaultdict(set)
    for sid, addr in zip(s1_ids, s1_addrs):
        for key in get_address_blocking_keys(addr):
            bucket = index.get(key)
            if bucket and len(bucket) <= max_bucket_size:
                candidates[sid].update(bucket)
                
    return candidates


# -----------------------------------------------------------------------------
# Strategy 3: Phonetic Blocking
# -----------------------------------------------------------------------------
def get_phonetic_blocking_keys(name_norm: str) -> List[str]:
    """Generates phonetic metaphone keys for the first significant name tokens."""
    if not name_norm:
        return []
    words = [w for w in name_norm.split() if len(w) >= 3 and w not in NAME_STOPWORDS]
    keys = []
    for w in words[:2]:
        try:
            m = jellyfish.metaphone(w)
            if m:
                keys.append(m)
        except Exception:
            pass
    return keys


def generate_phonetic_candidates(
    s1_names: List[str],
    s1_ids: List[str],
    pool_names: List[str],
    pool_ids: List[str],
    max_bucket_size: int = 150,
) -> Dict[str, Set[str]]:
    """Builds an inverted index on phonetic keys and retrieves candidate IDs."""
    print("  [Phonetic Blocking] Building inverted index on metaphone keys...")
    index: Dict[str, List[str]] = defaultdict(list)
    for pid, name in zip(pool_ids, pool_names):
        for key in get_phonetic_blocking_keys(name):
            index[key].append(pid)
            
    print(f"  [Phonetic Blocking] Querying {len(s1_ids):,} S1 entities against {len(index):,} keys...")
    candidates: Dict[str, Set[str]] = defaultdict(set)
    for sid, name in zip(s1_ids, s1_names):
        for key in get_phonetic_blocking_keys(name):
            bucket = index.get(key)
            if bucket and len(bucket) <= max_bucket_size:
                candidates[sid].update(bucket)
                
    return candidates


# -----------------------------------------------------------------------------
# Union, Deduplication, and Output Generation
# -----------------------------------------------------------------------------
def union_and_dedupe_candidates(
    s1_ids: List[str],
    tfidf_cands: Dict[str, Set[str]],
    addr_cands: Dict[str, Set[str]],
    phonetic_cands: Dict[str, Set[str]],
    max_candidates_per_entity: Optional[int] = 200,
) -> Dict[str, List[str]]:
    """
    Unions candidates from all 3 strategies and dedupes per S1 entity.
    Guarantees every S1 ID is present in the output dictionary.
    Excludes self-matches (S1 IDs).
    """
    combined: Dict[str, List[str]] = {}
    for sid in s1_ids:
        cset = set()
        cset.update(tfidf_cands.get(sid, set()))
        cset.update(addr_cands.get(sid, set()))
        cset.update(phonetic_cands.get(sid, set()))
        # Remove self-matches (e.g. any S1 IDs if accidentally present)
        cset = {cid for cid in cset if not cid.startswith("S1-") and cid != sid}
        
        # Sort for deterministic output
        cand_list = sorted(cset)
        if max_candidates_per_entity and len(cand_list) > max_candidates_per_entity:
            cand_list = cand_list[:max_candidates_per_entity]
        combined[sid] = cand_list
        
    return combined


def save_candidate_pairs(
    candidate_dict: Dict[str, List[str]],
    output_path: str,
    s1_id_order: Optional[List[str]] = None,
) -> None:
    """
    Writes candidate pairs TSV adhering strictly to the challenge schema:
    source1_entity_id\\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if s1_id_order is None:
        s1_id_order = list(candidate_dict.keys())
        
    print(f"Writing candidate pairs to {output_path} (sep='\\t')...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_id_order:
            cands = candidate_dict.get(sid, [])
            cand_str = ",".join(cands)
            f.write(f"{sid}\t{cand_str}\n")
    print(f"Successfully saved {len(s1_id_order):,} rows to {output_path}.")


def run_blocking_pipeline(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k_tfidf: int = 20,
    output_path: Optional[str] = "output/candidate_pairs.tsv",
) -> Dict[str, List[str]]:
    """
    Runs full 3-strategy blocking pipeline partitioned by country.
    Country partitioning ensures 100% adherence to ER boundaries,
    massively reduces search space, and keeps memory within bounds.
    """
    t_start = time.time()
    print("="*60)
    print("STARTING CANDIDATE GENERATION PIPELINE")
    print("="*60)
    
    # Preprocess inputs if not already normalized
    if "business_name_norm" not in s1_df.columns:
        print("Preprocessing S1 records...")
        s1_df = preprocess_dataframe(s1_df)
    if "business_name_norm" not in s2_df.columns:
        print("Preprocessing S2 records...")
        s2_df = preprocess_dataframe(s2_df)
    if "business_name_norm" not in s3_df.columns:
        print("Preprocessing S3 records...")
        s3_df = preprocess_dataframe(s3_df)
        
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    all_s1_ids = list(s1_df["entity_id"].values)
    all_candidates: Dict[str, List[str]] = {}
    
    # Process country-by-country (open set: US, India, France, or any unseen country)
    unique_countries = s1_df["country"].unique()
    print(f"Countries to process: {list(unique_countries)}")
    
    for country in unique_countries:
        print(f"\n--- Processing Country: {country} ---")
        sub_s1 = s1_df[s1_df["country"] == country]
        sub_pool = pool_df[pool_df["country"] == country]
        
        s1_ids = list(sub_s1["entity_id"].values)
        s1_names = list(sub_s1["business_name_norm"].values)
        s1_addrs = list(sub_s1["business_address_norm"].values)
        
        pool_ids = list(sub_pool["entity_id"].values)
        pool_names = list(sub_pool["business_name_norm"].values)
        pool_addrs = list(sub_pool["business_address_norm"].values)
        
        print(f"Country {country}: {len(s1_ids):,} S1 queries vs {len(pool_ids):,} Pool records")
        
        # Strategy 1: TF-IDF char n-grams
        tfidf_cands = generate_tfidf_candidates(
            s1_names, s1_ids, pool_names, pool_ids, top_k=top_k_tfidf
        )
        
        # Strategy 2: Address / Token
        addr_cands = generate_address_candidates(
            s1_addrs, s1_ids, pool_addrs, pool_ids
        )
        
        # Strategy 3: Phonetic
        phonetic_cands = generate_phonetic_candidates(
            s1_names, s1_ids, pool_names, pool_ids
        )
        
        # Combine
        combined_country = union_and_dedupe_candidates(
            s1_ids, tfidf_cands, addr_cands, phonetic_cands
        )
        all_candidates.update(combined_country)
        
    print("\n" + "="*60)
    print(f"BLOCKING COMPLETE IN {time.time()-t_start:.2f}s")
    total_cands = sum(len(v) for v in all_candidates.values())
    avg_cands = total_cands / max(len(all_s1_ids), 1)
    print(f"Total S1 entities processed: {len(all_s1_ids):,}")
    print(f"Total candidate pairs generated: {total_cands:,} (Avg: {avg_cands:.1f} per S1)")
    print("="*60)
    
    if output_path:
        save_candidate_pairs(all_candidates, output_path, s1_id_order=all_s1_ids)
        
    return all_candidates


if __name__ == "__main__":
    print("src/blocking.py module loaded successfully.")
