"""Diagnose cross-country truth pairs and unrecovered validation matches.

Run from the repository root:
    python -u diagnose_blocking_recall.py

The validation sample, random seed, background pool, and blocking strategies
mirror evaluate_candidate_recall.py. Detailed misses are written to an ignored
TSV under output/; blocking.py is not modified by this diagnostic.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.blocking import (  # noqa: E402
    generate_address_candidates,
    generate_phonetic_candidates,
    generate_tfidf_candidates,
)
from src.preprocessing import preprocess_dataframe  # noqa: E402

DATA = os.path.join(ROOT, "student_resource", "dataset", "train")


def load_tsv(name, usecols=None):
    return pd.read_csv(
        os.path.join(DATA, name), sep="\t", usecols=usecols,
        dtype={"entity_id": str, "source1_entity_id": str,
               "matched_entity_ids": str, "country": str},
    )


def report_country_consistency(gt):
    """Compare country labels for every match edge in the complete ground truth."""
    id_country = {}
    for filename in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        source = load_tsv(filename, ["entity_id", "country"])
        id_country.update(zip(source["entity_id"], source["country"]))

    edges = gt[["source1_entity_id", "matched_entity_ids"]].dropna().copy()
    edges["matched_entity_ids"] = edges["matched_entity_ids"].str.split(",")
    edges = edges.explode("matched_entity_ids")
    edges["matched_entity_ids"] = edges["matched_entity_ids"].str.strip()
    edges = edges[edges["matched_entity_ids"].ne("")]
    edges["s1_country"] = edges["source1_entity_id"].map(id_country)
    edges["match_country"] = edges["matched_entity_ids"].map(id_country)
    edges["country_differs"] = (
        edges["s1_country"].isna() | edges["match_country"].isna()
        | edges["s1_country"].ne(edges["match_country"])
    )
    cross = edges[edges["country_differs"]]
    print("\n=== Country partition check (complete ground truth) ===")
    print(f"True match pairs checked: {len(edges):,}")
    print(f"Pairs with differing OR missing country label: {len(cross):,}")
    print(f"Pairs with both labels present and differing: "
          f"{int((cross.s1_country.notna() & cross.match_country.notna()).sum()):,}")
    if not cross.empty:
        print("Country discrepancy examples (first 20):")
        print(cross[["source1_entity_id", "s1_country", "matched_entity_ids", "match_country"]]
              .head(20).to_string(index=False))


def main(sample_size=2500, random_state=42, top_k=20, output_path="output/blocking_misses.tsv"):
    started = time.time()
    gt = load_tsv("train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    report_country_consistency(gt)

    val_path = os.path.join(ROOT, "student_resource", "dataset", "val_split_ids.txt")
    with open(val_path, encoding="utf-8") as f:
        val_ids = {line.strip() for line in f if line.strip()}
    gt_val = gt[gt.source1_entity_id.isin(val_ids)].dropna(subset=["matched_entity_ids"])
    if sample_size and sample_size < len(gt_val):
        eval_gt = gt_val.sample(n=sample_size, random_state=random_state)
    else:
        eval_gt = gt_val

    pairs = []
    for sid, raw_matches in zip(eval_gt.source1_entity_id, eval_gt.matched_entity_ids):
        pairs.extend((sid, mid.strip()) for mid in str(raw_matches).split(",") if mid.strip())
    print(f"\nValidation sample: {eval_gt.source1_entity_id.nunique():,} S1 entities; "
          f"{len(pairs):,} true pairs")

    s1_all = load_tsv("train_source1.tsv")
    s2_all = load_tsv("train_source2.tsv")
    s3_all = load_tsv("train_source3.tsv")
    s1_ids = {sid for sid, _ in pairs}
    needed = {mid for _, mid in pairs}
    s1 = preprocess_dataframe(s1_all[s1_all.entity_id.isin(s1_ids)].copy())
    s2_targets = s2_all[s2_all.entity_id.isin(needed)]
    s3_targets = s3_all[s3_all.entity_id.isin(needed)]
    n_neg = min(100000, len(s2_all))
    pool = pd.concat([
        s2_targets, s3_targets,
        s2_all.sample(n=n_neg, random_state=random_state),
        s3_all.sample(n=n_neg, random_state=random_state),
    ]).drop_duplicates(subset=["entity_id"]).copy()
    pool = preprocess_dataframe(pool)

    # Track each strategy independently, exactly as in the evaluation pipeline.
    strategy_hits = {name: defaultdict(set) for name in ("tfidf", "address", "phonetic")}
    tfidf_ranks = {}
    for country in s1.country.drop_duplicates():
        q = s1[s1.country == country]
        p = pool[pool.country == country]
        qids = q.entity_id.tolist()
        pids = p.entity_id.tolist()
        qnames = q.business_name_norm.tolist()
        pnames = p.business_name_norm.tolist()
        qaddrs = q.business_address_norm.tolist()
        paddrs = p.business_address_norm.tolist()
        tf = generate_tfidf_candidates(qnames, qids, pnames, pids, top_k=top_k)
        ad = generate_address_candidates(qaddrs, qids, paddrs, pids)
        ph = generate_phonetic_candidates(qnames, qids, pnames, pids)
        for sid in qids:
            strategy_hits["tfidf"][sid].update(tf.get(sid, ()))
            strategy_hits["address"][sid].update(ad.get(sid, ()))
            strategy_hits["phonetic"][sid].update(ph.get(sid, ()))

        # Exact cosine rank (within this country pool) for true pairs that were
        # not returned in top-k. This separates cutoff misses from representation gaps.
        if not q.empty and not p.empty:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), max_features=50000,
                dtype=np.float32,
            )
            vectorizer.fit(qnames + pnames[:min(len(pnames), 500000)])
            qx = vectorizer.transform(qnames)
            px = vectorizer.transform(pnames)
            pindex = {pid: i for i, pid in enumerate(pids)}
            qindex = {sid: i for i, sid in enumerate(qids)}
            # Only score true pairs in this country. Sparse row products avoid
            # allocating the full query-by-pool similarity matrix.
            relevant = defaultdict(list)
            for sid, mid in pairs:
                if sid in qindex and mid in pindex:
                    relevant[sid].append(mid)
            for sid, mids in relevant.items():
                scores = (px @ qx[qindex[sid]].T).toarray().ravel()
                for mid in mids:
                    score = float(scores[pindex[mid]])
                    tfidf_ranks[(sid, mid)] = int(np.count_nonzero(scores > score) + 1)

    s1_lookup = s1.set_index("entity_id")
    pool_lookup = pool.set_index("entity_id")
    rows = []
    for sid, mid in pairs:
        got = any(mid in strategy_hits[name].get(sid, set())
                  for name in ("tfidf", "address", "phonetic"))
        if got:
            continue
        if sid not in s1_lookup.index or mid not in pool_lookup.index:
            continue
        a, b = s1_lookup.loc[sid], pool_lookup.loc[mid]
        country_excluded = (pd.isna(a.country) or pd.isna(b.country) or a.country != b.country)
        rank = tfidf_ranks.get((sid, mid))
        if country_excluded:
            cause = "excluded by country partition"
        elif rank is not None and rank > top_k:
            cause = f"missed TF-IDF top-{top_k} cutoff"
        else:
            cause = "tokenization/format gap (not TF-IDF cutoff)"
        rows.append({
            "diagnosis": cause, "tfidf_rank_within_country": rank,
            "source1_entity_id": sid, "true_match_entity_id": mid,
            "source1_country": a.country, "match_country": b.country,
            "source1_name": a.business_name, "match_name": b.business_name,
            "source1_address": a.business_address, "match_address": b.business_address,
            "source1_name_norm": a.business_name_norm, "match_name_norm": b.business_name_norm,
            "source1_address_norm": a.business_address_norm, "match_address_norm": b.business_address_norm,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values(["diagnosis", "source1_entity_id", "true_match_entity_id"], inplace=True)
    dest = os.path.join(ROOT, output_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    result.to_csv(dest, sep="\t", index=False)
    print("\n=== Unrecovered validation true pairs ===")
    print(f"Not recovered by any strategy: {len(result):,}")
    if not result.empty:
        print(result.diagnosis.value_counts().to_string())
        borderline = result[result.diagnosis.eq("tokenization/format gap (not TF-IDF cutoff)")]
        if not borderline.empty:
            print("\nPairs with an exact-score rank inside top-k but absent from returned candidates; "
                  "inspect for TF-IDF ties at the cutoff:")
            print(borderline[["source1_entity_id", "true_match_entity_id",
                              "tfidf_rank_within_country"]].to_string(index=False))
    print(f"Detailed side-by-side report: {os.path.relpath(dest, ROOT)}")
    print(f"Total diagnostic time: {time.time() - started:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=2500,
                        help="Validation S1 sample size; 0 means all validation entities")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="output/blocking_misses.tsv")
    args = parser.parse_args()
    main(args.sample, args.seed, args.top_k, args.output)
