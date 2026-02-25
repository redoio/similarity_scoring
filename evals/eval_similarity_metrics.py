# eval_similarity_metrics.py
from __future__ import annotations

import os
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import config as CFG
import compute_metrics as cm

# Try local or package-style import
try:
    import similarity_metrics as sim
except Exception:
    from . import similarity_metrics as sim  # type: ignore


def _safe_isnan(x) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def _is_constant(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return True
    return s.nunique() <= 1

def compute_feature_vectors(
    demo: pd.DataFrame,
    cur: pd.DataFrame,
    pri: pd.DataFrame,
    n_ids: int = 50,
    seed: int = 7,
    max_pool: int = 5000,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Sample IDs and compute named feature dict for each, but KEEP ONLY IDs
    that have at least MIN_OVERLAP_FOR_SIMILARITY valid (non-NaN) features.

    This avoids the "overlap too small" / NaN similarity issue when vectors
    are sparse (e.g., age/frequency missing).
    """
    rng = random.Random(seed)
    id_col = CFG.COLS["id"]

    all_ids = (
        demo[id_col].astype(str).dropna().unique().tolist()
        if id_col in demo.columns
        else []
    )
    if len(all_ids) == 0:
        raise RuntimeError(f"No IDs found in demographics column '{id_col}'.")

    rng.shuffle(all_ids)
    pool = all_ids[: min(max_pool, len(all_ids))]

    vecs: Dict[str, Dict[str, float]] = {}
    kept: List[str] = []

    min_k = int(getattr(CFG, "MIN_OVERLAP_FOR_SIMILARITY", 3))

    for uid in pool:
        feats, _aux = cm.compute_features(uid, demo, cur, pri, CFG.OFFENSE_LISTS)

        # Keep only numeric, non-NaN values
        feats_clean = {}
        for k, v in feats.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv):
                continue
            feats_clean[k] = fv

        # IMPORTANT: filter out sparse vectors
        if len(feats_clean) < min_k:
            continue

        vecs[uid] = feats_clean
        kept.append(uid)

        if len(kept) >= n_ids:
            break

    if len(kept) < n_ids:
        print(
            f"[WARN] Only found {len(kept)} IDs with >= {min_k} valid features "
            f"(requested {n_ids}). Consider increasing max_pool or enabling more features "
            f"(e.g., age_years, freq bounds)."
        )

    return kept, vecs


def similarity_scores_for_query(
    qid: str,
    vecs: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Compute similarity scores from qid to all other IDs for multiple metrics."""
    qv = vecs[qid]
    rows = []
    for oid, ov in vecs.items():
        if oid == qid:
            continue

        cos = sim.cosine_similarity_named(qv, ov)
        euc = sim.euclidean_similarity_named(qv, ov)
        tan = sim.tanimoto_similarity_named(qv, ov)

        rows.append(
            {"query": qid, "other": oid, "cosine": cos, "euclidean": euc, "tanimoto": tan}
        )
    df = pd.DataFrame(rows)

    # Drop NaNs such as when not enough overlap
    for col in ["cosine", "euclidean", "tanimoto"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["cosine", "euclidean", "tanimoto"], how="any")


def topk_overlap(a: List[str], b: List[str]) -> float:
    """Jaccard overlap of two top-k neighbor ID lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return math.nan
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    out_dir = "similarity_eval_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Load raw tables
    demo = cm.read_table(CFG.PATHS["demographics"])
    cur = cm.read_table(CFG.PATHS["current_commitments"])
    pri = cm.read_table(CFG.PATHS["prior_commitments"])

    # 1) sample IDs + compute vectors
    ids, vecs = compute_feature_vectors(demo, cur, pri, n_ids=100, seed=7)
    print(f"Computed vectors for {len(ids)} IDs.")

    # 2) pick query IDs (this can be changed)
    query_ids = ids
    # query_ids = ids[:10]  # 10 queries for analysis

    overlaps = []
    spearmans = []

    # 3) per-query similarity comparisons
    for qid in query_ids:
        df = similarity_scores_for_query(qid, vecs)

        if df.empty:
            print(f"[WARN] No valid comparisons for query {qid} (overlap too small).")
            continue

        # Rank correlation (Spearman) across candidates
        r_ce = math.nan
        r_ct = math.nan
        r_et = math.nan

        if not (_is_constant(df["cosine"]) or _is_constant(df["euclidean"])):
            r_ce, _ = spearmanr(df["cosine"], df["euclidean"], nan_policy="omit")

        if not (_is_constant(df["cosine"]) or _is_constant(df["tanimoto"])):
            r_ct, _ = spearmanr(df["cosine"], df["tanimoto"], nan_policy="omit")

        if not (_is_constant(df["euclidean"]) or _is_constant(df["tanimoto"])):
            r_et, _ = spearmanr(df["euclidean"], df["tanimoto"], nan_policy="omit")

        spearmans.append(
            {
                "query": qid,
                "spearman_cosine_euclidean": r_ce,
                "spearman_cosine_tanimoto": r_ct,
                "spearman_euclidean_tanimoto": r_et,
                "n_pairs": len(df),
            }
        )

        # Top-K neighbor overlap
        K = 20
        top_cos = df.sort_values("cosine", ascending=False)["other"].head(K).tolist()
        top_euc = df.sort_values("euclidean", ascending=False)["other"].head(K).tolist()
        top_tan = df.sort_values("tanimoto", ascending=False)["other"].head(K).tolist()

        overlaps.append(
            {
                "query": qid,
                "overlap_cos_euc": topk_overlap(top_cos, top_euc),
                "overlap_cos_tan": topk_overlap(top_cos, top_tan),
                "overlap_euc_tan": topk_overlap(top_euc, top_tan),
            }
        )

        # Scatterplots for this query
        plt.figure()
        plt.scatter(df["cosine"], df["euclidean"])
        plt.xlabel("Cosine similarity")
        plt.ylabel("Euclidean similarity")
        plt.title(f"{qid}: Cosine vs Euclidean (n={len(df)})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{qid}_scatter_cosine_vs_euclidean.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(df["cosine"], df["tanimoto"])
        plt.xlabel("Cosine similarity")
        plt.ylabel("Tanimoto similarity")
        plt.title(f"{qid}: Cosine vs Tanimoto (n={len(df)})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{qid}_scatter_cosine_vs_tanimoto.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(df["euclidean"], df["tanimoto"])
        plt.xlabel("Euclidean similarity")
        plt.ylabel("Tanimoto similarity")
        plt.title(f"{qid}: Euclidean vs Tanimoto (n={len(df)})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{qid}_scatter_euclidean_vs_tanimoto.png"), dpi=200)
        plt.close()

    # 4) Save summary tables
    overlap_df = pd.DataFrame(overlaps)
    spearman_df = pd.DataFrame(spearmans)

    overlap_path = os.path.join(out_dir, "topk_overlap_summary.csv")
    spearman_path = os.path.join(out_dir, "spearman_summary.csv")

    overlap_df.to_csv(overlap_path, index=False)
    spearman_df.to_csv(spearman_path, index=False)

    print("\nSaved:")
    print(f" - {overlap_path}")
    print(f" - {spearman_path}")
    print(f" - scatterplots in {out_dir}/")

    # 5) Quick console summary
    if not overlap_df.empty:
        print("\nAverage Top-K overlaps:")
        print(overlap_df[["overlap_cos_euc", "overlap_cos_tan", "overlap_euc_tan"]].mean(numeric_only=True))

    if not spearman_df.empty:
        print("\nAverage Spearman correlations:")
        print(
            spearman_df[
                ["spearman_cosine_euclidean", "spearman_cosine_tanimoto", "spearman_euclidean_tanimoto"]
            ].mean(numeric_only=True)
        )


if __name__ == "__main__":
    main()
