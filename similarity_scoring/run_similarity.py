#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_similarity.py — compute features, suitability, and (optionally) similarity
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Any

import config as CFG
import compute_metrics as cm
import sentencing_math as sm

from similarity_metrics import (
    cosine_similarity_named,
    euclidean_distance_named,
    euclidean_similarity_named,
    tanimoto_similarity_named,
    jaccard_similarity_named,
)


def _load_dataframes():
    demo = cm.read_table(CFG.PATHS["demographics"])
    curr = cm.read_table(CFG.PATHS["current_commitments"])
    prior = cm.read_table(CFG.PATHS["prior_commitments"])
    return demo, curr, prior


def _compute_named_features(
    cdcr_id: str,
    demo,
    curr,
    prior,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    lists = getattr(CFG, "OFFENSE_LISTS", {"violent": [], "nonviolent": []})
    feats, aux = cm.compute_features(cdcr_id, demo, curr, prior, lists)
    return feats, aux


def _print_person(cdcr_id: str, feats: Dict[str, float], aux: Dict[str, Any]):
    print(f"\n=== {cdcr_id} ===")
    if not feats:
        print("No computable features (inputs missing or all offenses classified as 'other').")
        return

    for k in sorted(feats.keys()):
        v = feats[k]
        try:
            print(f"{k}: {float(v):.3f}")
        except Exception:
            print(f"{k}: {v}")

    weights = getattr(CFG, "METRIC_WEIGHTS", {})
    directions = getattr(CFG, "METRIC_DIRECTIONS", {})

    ratio, num, denom = sm.suitability_score_named(
        feats,
        weights=weights,
        directions=directions,
        return_parts=True,
    )  # type: ignore[assignment]

    no_denom = denom is None or (isinstance(denom, float) and (math.isnan(denom) or denom == 0.0))

    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)) or no_denom:
        print("Suitability score: NOT EVALUATED (no valid weighted metrics / out_of=0)")
        print(f"  ├─ numerator (Σ w·m): {num}")
        print(f"  └─ out_of (denominator): {denom}")
    else:
        print(f"Suitability score: {float(ratio):.3f}")
        print(f"  ├─ numerator (Σ w·m): {float(num):.3f}")
        print(f"  └─ out_of (denominator): {float(denom):.3f}")

    if aux:
        bits = []
        for key in (
            "age_value",
            "pct_completed",
            "time_outside",
            "months_elapsed_for_frequency",
            "years_elapsed_prior_curr_commitments",
            "years_elapsed_for_trend",
        ):
            if key in aux and aux[key] is not None:
                bits.append(f"{key}={aux[key]}")
        if bits:
            print(f"({', '.join(bits)})")


def main():
    p = argparse.ArgumentParser(description="Compute features, suitability, and similarity.")
    p.add_argument("--cdcr-id", required=True, help="CDCR ID of the primary individual")
    p.add_argument("--compare-id", default=None, help="Optional CDCR ID for similarity comparison")
    args = p.parse_args()

    demo, curr, prior = _load_dataframes()

    feats1, aux1 = _compute_named_features(args.cdcr_id, demo, curr, prior)
    _print_person(args.cdcr_id, feats1, aux1)

    if args.compare_id:
        feats2, aux2 = _compute_named_features(args.compare_id, demo, curr, prior)
        _print_person(args.compare_id, feats2, aux2)

        print(f"\n=== Pairwise similarity: {args.cdcr_id} vs {args.compare_id} ===")

        cos_sim = cosine_similarity_named(feats1, feats2, weights=None)
        euc_dist = euclidean_distance_named(feats1, feats2, weights=None)
        euc_sim = euclidean_similarity_named(feats1, feats2, weights=None)
        tani_sim = tanimoto_similarity_named(feats1, feats2, weights=None)
        jacc = jaccard_similarity_named(feats1, feats2, thresh=0.0)

        def _fmt(name: str, val: Any, extra: str = ""):
            if isinstance(val, float) and math.isnan(val):
                print(f"{name}: NaN (insufficient overlapping valid features / degenerate vectors){extra}")
            else:
                print(f"{name}: {float(val):.3f}{extra}")

        _fmt("Cosine similarity", cos_sim)
        _fmt("Euclidean distance", euc_dist)
        _fmt("Euclidean similarity", euc_sim, "  [1 / (1 + distance)]")
        _fmt("Tanimoto similarity", tani_sim)
        _fmt("Jaccard similarity (active keys)", jacc)


if __name__ == "__main__":
    main()
