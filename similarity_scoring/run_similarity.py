#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_similarity.py — compute features, suitability, and (optionally) similarity

Uses the canonical, library-only modules:
  - config.py                (PATHS, COLS, OFFENSE_LISTS, METRIC_WEIGHTS, METRIC_DIRECTIONS)
  - compute_metrics.py       (read_table, compute_features)
  - sentencing_math.py       (suitability_score_named, suitability_out_of_named)
  - vector_similarity.py     (cosine_from_named)  ← lives in this repo

Examples:
  python run_similarity.py --cdcr-id A1234
  python run_similarity.py --cdcr-id A1234 --compare-id B5678
"""

from __future__ import annotations
import argparse
import math
from typing import Dict, Any, Tuple

import config as CFG
import compute_metrics as cm
import sentencing_math as sm
from vector_similarity import cosine_from_named


def _load_dataframes():
    demo = cm.read_table(CFG.PATHS["demographics"])
    curr = cm.read_table(CFG.PATHS["current_commitments"])
    prior = cm.read_table(CFG.PATHS["prior_commitments"])
    return demo, curr, prior


def _compute_named_features(cdcr_id: str,
                            demo,
                            curr,
                            prior) -> tuple[Dict[str, float], Dict[str, Any]]:
    lists = getattr(CFG, "OFFENSE_LISTS", {"violent": [], "nonviolent": []})
    feats, aux = cm.compute_features(cdcr_id, demo, curr, prior, lists)
    return feats, aux


def _print_person(cdcr_id: str, feats: Dict[str, float], aux: Dict[str, Any]):
    print(f"\n=== {cdcr_id} ===")
    if not feats:
        print("No computable features (inputs likely missing or all offenses → 'other').")
        return

    # deterministic order for readability
    for k in sorted(feats.keys()):
        v = feats[k]
        try:
            print(f"{k}: {float(v):.3f}")
        except Exception:
            print(f"{k}: {v}")

    # Suitability (name-based, uses only present features)
    weights = getattr(CFG, "METRIC_WEIGHTS", {})
    directions = getattr(CFG, "METRIC_DIRECTIONS", {})

    # get full parts for transparency (ratio, numerator, denominator)
    ratio, num, denom = sm.suitability_score_named(
        feats,
        weights=weights,
        directions=directions,
        return_parts=True,
    )  # type: ignore[assignment]

    # NA / not-evaluable guard
    no_denom = (
        denom is None
        or (isinstance(denom, float) and (math.isnan(denom) or denom == 0.0))
    )

    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)) or no_denom:
        print("Suitability score: NOT EVALUATED (no offense-based metrics / out_of=0)")
        print(f"  ├─ numerator (Σ w·m): {num}")
        print(f"  └─ out_of (denominator): {denom}")
    else:
        print(f"Suitability score: {float(ratio):.3f}")
        print(f"  ├─ numerator (Σ w·m): {float(num):.3f}")
        print(f"  └─ out_of (denominator): {float(denom):.3f}")

    # optional quick aux preview
    if aux:
        bits = []
        for key in ("age_value", "pct_completed", "time_outside"):
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

    # Primary person
    feats1, aux1 = _compute_named_features(args.cdcr_id, demo, curr, prior)
    _print_person(args.cdcr_id, feats1, aux1)

    # Optional comparison
    if args.compare_id:
        feats2, aux2 = _compute_named_features(args.compare_id, demo, curr, prior)
        _print_person(args.compare_id, feats2, aux2)

        # Cosine over intersecting keys only (vector_similarity handles intersection)
        sim = cosine_from_named(feats1, feats2)
        print(f"\nCosine similarity ({args.cdcr_id} vs {args.compare_id}): {sim:.3f}")


if __name__ == "__main__":
    main()
