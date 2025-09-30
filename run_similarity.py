#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_similarity.py — compute features, suitability, and (optionally) similarity

Uses the canonical, library-only modules:
  - config.py                (PATHS, COLS, OFFENSE_LISTS, METRIC_WEIGHTS)
  - compute_metrics.py       (read_table, compute_features)
  - sentencing_math.py       (suitability_score_named)
  - similarity.py            (cosine_from_named)  ← lives in this repo

Example:
  python run_similarity.py --cdcr-id A1234
  python run_similarity.py --cdcr-id A1234 --compare-id B5678
"""

from __future__ import annotations
import argparse
from typing import Dict, Any

import config as CFG
import compute_metrics as cm
import sentencing_math as sm
# similarity helpers live in this repo (not in sentencing_math)
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
        print("No computable features (inputs likely missing).")
        return
    # deterministic order for readability
    for k in sorted(feats.keys()):
        print(f"{k}: {feats[k]:.3f}")
    # suitability (name-based, uses only present features)
    weights = getattr(CFG, "METRIC_WEIGHTS", getattr(CFG, "WEIGHTS_10D", {}))
    score = sm.suitability_score_named(feats, weights)
    print(f"Suitability score: {score:.3f}")
    # optional quick aux preview
    if "age_value" in aux:
        print(f"(age_value={aux['age_value']}, pct_completed={aux.get('pct_completed')}, time_outside={aux.get('time_outside')})")


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

        # Cosine over intersecting keys only
        sim = cosine_from_named(feats1, feats2)
        print(f"\nCosine similarity ({args.cdcr_id} vs {args.compare_id}): {sim:.3f}")


if __name__ == "__main__":
    main()
