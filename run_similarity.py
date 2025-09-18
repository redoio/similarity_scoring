#!/usr/bin/env python3
"""
run_similarity.py

Unified run script to:
1. Compute sentencing metrics for an individual (compute_metrics_v2.py)
2. Build feature vectors (sentencing_math.py)
3. Use config.py for file paths, offense lists, and weights
"""

import argparse
import pandas as pd

import config as CFG
from compute_metrics_v2 import compute_all_metrics
from sentencing_math import build_feature_vector, cosine_similarity


def main():
    parser = argparse.ArgumentParser(description="Run sentencing metrics + similarity")
    parser.add_argument("--cdcr-id", type=str, required=True,
                        help="CDCR ID of the individual to analyze")
    parser.add_argument("--compare-id", type=str, default=None,
                        help="Optional second CDCR ID for similarity comparison")
    args = parser.parse_args()

    # Load data
    demo = pd.read_excel(CFG.DEMOGRAPHICS_PATH)
    curr = pd.read_excel(CFG.CURRENT_COMMIT_PATH)
    prior = pd.read_excel(CFG.PRIOR_COMMIT_PATH)

    # Compute metrics for the main ID
    print(f"\n=== Computing metrics for {args.cdcr_id} ===")
    m1 = compute_all_metrics(args.cdcr_id, demo, curr, prior, CFG.OFFENSE_LISTS)

    if m1 is None:
        print(f"⚠️ No record found for {args.cdcr_id}")
        return

    vec1 = build_feature_vector(m1, CFG.WEIGHTS)
    print("Feature vector:", vec1)

    # If compare-id is provided, compute similarity
    if args.compare_id:
        print(f"\n=== Comparing {args.cdcr_id} vs {args.compare_id} ===")
        m2 = compute_all_metrics(args.compare_id, demo, curr, prior, CFG.OFFENSE_LISTS)

        if m2 is None:
            print(f" No record found for {args.compare_id}")
            return

        vec2 = build_feature_vector(m2, CFG.WEIGHTS)
        print("Feature vector:", vec2)

        sim = cosine_similarity(vec1, vec2)
        print(f"Cosine similarity: {sim:.3f}")


if __name__ == "__main__":
    main()
