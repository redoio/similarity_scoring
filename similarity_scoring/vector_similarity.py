# vector_similarity.py
from __future__ import annotations
from typing import Dict, Sequence, List, Optional
import math

# Delegate named-vector logic to similarity_metrics.
# Support both package imports (pytest, from . import) and script imports.
try:
    # Package-relative import (when _Milestone_2 is a package)
    from .similarity_metrics import (
        cosine_similarity_named,
        euclidean_distance_named,
    )
except ImportError:
    # Fallback for direct script usage (python smoke_similarity_test.py)
    from similarity_metrics import (
        cosine_similarity_named,
        euclidean_distance_named,
    )

__all__ = [
    "align_keys",
    "cosine",
    "cosine_from_named",
    "cosine_from_named_weighted",
    "euclidean_from_named",
]


def align_keys(a: Dict[str, float], b: Dict[str, float]) -> list[str]:
    """Sorted intersection of feature names (only shared features are comparable)."""
    return sorted(set(a).intersection(b))


def _to_float_or_none(x) -> Optional[float]:
    """Best-effort cast to float; return None for NaN/inf/bad values."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _finite_pairs(u: Sequence, v: Sequence) -> List[tuple[float, float]]:
    """Return list of (u_i, v_i) where both are finite floats."""
    pairs: List[tuple[float, float]] = []
    for ux, vx in zip(u, v):
        fu = _to_float_or_none(ux)
        fv = _to_float_or_none(vx)
        if fu is not None and fv is not None:
            pairs.append((fu, fv))
    return pairs


# UNWEIGHTED COSINE on raw sequences (no MIN_OVERLAP here)

def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Plain cosine similarity on numeric sequences.

    Returns:
      • NaN if no finite overlapping elements
      • NaN if zero-norm case (undefined)
      • otherwise cosine similarity in [-1, 1]
    """
    pairs = _finite_pairs(u, v)
    if not pairs:
        return math.nan

    num = sum(ux * vx for ux, vx in pairs)
    sum_u2 = sum(ux * ux for ux, _ in pairs)
    sum_v2 = sum(vx * vx for _, vx in pairs)

    nu = math.sqrt(sum_u2)
    nv = math.sqrt(sum_v2)
    if nu == 0.0 or nv == 0.0:
        return math.nan

    return num / (nu * nv)


# NAMED-VECTOR WRAPPERS (Aparna rules via similarity_metrics)

def cosine_from_named(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Cosine similarity between two name-keyed feature dicts.

    Behavior is fully governed by similarity_metrics.cosine_similarity_named:
      • Uses MIN_OVERLAP_FOR_SIMILARITY (from config.py).
      • If overlap < MIN_OVERLAP → NaN.
      • If all intersecting values are 0 → similarity = 1.0.
      • Otherwise standard weighted cosine in [0, 1].
    """
    return float(cosine_similarity_named(a, b))


def cosine_from_named_weighted(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Weighted cosine similarity between two name-keyed feature dicts.

    Delegates to similarity_metrics.cosine_similarity_named with the given weights.
    Weights must be non-negative; non-positive weights are ignored inside the metric.
    """
    if not weights:
        # fallback to unweighted behavior if no weights provided
        return cosine_from_named(a, b)
    return float(cosine_similarity_named(a, b, weights=weights))


def euclidean_from_named(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Euclidean distance between two name-keyed feature dicts.

    Delegates to similarity_metrics.euclidean_distance_named, which:
      • Enforces MIN_OVERLAP_FOR_SIMILARITY via the shared key set.
      • Returns NaN if there are not enough overlapping finite values.
    """
    return float(euclidean_distance_named(a, b))
