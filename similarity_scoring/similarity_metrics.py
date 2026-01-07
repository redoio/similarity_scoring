# similarity_scoring/similarity_metrics.py
from __future__ import annotations

"""
similarity_metrics.py — canonical similarity/distance implementations.

This module is the single source of truth for:
  - MIN_OVERLAP_FOR_SIMILARITY gating
  - NaN behavior when overlap is insufficient or unusable
  - "all-zero overlap" rule (similarity = 1.0 for cosine/tanimoto; jaccard-on-keys = 1.0)
  - consistent weighted handling for magnitude-based metrics

The app/notebook-facing wrapper API lives in vector_similarity.py.
"""

import math
from typing import Dict, Optional, Iterable, Any

import numpy as np
import pandas as pd  # used for gower()


# CONFIG IMPORT
try:
    import config as CFG
    MIN_OVERLAP = int(getattr(CFG, "MIN_OVERLAP_FOR_SIMILARITY", 3))
except Exception:
    MIN_OVERLAP = 3  # fallback if config not available


__all__ = [
    # matrix-level
    "cosine",
    "euclidean",
    "manhattan",
    "jaccard_binary",
    "dice_binary",
    "hamming_binary",
    "gower",

    # named-vector distance/sim pairs (wrapper imports these)
    "cosine_distance_named",
    "cosine_similarity_named",
    "euclidean_distance_named",
    "euclidean_similarity_named",
    "tanimoto_distance_named",
    "tanimoto_similarity_named",

    # backwards compatibility
    "tanimoto_from_named",

    # jaccard on keys
    "jaccard_on_keys",
]


# MATRIX-LEVEL METRICS
def cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (X @ X.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def manhattan(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)


# BINARY SET METRICS (matrix)
def jaccard_binary(B: np.ndarray) -> np.ndarray:
    B = (np.asarray(B) > 0).astype(np.uint8)
    inter = (B[:, None, :] & B[None, :, :]).sum(axis=2)
    union = (B[:, None, :] | B[None, :, :]).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return inter / np.maximum(union, 1)


def dice_binary(B: np.ndarray) -> np.ndarray:
    B = (np.asarray(B) > 0).astype(np.uint8)
    inter = (B[:, None, :] & B[None, :, :]).sum(axis=2)
    sizes = B.sum(axis=1, keepdims=True)
    denom = np.maximum(sizes + sizes.T, 1)
    return (2 * inter) / denom


def hamming_binary(B: np.ndarray) -> np.ndarray:
    B = (np.asarray(B) > 0).astype(np.uint8)
    p = B.shape[1] if B.ndim == 2 else 1
    neq = (B[:, None, :] != B[None, :, :]).sum(axis=2)
    return neq / float(p)


# HELPERS
def _finite(val: Any) -> Optional[float]:
    """Return finite float value, else None."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _intersect(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> list[str]:
    """
    Return overlapping keys of a and b, optionally restricted to keys present in weights.
    Note: this uses key intersection, not "valid numeric" filtering (that happens later).
    """
    ks = set(a) & set(b)
    if weights:
        ks &= set(weights)
    return sorted(ks)


def _check_overlap_and_zero_case(a: Dict[str, float], b: Dict[str, float], keys: list[str]) -> tuple[bool, bool]:
    """
    Applies global MIN_OVERLAP and "all-zero overlap" rule.

    Returns:
        (allowed: bool, all_zero: bool)
    """
    # 1) Not enough overlapping keys
    if len(keys) < MIN_OVERLAP:
        return False, False

    # 2) Are all overlapping finite values exactly zero?
    all_zero = True
    for k in keys:
        x = _finite(a.get(k))
        y = _finite(b.get(k))
        if x is None or y is None:
            continue
        if x != 0.0 or y != 0.0:
            all_zero = False
            break

    return True, all_zero


# COSINE (named vectors)
def cosine_distance_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Cosine distance = 1 - cosine similarity."""
    sim = cosine_similarity_named(a, b, weights=weights)
    return math.nan if isinstance(sim, float) and math.isnan(sim) else 1.0 - sim


def cosine_similarity_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    ks = _intersect(a, b, weights)

    allowed, all_zero = _check_overlap_and_zero_case(a, b, ks)
    if not allowed:
        return math.nan
    if all_zero:
        return 1.0

    num = den_x = den_y = 0.0
    used = False

    for k in ks:
        x = _finite(a.get(k))
        y = _finite(b.get(k))
        if x is None or y is None:
            continue

        w = float(weights.get(k, 1.0) if weights else 1.0)
        if w <= 0:
            continue

        used = True
        num += w * (x * y)
        den_x += w * (x * x)
        den_y += w * (y * y)

    if not used or den_x <= 0.0 or den_y <= 0.0:
        return math.nan

    return num / math.sqrt(den_x * den_y)


# EUCLIDEAN (named vectors)
def euclidean_distance_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    ks = _intersect(a, b, weights)

    allowed, _ = _check_overlap_and_zero_case(a, b, ks)
    if not allowed:
        return math.nan

    used = False
    s = 0.0

    for k in ks:
        x = _finite(a.get(k))
        y = _finite(b.get(k))
        if x is None or y is None:
            continue

        w = float(weights.get(k, 1.0) if weights else 1.0)
        if w <= 0:
            continue

        used = True
        s += w * (x - y) ** 2

    return math.sqrt(s) if used else math.nan


def euclidean_similarity_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Euclidean similarity = 1 / (1 + distance)."""
    dist = euclidean_distance_named(a, b, weights=weights)
    if isinstance(dist, float) and math.isnan(dist):
        return math.nan
    return 1.0 / (1.0 + dist)


# TANIMOTO (named vectors)
def tanimoto_distance_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    sim = tanimoto_similarity_named(a, b, weights=weights)
    return math.nan if isinstance(sim, float) and math.isnan(sim) else 1.0 - sim


def tanimoto_similarity_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    ks = _intersect(a, b, weights)

    allowed, all_zero = _check_overlap_and_zero_case(a, b, ks)
    if not allowed:
        return math.nan
    if all_zero:
        return 1.0

    num = den_x = den_y = 0.0
    used = False

    for k in ks:
        x = _finite(a.get(k))
        y = _finite(b.get(k))
        if x is None or y is None:
            continue

        w = float(weights.get(k, 1.0) if weights else 1.0)
        if w <= 0:
            continue

        used = True
        num += w * (x * y)
        den_x += w * (x * x)
        den_y += w * (y * y)

    den = den_x + den_y - num
    if not used or den <= 0.0:
        return math.nan

    return num / den


def tanimoto_from_named(a: Dict[str, float], b: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Backwards compatibility alias for older notebooks/READMEs."""
    return tanimoto_similarity_named(a, b, weights=weights)


# JACCARD ON KEYS (named vectors)
def jaccard_on_keys(a: Dict[str, float], b: Dict[str, float], thresh: float = 0.0) -> float:
    """
    Jaccard on keys uses which features are "active" (value > thresh).
    MIN_OVERLAP is applied to the INTERSECTION of active keys.
    """
    A = {k for k, v in a.items() if _finite(v) is not None and float(v) > thresh}
    B = {k for k, v in b.items() if _finite(v) is not None and float(v) > thresh}

    ks = A & B

    # MIN OVERLAP RULE (active-key intersection)
    if len(ks) < MIN_OVERLAP:
        return math.nan

    # ALL ZERO case on intersection keys
    all_zero = all(float(a.get(k, 0.0) or 0.0) == 0.0 and float(b.get(k, 0.0) or 0.0) == 0.0 for k in ks)
    if all_zero:
        return 1.0

    inter = len(ks)
    union = len(A | B)
    return inter / union if union > 0 else math.nan


# GOWER SIMILARITY (matrix)
def gower(
    df: pd.DataFrame,
    num_cols: Iterable[str] | None = None,
    cat_cols: Iterable[str] | None = None,
    bin_cols: Iterable[str] | None = None,
) -> np.ndarray:
    """
    Compute Gower similarity matrix for a mixed-type DataFrame.

    Args:
        df: pandas DataFrame with rows as observations.
        num_cols: numeric columns (range-normalized).
        cat_cols: categorical columns (0/1 match).
        bin_cols: binary columns (0/1 match).

    Returns:
        n x n numpy array with Gower similarities in [0, 1].
    """
    df = pd.DataFrame(df).copy()
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # Guess column types if not provided
    if num_cols is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols is None:
        cat_cols = [c for c in df.columns if c not in num_cols]
    if bin_cols is None:
        bin_cols = []

    num_cols = list(num_cols)
    cat_cols = [c for c in cat_cols if c not in bin_cols]
    bin_cols = list(bin_cols)

    num_sim = np.zeros((n, n), dtype=float)
    p_num = 0
    for c in num_cols:
        col = df[c].astype(float)
        col = col.fillna(col.mean())
        arr = col.values.astype(float)
        rng = arr.max() - arr.min()
        if rng == 0:
            sij = np.ones((n, n), dtype=float)
        else:
            diff = np.abs(arr[:, None] - arr[None, :]) / rng
            sij = 1.0 - diff
        num_sim += sij
        p_num += 1

    cat_sim = np.zeros((n, n), dtype=float)
    p_cat = 0
    for c in cat_cols:
        col = df[c].astype("category")
        arr = col.cat.codes.values
        sij = (arr[:, None] == arr[None, :]).astype(float)
        cat_sim += sij
        p_cat += 1

    bin_sim = np.zeros((n, n), dtype=float)
    p_bin = 0
    for c in bin_cols:
        col = df[c].fillna(0).astype(int)
        arr = col.values
        sij = (arr[:, None] == arr[None, :]).astype(float)
        bin_sim += sij
        p_bin += 1

    total_parts = p_num + p_cat + p_bin
    if total_parts == 0:
        return np.eye(n, dtype=float)

    return (num_sim + cat_sim + bin_sim) / float(total_parts)
