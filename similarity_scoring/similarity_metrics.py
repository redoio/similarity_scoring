# similarity_metrics.py
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Iterable, List

import numpy as np
import pandas as pd  # <-- NEW: for gower()

# --- CONFIG IMPORT (safe) ---
try:
    import config as CFG
    MIN_OVERLAP = int(getattr(CFG, "MIN_OVERLAP_FOR_SIMILARITY", 3))
except Exception:
    MIN_OVERLAP = 3  # fallback if config not available


__all__ = [
    # matrix-level (unchanged)
    "cosine",
    "euclidean",
    "manhattan",
    "jaccard_binary",
    "dice_binary",
    "hamming_binary",
    "gower",

    # new named-vector distance/sim pairs
    "cosine_distance_named",
    "cosine_similarity_named",
    "euclidean_distance_named",
    "euclidean_similarity_named",
    "tanimoto_distance_named",
    "tanimoto_similarity_named",

    # backwards compatibility
    "tanimoto_from_named",

    # jaccard on keys (updated)
    "jaccard_on_keys",
]


#  MATRIX-LEVEL METRICS 

def cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    sq = np.sum(X**2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2*(X @ X.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def manhattan(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)


#  BINARY SET METRICS 

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


#  HELPERS

def _finite(val):
    """Return float or None."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None


def _intersect(a: Dict, b: Dict, weights=None):
    """Return overlapping keys considering optional weights."""
    ks = set(a) & set(b)
    if weights:
        ks &= set(weights)
    return sorted(ks)


def _check_overlap_and_zero_case(a, b, keys):
    """
    Applies global MIN_OVERLAP and zero-vector rule.

    Returns:
        (allowed: bool, all_zero: bool)
    """
    # 1. Not enough overlap
    if len(keys) < MIN_OVERLAP:
        return False, False

    # 2. Are all matching values zero?
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


#  COSINE 

def cosine_distance_named(a: Dict[str, float], b: Dict[str, float], weights=None) -> float:
    """Cosine DISTANCE = 1 - cosine similarity."""
    sim = cosine_similarity_named(a, b, weights)
    return math.nan if isinstance(sim, float) and math.isnan(sim) else 1.0 - sim


def cosine_similarity_named(a: Dict[str, float], b: Dict[str, float], weights=None) -> float:
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

    if not used or den_x <= 0 or den_y <= 0:
        return math.nan

    return num / math.sqrt(den_x * den_y)


#  EUCLIDEAN 

def euclidean_distance_named(a: Dict[str, float], b: Dict[str, float], weights=None) -> float:
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
        s += w * (x - y)**2

    return math.sqrt(s) if used else math.nan


def euclidean_similarity_named(a: Dict[str, float], b: Dict[str, float], weights=None) -> float:
    """Euclidean SIMILARITY = 1 / (1 + distance)."""
    dist = euclidean_distance_named(a, b, weights)
    if isinstance(dist, float) and math.isnan(dist):
        return math.nan
    return 1.0 / (1.0 + dist)


#  TANIMOTO 

def tanimoto_distance_named(a: Dict[str, float], b: Dict[str, float], weights=None) -> float:
    sim = tanimoto_similarity_named(a, b, weights)
    return math.nan if isinstance(sim, float) and math.isnan(sim) else 1.0 - sim


def tanimoto_similarity_named(a: Dict[str, float], b: Dict[str, float], weights=None) -> float:
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
    if not used or den <= 0:
        return math.nan

    return num / den


#  Backwards compatibility alias (old name from earlier README)

def tanimoto_from_named(a, b, weights=None):
    """Alias kept for backward compatibility with older notebooks/READMEs."""
    return tanimoto_similarity_named(a, b, weights=weights)


#  JACCARD on KEYS 

def jaccard_on_keys(a: Dict[str, float], b: Dict[str, float], thresh=0.0) -> float:

    A = {k for k, v in a.items() if _finite(v) is not None and v > thresh}
    B = {k for k, v in b.items() if _finite(v) is not None and v > thresh}

    ks = A & B

    # MIN OVERLAP RULE
    if len(ks) < MIN_OVERLAP:
        return math.nan

    # ALL ZERO case → treat union as valid
    all_zero = all(a.get(k, 0) == 0 and b.get(k, 0) == 0 for k in ks)
    if all_zero:
        return 1.0

    inter = len(ks)
    union = len(A | B)

    return inter / union if union > 0 else math.nan


#  GOWER SIMILARITY (matrix)  

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
        num_cols: list of numeric columns (range-normalized).
        cat_cols: list of categorical columns (0/1 match).
        bin_cols: list of binary columns (0/1 match).

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
        # no usable columns → identity similarity
        return np.eye(n, dtype=float)

    sim = (num_sim + cat_sim + bin_sim) / float(total_parts)
    return sim
