# similarity_scoring/similarity_metrics.py
from __future__ import annotations

"""
similarity_metrics.py — canonical similarity/distance implementations.

Single source of truth for:
  - MIN_OVERLAP_FOR_SIMILARITY gating
  - NaN behavior when overlap is insufficient/unusable
  - "all-zero overlap" rule (similarity = 1.0 for cosine/tanimoto; jaccard = 1.0)
  - consistent weighted handling for magnitude-based metrics

These functions are intended to be imported directly by scripts/CLI modules.
"""

import math
from typing import Dict, Optional, Iterable, Any

import numpy as np
import pandas as pd  # used for gower()

# CONFIG IMPORT (package-safe)
try:
    from . import config as CFG  # when imported as a package
    MIN_OVERLAP = int(getattr(CFG, "MIN_OVERLAP_FOR_SIMILARITY", 3))
except Exception:
    try:
        import config as CFG  # when run as a loose script
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
    # named-vector (canonical)
    "cosine_similarity_named",
    "euclidean_distance_named",
    "euclidean_similarity_named",
    "tanimoto_similarity_named",
    # key-based set similarity (named vectors)
    "jaccard_similarity_named",
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


def _overlapping_keys(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> list[str]:
    """
    Overlapping keys in a and b, optionally restricted to keys present in weights.

    Note: key intersection only; numeric validity filtering happens later.
    """
    ks = set(a) & set(b)
    if weights:
        ks &= set(weights)
    return sorted(ks)


def _allowed_and_all_zero(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: list[str],
) -> tuple[bool, bool]:
    """
    Applies global MIN_OVERLAP and "all-zero overlap" rule.

    Returns:
        (allowed: bool, all_zero: bool)
    """
    if len(keys) < MIN_OVERLAP:
        return False, False

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

def cosine_similarity_named(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Cosine similarity on name-keyed vectors.

    If overlap < MIN_OVERLAP_FOR_SIMILARITY, returns NaN.
    If all finite overlapping values are 0, returns 1.0 (degenerate-but-identical).
    """
    ks = _overlapping_keys(a, b, weights)
    allowed, all_zero = _allowed_and_all_zero(a, b, ks)
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

def euclidean_distance_named(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    ks = _overlapping_keys(a, b, weights)
    allowed, _ = _allowed_and_all_zero(a, b, ks)
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


def euclidean_similarity_named(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Euclidean similarity = 1 / (1 + distance)."""
    dist = euclidean_distance_named(a, b, weights=weights)
    if isinstance(dist, float) and math.isnan(dist):
        return math.nan
    return 1.0 / (1.0 + dist)


# TANIMOTO (named vectors)

def tanimoto_similarity_named(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Tanimoto similarity (continuous Jaccard) for real-valued vectors.

    If overlap < MIN_OVERLAP_FOR_SIMILARITY, returns NaN.
    If all finite overlapping values are 0, returns 1.0.
    """
    ks = _overlapping_keys(a, b, weights)
    allowed, all_zero = _allowed_and_all_zero(a, b, ks)
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


# JACCARD SIMILARITY (named vectors; active keys)

def jaccard_similarity_named(
    a: Dict[str, float],
    b: Dict[str, float],
    thresh: float = 0.0,
) -> float:
    """
    Jaccard similarity on ACTIVE keys, where a key is active if value > thresh.

    MIN_OVERLAP is applied to the INTERSECTION of active keys.
    If intersection < MIN_OVERLAP, returns NaN.
    If intersection keys are all zero (degenerate), returns 1.0.
    """
    A = {k for k, v in a.items() if _finite(v) is not None and float(v) > thresh}
    B = {k for k, v in b.items() if _finite(v) is not None and float(v) > thresh}

    inter_keys = A & B
    if len(inter_keys) < MIN_OVERLAP:
        return math.nan

    all_zero = all(
        float(a.get(k, 0.0) or 0.0) == 0.0 and float(b.get(k, 0.0) or 0.0) == 0.0
        for k in inter_keys
    )
    if all_zero:
        return 1.0

    union = len(A | B)
    return (len(inter_keys) / union) if union > 0 else math.nan


# GOWER SIMILARITY (matrix)

def gower(
    df: pd.DataFrame,
    num_cols: Iterable[str] | None = None,
    cat_cols: Iterable[str] | None = None,
    bin_cols: Iterable[str] | None = None,
) -> np.ndarray:
    """Compute Gower similarity matrix for a mixed-type DataFrame."""
    df = pd.DataFrame(df).copy()
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

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
        col = df[c].astype(float).fillna(df[c].astype(float).mean())
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
