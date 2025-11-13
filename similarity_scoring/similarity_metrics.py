from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    # matrix-level similarities / distances
    "cosine",
    "euclidean",
    "manhattan",
    "jaccard_binary",
    "dice_binary",
    "hamming_binary",
    "gower",
    # named-vector helpers used in similarity_scoring demos
    "euclidean_distance_named",
    "tanimoto_from_named",
    "jaccard_on_keys",
]


# Matrix-based metrics (keep API stable for population_analytics)

# numeric (matrix-in, matrix-out)
def cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def manhattan(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)


# binary (0/1)
def jaccard_binary(B: np.ndarray) -> np.ndarray:
    B = (np.asarray(B) > 0).astype(np.uint8)
    inter = (B[:, None, :] & B[None, :, :]).sum(axis=2)
    union = (B[:, None, :] | B[None, :, :]).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        J = inter / np.maximum(union, 1)
    return J


def dice_binary(B: np.ndarray) -> np.ndarray:
    B = (np.asarray(B) > 0).astype(np.uint8)
    inter = (B[:, None, :] & B[None, :, :]).sum(axis=2)
    sizes = B.sum(axis=1, keepdims=True)
    denom = np.maximum(sizes + sizes.T, 1)
    return (2.0 * inter) / denom


def hamming_binary(B: np.ndarray) -> np.ndarray:
    B = (np.asarray(B) > 0).astype(np.uint8)
    p = B.shape[1] if B.ndim == 2 and B.shape[1] > 0 else 1
    neq = (B[:, None, :] != B[None, :, :]).sum(axis=2)
    return neq / float(p)


# mixed (Gower similarity) — UPDATED to exclude missing per pair
def gower(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    bin_cols: list[str],
) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # total variable count for fallback when all missing
    p_total = len(num_cols) + len(cat_cols) + len(bin_cols)
    if p_total == 0:
        return np.eye(n, dtype=float)

    S_sum = np.zeros((n, n), dtype=float)  # sum of per-variable similarities
    W_sum = np.zeros((n, n), dtype=float)  # sum of per-variable availability

    #  numeric 
    if num_cols:
        sub = df[num_cols].astype(float)
        col_min = sub.min(skipna=True)
        col_max = sub.max(skipna=True)
        rng = (col_max - col_min).replace(0, 1.0)

        Xn = (sub - col_min) / rng
        Xv = Xn.values
        M = ~np.isnan(Xv)

        for j in range(Xv.shape[1]):
            x = Xv[:, j][:, None]
            m = M[:, j][:, None]
            diff = np.abs(x - x.T)
            avail = (m & m.T).astype(float)
            sim = 1.0 - diff
            sim[avail == 0] = 0.0
            S_sum += sim
            W_sum += avail

    #  categorical 
    if cat_cols:
        Xc = df[cat_cols].astype(object).values
        for j in range(Xc.shape[1]):
            col = Xc[:, j]
            m = ~pd.isna(col).to_numpy()
            a = col[:, None]
            b = col[None, :]
            avail = ((m[:, None]) & (m[None, :])).astype(float)
            sim = (a == b).astype(float)
            sim[avail == 0] = 0.0
            S_sum += sim
            W_sum += avail

    #  binary 
    if bin_cols:
        Xb = df[bin_cols].to_numpy()
        Mb = ~pd.isna(df[bin_cols]).to_numpy()

        for j in range(Xb.shape[1]):
            col = Xb[:, j]
            m = Mb[:, j]
            a = col[:, None]
            b = col[None, :]
            avail = ((m[:, None]) & (m[None, :])).astype(float)
            sim = (a == b).astype(float)
            sim[avail == 0] = 0.0
            S_sum += sim
            W_sum += avail

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.divide(S_sum, W_sum, out=np.zeros_like(S_sum), where=(W_sum > 0))

    np.fill_diagonal(S, 1.0)
    return S



# Named-vector helpers (used by sentencing similarity demos)

def _finite(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


def _intersect_keys(a: Dict[str, float], b: Dict[str, float], *more):
    ks = set(a) & set(b)
    for d in more:
        ks &= set(d)
    return sorted(ks)


def euclidean_distance_named(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    ks = _intersect_keys(a, b, *([weights] if weights else []))
    if not ks:
        return 0.0
    s = 0.0
    used = False
    for k in ks:
        x, y = _finite(a.get(k)), _finite(b.get(k))
        if x is None or y is None:
            continue
        w = float(weights.get(k, 1.0)) if weights else 1.0
        if w <= 0:
            continue
        s += w * (x - y) ** 2
        used = True
    return 0.0 if not used else math.sqrt(s)


def tanimoto_from_named(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Continuous Tanimoto / Jaccard for real-valued vectors."""
    ks = _intersect_keys(a, b, *([weights] if weights else []))
    if not ks:
        return 0.0
    num = den_x = den_y = 0.0
    used = False
    for k in ks:
        x, y = _finite(a.get(k)), _finite(b.get(k))
        if x is None or y is None:
            continue
        w = float(weights.get(k, 1.0)) if weights else 1.0
        if w <= 0:
            continue
        num += w * (x * y)
        den_x += w * (x * x)
        den_y += w * (y * y)
        used = True
    den = den_x + den_y - num
    return 0.0 if (not used or den <= 0) else (num / den)


def jaccard_on_keys(
    a: Dict[str, float],
    b: Dict[str, float],
    thresh: float = 0.0,
) -> float:
    A = {k for k, v in a.items() if _finite(v) is not None and float(v) > thresh}
    B = {k for k, v in b.items() if _finite(v) is not None and float(v) > thresh}
    if not A and not B:
        return 1.0  # both empty → identical
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union
