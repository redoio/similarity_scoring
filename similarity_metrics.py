from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = [
    "cosine", "euclidean", "manhattan",
    "jaccard_binary", "dice_binary", "hamming_binary",
    "gower",
]

# numeric (matrix-in, matrix-out)
def cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T

def euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = np.sum(X**2, axis=1, keepdims=True)
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
def gower(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], bin_cols: list[str]) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # total variable count for fallback when all missing
    p_total = (len(num_cols) + len(cat_cols) + len(bin_cols))
    if p_total == 0:
        return np.eye(n, dtype=float)

    S_sum = np.zeros((n, n), dtype=float)   # sum of per-variable similarities
    W_sum = np.zeros((n, n), dtype=float)   # sum of per-variable availability (weights)

    #  numeric 
    if num_cols:
        sub = df[num_cols].astype(float)
        # ranges ignoring NaNs; zero range treated as 1 to avoid divide-by-zero
        col_min = sub.min(skipna=True)
        col_max = sub.max(skipna=True)
        rng = (col_max - col_min).replace(0, 1.0)

        # normalize per column: (x - min) / range; NaNs preserved
        Xn = (sub - col_min) / rng
        Xv = Xn.values
        M = ~np.isnan(Xv)

        # pairwise absolute differences for available pairs only
        # sim = 1 - |xi - xj|
        for j in range(Xv.shape[1]):
            x = Xv[:, j][:, None]  # n x 1
            m = M[:, j][:, None]   # n x 1 mask
            # broadcast; invalid where either missing
            diff = np.abs(x - x.T)
            avail = (m & m.T).astype(float)
            sim = 1.0 - diff
            sim[avail == 0] = 0.0
            S_sum += sim
            W_sum += avail

    #  categorical 
    if cat_cols:
        # preserve NaN; don't count NaN==NaN as a match
        Xc = df[cat_cols].astype(object).values
        for j in range(Xc.shape[1]):
            col = Xc[:, j]
            m = ~pd.isna(col).to_numpy()
            a = col[:, None]
            b = col[None, :]
            avail = ((m[:, None]) & (m[None, :])).astype(float)
            sim = (a == b).astype(float)
            sim[avail == 0] = 0.0  # if either missing, no contribution
            S_sum += sim
            W_sum += avail

    #  binary 
    if bin_cols:
        # keep NaNs; exclude them from availability
        Xb = df[bin_cols].to_numpy()
        Mb = ~pd.isna(df[bin_cols]).to_numpy()

        # For each binary column: equal -> 1, different -> 0; exclude missing pairs
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

    # Avoid divide-by-zero: where W_sum==0, similarity=1 on diagonal else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.divide(S_sum, W_sum, out=np.zeros_like(S_sum), where=(W_sum > 0))

    # ensure diagonals are 1.0 when any variables exist
    np.fill_diagonal(S, 1.0)
    return S
