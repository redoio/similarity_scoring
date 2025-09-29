from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = [
    "cosine", "euclidean", "manhattan",
    "jaccard_binary", "dice_binary", "hamming_binary",
    "gower",
]

# --- numeric (matrix-in, matrix-out) ---

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

# --- binary (0/1) ---

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

# --- mixed (Gower similarity) ---

def gower(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], bin_cols: list[str]) -> np.ndarray:
    n = len(df)
    p = len(num_cols) + len(cat_cols) + len(bin_cols)
    if p == 0:
        return np.eye(n, dtype=float)

    S = np.zeros((n, n), dtype=float)

    if num_cols:
        sub = df[num_cols].astype(float)
        rng = sub.max() - sub.min()
        rng = rng.replace(0, 1.0)
        Xn = (sub - sub.min()) / rng
        num_sim = 1.0 - np.abs(Xn.values[:, None, :] - Xn.values[None, :, :])
        S += np.nan_to_num(num_sim.sum(axis=2))

    if cat_cols:
        Xc = df[cat_cols].astype("category").apply(lambda s: s.cat.codes)
        cat_sim = (Xc.values[:, None, :] == Xc.values[None, :, :]).sum(axis=2)
        S += cat_sim

    if bin_cols:
        Xb = df[bin_cols].fillna(0).astype(int).values
        # XNOR per bit (equal -> 1, different -> 0)
        bin_sim = 1 - (Xb[:, None, :] ^ Xb[None, :, :])
        S += bin_sim.sum(axis=2)

    return S / float(p)
