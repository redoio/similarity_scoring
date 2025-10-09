# similarity_scoring/vector_similarity.py
from __future__ import annotations
from typing import Dict, Sequence, List, Optional
import math

__all__ = [
    "align_keys",
    "cosine",
    "cosine_from_named",
    "cosine_from_named_weighted",  # new (optional)
]

def align_keys(a: Dict[str, float], b: Dict[str, float]) -> list[str]:
    """Sorted intersection of feature names (only shared features are comparable)."""
    return sorted(set(a).intersection(b))

def _to_float_or_none(x) -> Optional[float]:
    """Best-effort cast to float; return None for NaN/None/bad values."""
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None

def _finite_pairs(u: Sequence, v: Sequence) -> List[tuple[float, float]]:
    """Zip u,v and keep only pairs where both are finite floats."""
    pairs: List[tuple[float, float]] = []
    for ux, vx in zip(u, v):
        fu = _to_float_or_none(ux)
        fv = _to_float_or_none(vx)
        if fu is not None and fv is not None:
            pairs.append((fu, fv))
    return pairs

def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity; returns 0.0 if no finite overlap or either norm is zero."""
    pairs = _finite_pairs(u, v)
    if not pairs:
        return 0.0
    num = sum(ux * vx for ux, vx in pairs)
    sum_u2 = sum(ux * ux for ux, _ in pairs)
    sum_v2 = sum(vx * vx for _, vx in pairs)
    nu = math.sqrt(sum_u2)
    nv = math.sqrt(sum_v2)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return num / (nu * nv)

def cosine_from_named(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine over intersecting keys only; skips missing/NaN values (no zero imputation)."""
    keys = align_keys(a, b)
    if not keys:
        return 0.0
    ua = [a.get(k) for k in keys]
    vb = [b.get(k) for k in keys]
    return cosine(ua, vb)

#  Optional: weighted cosine with explicit weights (no silent defaults) 
def cosine_from_named_weighted(
    a: Dict[str, float],
    b: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Weighted cosine across the intersection of (a,b,weights) only.
    Features without an explicit weight are excluded (no implicit 1.0).
    """
    if not weights:
        return cosine_from_named(a, b)

    keys = sorted(set(a.keys()) & set(b.keys()) & set(weights.keys()))
    if not keys:
        return 0.0

    # Build weighted vectors by multiplying sqrt(weights) into each side
    # so standard cosine(u',v') equals weighted cosine.
    w_sqrt = []
    ua = []
    vb = []
    for k in keys:
        x = _to_float_or_none(a.get(k))
        y = _to_float_or_none(b.get(k))
        if x is None or y is None:
            continue
        w = float(weights[k])
        if w <= 0.0 or math.isnan(w) or math.isinf(w):
            continue
        w_sqrt_val = math.sqrt(w)
        w_sqrt.append(w_sqrt_val)
        ua.append(x * w_sqrt_val)
        vb.append(y * w_sqrt_val)

    if not ua:
        return 0.0
    return cosine(ua, vb)
