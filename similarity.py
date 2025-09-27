# similarity_scoring/similarity.py
from __future__ import annotations
from typing import Dict, Sequence
import math

__all__ = ["align_keys", "cosine", "cosine_from_named"]

def align_keys(a: Dict[str, float], b: Dict[str, float]) -> list[str]:
    """Sorted intersection of feature names (only shared features are comparable)."""
    return sorted(set(a).intersection(b))

def _to_num(x) -> float:
    """Best-effort cast to float; NaN/None/bad values -> 0.0."""
    try:
        val = float(x)
        if math.isnan(val):
            return 0.0
        return val
    except Exception:
        return 0.0

def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity; returns 0.0 if either vector has zero norm."""
    num = 0.0
    sum_u2 = 0.0
    sum_v2 = 0.0
    for ux, vx in zip(u, v):
        ux = _to_num(ux)
        vx = _to_num(vx)
        num += ux * vx
        sum_u2 += ux * ux
        sum_v2 += vx * vx
    nu = math.sqrt(sum_u2)
    nv = math.sqrt(sum_v2)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return num / (nu * nv)

def cosine_from_named(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine over intersecting keys only (skip-if-missing across persons)."""
    keys = align_keys(a, b)
    if not keys:
        return 0.0
    ua = [_to_num(a.get(k)) for k in keys]
    vb = [_to_num(b.get(k)) for k in keys]
    return cosine(ua, vb)
