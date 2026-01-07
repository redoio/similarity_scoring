# similarity_scoring/vector_similarity.py
from __future__ import annotations

from typing import Dict, Optional

# Delegate ALL named-vector rule enforcement to similarity_metrics
try:
    from .similarity_metrics import (
        cosine_similarity_named,
        cosine_distance_named,
        euclidean_distance_named,
        euclidean_similarity_named,
        tanimoto_similarity_named,
        tanimoto_distance_named,
        jaccard_on_keys,
    )
except ImportError:
    from similarity_metrics import (  # type: ignore
        cosine_similarity_named,
        cosine_distance_named,
        euclidean_distance_named,
        euclidean_similarity_named,
        tanimoto_similarity_named,
        tanimoto_distance_named,
        jaccard_on_keys,
    )

Weights = Dict[str, float]
Vec = Dict[str, float]

__all__ = [
    # cosine
    "cosine_similarity_from_named",
    "cosine_distance_from_named",
    "cosine_similarity_from_named_weighted",
    "cosine_distance_from_named_weighted",
    # euclidean
    "euclidean_distance_from_named",
    "euclidean_similarity_from_named",
    "euclidean_distance_from_named_weighted",
    "euclidean_similarity_from_named_weighted",
    # tanimoto
    "tanimoto_similarity_from_named",
    "tanimoto_distance_from_named",
    "tanimoto_similarity_from_named_weighted",
    "tanimoto_distance_from_named_weighted",
    # jaccard on keys (weights intentionally ignored)
    "jaccard_on_keys_from_named",
    "jaccard_on_keys_from_named_weighted",
    # backwards-compat aliases (optional)
    "cosine_from_named",
    "cosine_from_named_weighted",
    "euclidean_from_named",
]


# Cosine (similarity & distance)
def cosine_similarity_from_named(a: Vec, b: Vec) -> float:
    """Cosine similarity on named vectors (rule enforcement lives in similarity_metrics)."""
    return float(cosine_similarity_named(a, b, weights=None))


def cosine_distance_from_named(a: Vec, b: Vec) -> float:
    """Cosine distance = 1 - cosine similarity (rule enforcement lives in similarity_metrics)."""
    return float(cosine_distance_named(a, b, weights=None))


def cosine_similarity_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Weighted cosine similarity (weights are per-feature; non-positive weights are ignored by core)."""
    if not weights:
        return cosine_similarity_from_named(a, b)
    return float(cosine_similarity_named(a, b, weights=weights))


def cosine_distance_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Weighted cosine distance (delegates to core for all edge-case behavior)."""
    if not weights:
        return cosine_distance_from_named(a, b)
    return float(cosine_distance_named(a, b, weights=weights))


# Euclidean (distance & similarity)
def euclidean_distance_from_named(a: Vec, b: Vec) -> float:
    """Euclidean distance on named vectors (rule enforcement lives in similarity_metrics)."""
    return float(euclidean_distance_named(a, b, weights=None))


def euclidean_similarity_from_named(a: Vec, b: Vec) -> float:
    """Euclidean similarity = 1 / (1 + distance) (rule enforcement lives in similarity_metrics)."""
    return float(euclidean_similarity_named(a, b, weights=None))


def euclidean_distance_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Weighted Euclidean distance (per-feature weights)."""
    if not weights:
        return euclidean_distance_from_named(a, b)
    return float(euclidean_distance_named(a, b, weights=weights))


def euclidean_similarity_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Weighted Euclidean similarity (computed from weighted distance by the core)."""
    if not weights:
        return euclidean_similarity_from_named(a, b)
    return float(euclidean_similarity_named(a, b, weights=weights))


# Tanimoto (similarity & distance)
def tanimoto_similarity_from_named(a: Vec, b: Vec) -> float:
    """Tanimoto similarity on named vectors (rule enforcement lives in similarity_metrics)."""
    return float(tanimoto_similarity_named(a, b, weights=None))


def tanimoto_distance_from_named(a: Vec, b: Vec) -> float:
    """Tanimoto distance = 1 - tanimoto similarity (rule enforcement lives in similarity_metrics)."""
    return float(tanimoto_distance_named(a, b, weights=None))


def tanimoto_similarity_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Weighted Tanimoto similarity (per-feature weights)."""
    if not weights:
        return tanimoto_similarity_from_named(a, b)
    return float(tanimoto_similarity_named(a, b, weights=weights))


def tanimoto_distance_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Weighted Tanimoto distance (delegates to core for all edge-case behavior)."""
    if not weights:
        return tanimoto_distance_from_named(a, b)
    return float(tanimoto_distance_named(a, b, weights=weights))


# Jaccard on keys (weights ignored)
def jaccard_on_keys_from_named(a: Vec, b: Vec, thresh: float = 0.0) -> float:
    """
    Jaccard on keys uses only which features are "active" (value > thresh).
    Weights are intentionally not used (set-based metric).
    """
    return float(jaccard_on_keys(a, b, thresh=thresh))


def jaccard_on_keys_from_named_weighted(
    a: Vec,
    b: Vec,
    weights: Optional[Weights] = None,
    thresh: float = 0.0,
) -> float:
    """
    API-symmetric wrapper that accepts weights but intentionally ignores them.
    This avoids "missing weighted variant" confusion while keeping semantics correct.
    """
    _ = weights  # explicitly ignored
    return jaccard_on_keys_from_named(a, b, thresh=thresh)


# Backwards-compat aliases (for optional use)
def cosine_from_named(a: Vec, b: Vec) -> float:
    """Back-compat alias for cosine similarity."""
    return cosine_similarity_from_named(a, b)


def cosine_from_named_weighted(a: Vec, b: Vec, weights: Optional[Weights]) -> float:
    """Back-compat alias for weighted cosine similarity."""
    return cosine_similarity_from_named_weighted(a, b, weights)


def euclidean_from_named(a: Vec, b: Vec) -> float:
    """
    Back-compat alias.
    Historically used as 'euclidean_from_named' meaning distance.
    """
    return euclidean_distance_from_named(a, b)
