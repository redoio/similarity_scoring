# vector_similarity.py
from __future__ import annotations

from typing import Dict

# canonical implementations live here
try:
    from .similarity_metrics import (
        cosine_similarity_named,
        euclidean_distance_named,
        euclidean_similarity_named,
        tanimoto_similarity_named,
        jaccard_similarity_named,
    )
except ImportError:
    from similarity_metrics import (  # type: ignore
        cosine_similarity_named,
        euclidean_distance_named,
        euclidean_similarity_named,
        tanimoto_similarity_named,
        jaccard_similarity_named,
    )

Weights = Dict[str, float]
Vec = Dict[str, float]

__all__ = [
    "cosine_similarity_named",
    "euclidean_distance_named",
    "euclidean_similarity_named",
    "tanimoto_similarity_named",
    "jaccard_similarity_named",
]
