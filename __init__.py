from .similarity import align_keys, cosine as cosine_1d, cosine_from_named
from .metrics import (
    cosine, euclidean, manhattan,
    jaccard_binary, dice_binary, hamming_binary, gower,
)

__all__ = [
    # 1-D / named-vector helpers
    "align_keys", "cosine_1d", "cosine_from_named",
    # matrix-level metrics
    "cosine", "euclidean", "manhattan",
    "jaccard_binary", "dice_binary", "hamming_binary",
    "gower",
]
