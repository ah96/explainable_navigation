"""Similarity kernel that weights perturbations by closeness to the instance.

LIME weights each perturbation by how similar it is to the original instance.
Following the original implementation (and the LIME defaults referenced in the
paper) we use cosine distance between the binary perturbation vectors and an
exponential kernel with width 0.25.
"""

from __future__ import annotations

import numpy as np
import sklearn.metrics

DEFAULT_KERNEL_WIDTH = 0.25


def cosine_distances(data: np.ndarray) -> np.ndarray:
    """Cosine distance of every perturbation to the first row (the instance)."""
    data = np.asarray(data, dtype=np.float64)
    return sklearn.metrics.pairwise_distances(
        data, data[0].reshape(1, -1), metric="cosine"
    ).ravel()


def exp_kernel(distances: np.ndarray, kernel_width: float = DEFAULT_KERNEL_WIDTH) -> np.ndarray:
    """``sqrt(exp(-(d^2) / kernel_width^2))`` -- the original LIME kernel."""
    distances = np.asarray(distances, dtype=np.float64)
    return np.sqrt(np.exp(-(distances**2) / kernel_width**2))


def weights_for(data: np.ndarray, kernel_width: float = DEFAULT_KERNEL_WIDTH) -> np.ndarray:
    """Convenience: cosine distance -> kernel weight for a perturbation matrix."""
    return exp_kernel(cosine_distances(data), kernel_width)
