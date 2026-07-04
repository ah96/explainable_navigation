"""Perturbation generation for LIME.

A perturbation is a binary vector over interpretable features (segments). A 1
keeps the segment as-is; a 0 turns it off, i.e. replaces those costmap cells with
free space ("the obstacle is imagined away"). Column 0 is the free-space segment
and is always kept on, matching the original ``lime_image.py::data_labels``.

Two sampling strategies from the paper:

* ``one_off``    -- one perturbation per obstacle segment (each turns exactly one
                    segment off). Cheap; ``n_features`` samples.
* ``full_2n``    -- the full 2**n enumeration of obstacle on/off combinations
                    (free space pinned on). Exhaustive; used as the reference for
                    the robustness evaluation.
"""

from __future__ import annotations

import itertools

import numpy as np


def one_off_perturbations(n_features: int) -> np.ndarray:
    """``n_features`` rows; row 0 is all-on, each later row turns off one segment.

    Free space (column 0) is never turned off.
    """
    data = np.ones((n_features, n_features), dtype=np.int8)
    for i in range(1, n_features):
        data[i, i] = 0  # turn off obstacle segment i in row i
    return data


def full_2n_perturbations(n_features: int, free_space_on: bool = True) -> np.ndarray:
    """All 2**(n_features-1) on/off combinations of the obstacle segments.

    The first column (free space) is pinned to 1 when ``free_space_on``. The
    first row is the all-on original instance (required by LIME, which assumes
    ``data[0]`` is the unperturbed point).
    """
    n_obstacles = n_features - 1
    combos = list(itertools.product([1, 0], repeat=n_obstacles))  # (1,1,..) first
    data = np.array(combos, dtype=np.int8)
    free = np.ones((data.shape[0], 1), dtype=np.int8)
    if not free_space_on:
        free[:] = 1  # free space is conceptually always present in the costmap
    return np.hstack([free, data])


def apply_perturbation(
    costmap: np.ndarray,
    segments: np.ndarray,
    row: np.ndarray,
    fudged: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of ``costmap`` with the segments turned off in ``row`` hidden.

    Turned-off segments are replaced with the corresponding cells of ``fudged``
    (free space, i.e. zeros, by default).
    """
    if fudged is None:
        fudged = np.zeros_like(costmap)
    perturbed = np.asarray(costmap, dtype=np.float64).copy()
    labels = np.unique(segments)
    off = np.where(np.asarray(row) == 0)[0]
    mask = np.zeros(segments.shape, dtype=bool)
    for j in off:
        mask |= segments == labels[j]
    perturbed[mask] = fudged[mask]
    return perturbed
