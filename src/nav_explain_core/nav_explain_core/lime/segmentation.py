"""Costmap segmentation into interpretable obstacle features.

LIME for images needs interpretable features. The paper uses SLIC superpixels of
the local costmap, restricted to obstacles, with free space collapsed into a
single segment (label 0). If SLIC yields fewer than the desired number of
obstacle segments, the largest segment is split until the target is reached.

Reference: original ``lime_image.py::sm_only_obstacles`` and the SLIC
parametrisation quoted in the RAAD 2022 paper
(n_segments=10, compactness=100, max_iter=1000, min_size_factor=0.01,
max_size_factor=10).
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import slic

from ..costmap import OBSTACLE_LEVEL

# SLIC parameters from the paper. ``max_iter`` was renamed to ``max_num_iter`` in
# modern scikit-image, and ``channel_axis=None`` marks the costmap as a single
# grey channel (the original code used the same single-channel setting).
DEFAULT_SLIC = dict(
    n_segments=10,
    compactness=100.0,
    max_num_iter=1000,
    sigma=0,
    min_size_factor=0.01,
    max_size_factor=10,
    start_label=1,
    enforce_connectivity=True,
    channel_axis=None,
)


def obstacle_mask(costmap: np.ndarray) -> np.ndarray:
    """Boolean mask of obstacle cells (>= half lethal)."""
    return np.asarray(costmap, dtype=np.float64) >= OBSTACLE_LEVEL


def slic_obstacle_segments(
    costmap: np.ndarray,
    n_target: int = 8,
    slic_kwargs: dict | None = None,
) -> np.ndarray:
    """Segment a normalized costmap into obstacle features.

    SLIC is run *restricted to the obstacle pixels* (via its ``mask`` argument),
    so superpixels align to obstacles and free space is left as a single segment
    (label 0). This matches the paper, where each obstacle becomes one or more
    segments while free space is a single interpretable feature.

    Args:
        costmap: 2D array in the {0 free, 99 lethal} convention
            (see :func:`lime_path_explainer.core.costmap.normalize_costmap`).
        n_target: desired number of obstacle segments. If SLIC produces fewer,
            the largest obstacle segment is repeatedly split until reached.
        slic_kwargs: overrides for :data:`DEFAULT_SLIC`.

    Returns:
        ``segments``: integer array of the costmap's shape where ``0`` is free
        space and ``1..K`` are obstacle segments.
    """
    kwargs = dict(DEFAULT_SLIC)
    if slic_kwargs:
        kwargs.update(slic_kwargs)

    costmap = np.asarray(costmap, dtype=np.float64)
    mask = obstacle_mask(costmap)
    if not mask.any():
        return np.zeros(costmap.shape, dtype=np.int32)

    # Restrict SLIC to obstacle pixels; free space stays 0.
    raw = slic(costmap, mask=mask, **kwargs)

    # Relabel the masked superpixels contiguously as 1..K.
    segments = np.zeros(costmap.shape, dtype=np.int32)
    next_label = 1
    for slic_label in np.unique(raw):
        if slic_label == 0:
            continue  # outside the mask = free space
        segments[raw == slic_label] = next_label
        next_label += 1

    n_obstacles = next_label - 1
    if n_obstacles == 0:
        return segments
    if n_obstacles < n_target:
        segments = split_until_n(segments, n_target)
    elif n_obstacles > n_target:
        segments = merge_until_n(segments, n_target)
    return segments


def split_until_n(segments: np.ndarray, n_target: int) -> np.ndarray:
    """Split the largest obstacle segment along its long axis until ``n_target``.

    Splitting keeps the explanation's feature count stable across scenarios so
    that perturbation counts and the evaluation curves are comparable, as done
    in the original implementation.
    """
    segments = segments.copy()
    labels = [int(l) for l in np.unique(segments) if l != 0]
    while 0 < len(labels) < n_target:
        # Largest current obstacle segment.
        biggest = max(labels, key=lambda l: int(np.count_nonzero(segments == l)))
        rows, cols = np.where(segments == biggest)
        if rows.size < 2:
            break  # cannot split a single pixel

        # Split along the longer extent (the segment's "long axis").
        if (rows.max() - rows.min()) >= (cols.max() - cols.min()):
            axis_coord, threshold = rows, np.median(rows)
        else:
            axis_coord, threshold = cols, np.median(cols)

        new_label = max(labels) + 1
        far_side = axis_coord > threshold
        if not far_side.any() or far_side.all():
            break  # degenerate (all pixels on one side) -> stop to avoid a loop
        segments[rows[far_side], cols[far_side]] = new_label
        labels = [int(l) for l in np.unique(segments) if l != 0]
    return segments


def merge_until_n(segments: np.ndarray, n_target: int) -> np.ndarray:
    """Merge the smallest obstacle segments until exactly ``n_target`` remain.

    Each round, the smallest obstacle segment is absorbed into an adjacent
    obstacle segment (or, if isolated, the nearest one by centroid), keeping the
    interpretable feature count controlled so perturbation counts stay bounded
    (2**n_target) -- matching the paper, which always reduces to 8 segments.
    """
    from scipy import ndimage

    segments = segments.copy()
    labels = [int(l) for l in np.unique(segments) if l != 0]
    while len(labels) > n_target:
        smallest = min(labels, key=lambda l: int(np.count_nonzero(segments == l)))
        mask = segments == smallest
        # Find an adjacent obstacle segment by dilating the small segment.
        dilated = ndimage.binary_dilation(mask, iterations=2)
        neighbour_labels = set(int(v) for v in np.unique(segments[dilated]))
        neighbour_labels -= {0, smallest}
        if neighbour_labels:
            # absorb into the largest adjacent obstacle
            target = max(
                neighbour_labels, key=lambda l: int(np.count_nonzero(segments == l))
            )
        else:
            # isolated: merge into nearest segment by centroid distance
            cy, cx = ndimage.center_of_mass(mask)
            others = [l for l in labels if l != smallest]
            target = min(
                others,
                key=lambda l: _centroid_dist(segments == l, cy, cx),
            )
        segments[mask] = target
        labels = [int(l) for l in np.unique(segments) if l != 0]
    return _relabel_contiguous(segments)


def _centroid_dist(mask: np.ndarray, cy: float, cx: float) -> float:
    from scipy import ndimage

    oy, ox = ndimage.center_of_mass(mask)
    return float((oy - cy) ** 2 + (ox - cx) ** 2)


def _relabel_contiguous(segments: np.ndarray) -> np.ndarray:
    """Relabel obstacle segments to a contiguous 1..K range (free space stays 0)."""
    out = np.zeros_like(segments)
    for new_label, old in enumerate(
        [l for l in np.unique(segments) if l != 0], start=1
    ):
        out[segments == old] = new_label
    return out


def n_features(segments: np.ndarray) -> int:
    """Number of interpretable features = free space (0) + obstacle segments."""
    return int(np.unique(segments).shape[0])
