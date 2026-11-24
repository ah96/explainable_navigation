"""Perturbation generation, shared by every explanation method.

A perturbation is a binary vector over interpretable features (segments). A 1
keeps the segment as-is; a 0 turns it off, i.e. replaces those costmap cells with
free space ("the obstacle is imagined away"). Column 0 is the free-space segment
and is always kept on, matching the original ``lime_image.py::data_labels``.

Two sampling strategies:

* ``one_off``    -- one perturbation per obstacle segment (each turns exactly one
                    segment off). Cheap; ``n_features`` samples.
* ``full_2n``    -- the full 2**n enumeration of obstacle on/off combinations
                    (free space pinned on). Exhaustive; used as the reference for
                    the robustness evaluation.
"""

from __future__ import annotations

import itertools

import numpy as np

#: the two sampling strategies, as accepted by every ``mode`` argument
MODES = ("one_off", "full_2n")


def one_off_perturbations(n_features: int) -> np.ndarray:
    """``n_features`` rows; row 0 is all-on, each later row turns off one segment.

    Free space (column 0) is never turned off.
    """
    data = np.ones((n_features, n_features), dtype=np.int8)
    obstacles = np.arange(1, n_features)
    data[obstacles, obstacles] = 0  # row i turns off obstacle segment i
    return data


def full_2n_perturbations(n_features: int) -> np.ndarray:
    """All 2**(n_features-1) on/off combinations of the obstacle segments.

    The first column (free space) is pinned to 1. The first row is the all-on
    original instance. Every method assumes that ``data[0]`` is the unperturbed
    point.
    """
    n_obstacles = n_features - 1
    combos = list(itertools.product([1, 0], repeat=n_obstacles))  # (1,1,..) first
    data = np.array(combos, dtype=np.int8).reshape(len(combos), n_obstacles)
    free = np.ones((data.shape[0], 1), dtype=np.int8)
    return np.hstack([free, data])


def perturbations(n_features: int, mode: str) -> np.ndarray:
    """The perturbation matrix of the given ``mode`` (one of :data:`MODES`)."""
    if mode == "one_off":
        return one_off_perturbations(n_features)
    if mode == "full_2n":
        return full_2n_perturbations(n_features)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def apply_perturbation(
    costmap: np.ndarray,
    segments: np.ndarray,
    row: np.ndarray,
    fudged: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of ``costmap`` with the segments turned off in ``row`` hidden.

    Turned-off segments are replaced with the corresponding cells of ``fudged``
    (free space, i.e. zeros, by default). ``labels`` maps the columns of ``row``
    to segment labels. Pass it when calling in a loop, since it defaults to
    ``np.unique(segments)``, which is recomputed on every call.
    """
    perturbed = np.asarray(costmap, dtype=np.float64).copy()
    if labels is None:
        labels = np.unique(segments)
    mask = np.isin(segments, labels[np.asarray(row) == 0])
    perturbed[mask] = 0.0 if fudged is None else fudged[mask]
    return perturbed
