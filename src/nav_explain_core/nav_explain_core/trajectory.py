"""Trajectory-distance metrics.

A cleaned, vectorised port of the original ``traj_dist.py`` used to compare a
robot's local plan against its global plan. ``deviation`` is the LIME/Anchors
target used throughout the explainers; the remaining metrics (SSPD, Hausdorff,
discrete Fréchet, DTW, LCSS, EDR, ERP, OWD) are provided as a small reusable
library for trajectory comparison.

Changes from the ROS1 original:
* ``deviation`` / point-to-trajectory are vectorised with
  ``scipy.spatial.distance.cdist`` instead of nested Python loops;
* the ``e_hausdorff`` bug (a bare ``return`` that returned ``None``) is fixed;
* debug ``print`` statements are removed; functions get docstrings + types.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Core target used by the explainers
# --------------------------------------------------------------------------- #
def deviation(local_plan: Array, global_plan: Array) -> float:
    """Sum over local-plan points of the min Euclidean distance to the global plan.

    This is the regression target for the LIME/Anchors explainers: how far the
    (perturbed) local plan strays from the global plan. Vectorised via ``cdist``.
    Returns 0.0 if either plan is empty.
    """
    local_plan = np.asarray(local_plan, dtype=np.float64).reshape(-1, 2)
    global_plan = np.asarray(global_plan, dtype=np.float64).reshape(-1, 2)
    if local_plan.size == 0 or global_plan.size == 0:
        return 0.0
    return float(cdist(local_plan, global_plan).min(axis=1).sum())


def point_to_trajectory(point: Array, trajectory: Array) -> float:
    """Minimum Euclidean distance from ``point`` (2,) to a polyline's vertices."""
    trajectory = np.asarray(trajectory, dtype=np.float64).reshape(-1, 2)
    if trajectory.size == 0:
        return 0.0
    return float(np.linalg.norm(trajectory - np.asarray(point, float), axis=1).min())


# --------------------------------------------------------------------------- #
# General trajectory-distance library
# --------------------------------------------------------------------------- #
def _pairwise(t0: Array, t1: Array) -> Array:
    return cdist(np.asarray(t0, float).reshape(-1, 2), np.asarray(t1, float).reshape(-1, 2))


def spd(t1: Array, t2: Array) -> float:
    """Mean of the point-to-trajectory distances of t1's points from t2."""
    m = _pairwise(t1, t2)
    return float(m.min(axis=1).mean())


def sspd(t1: Array, t2: Array) -> float:
    """Symmetric segment-path distance: mean of spd(t1,t2) and spd(t2,t1)."""
    return 0.5 * (spd(t1, t2) + spd(t2, t1))


def directed_hausdorff(t1: Array, t2: Array) -> float:
    m = _pairwise(t1, t2)
    return float(m.min(axis=1).max())


def hausdorff(t1: Array, t2: Array) -> float:
    """Symmetric Hausdorff distance (bug-fixed vs. the original)."""
    return max(directed_hausdorff(t1, t2), directed_hausdorff(t2, t1))


def discrete_frechet(t0: Array, t1: Array) -> float:
    """Discrete Fréchet distance."""
    m = _pairwise(t0, t1)
    n0, n1 = m.shape
    ca = np.full((n0, n1), -1.0)
    ca[0, 0] = m[0, 0]
    for i in range(1, n0):
        ca[i, 0] = max(ca[i - 1, 0], m[i, 0])
    for j in range(1, n1):
        ca[0, j] = max(ca[0, j - 1], m[0, j])
    for i in range(1, n0):
        for j in range(1, n1):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), m[i, j])
    return float(ca[-1, -1])


def dtw(t0: Array, t1: Array) -> float:
    """Dynamic Time Warping distance."""
    m = _pairwise(t0, t1)
    n0, n1 = m.shape
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            C[i, j] = m[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return float(C[n0, n1])


def lcss(t0: Array, t1: Array, eps: float) -> float:
    """Longest-Common-Subsequence distance (1 - LCSS/min length)."""
    m = _pairwise(t0, t1)
    n0, n1 = m.shape
    C = np.zeros((n0 + 1, n1 + 1))
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if m[i - 1, j - 1] < eps:
                C[i, j] = C[i - 1, j - 1] + 1
            else:
                C[i, j] = max(C[i, j - 1], C[i - 1, j])
    return float(1.0 - C[n0, n1] / min(n0, n1))


def edr(t0: Array, t1: Array, eps: float) -> float:
    """Edit Distance on Real sequence."""
    m = _pairwise(t0, t1)
    n0, n1 = m.shape
    C = np.zeros((n0 + 1, n1 + 1))
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            subcost = 0 if m[i - 1, j - 1] < eps else 1
            C[i, j] = min(C[i, j - 1] + 1, C[i - 1, j] + 1, C[i - 1, j - 1] + subcost)
    return float(C[n0, n1] / max(n0, n1))


def erp(t0: Array, t1: Array, g: Array) -> float:
    """Edit distance with Real Penalty (reference point ``g``)."""
    t0 = np.asarray(t0, float).reshape(-1, 2)
    t1 = np.asarray(t1, float).reshape(-1, 2)
    g = np.asarray(g, float).reshape(2)
    gt0 = np.linalg.norm(t0 - g, axis=1)
    gt1 = np.linalg.norm(t1 - g, axis=1)
    m = _pairwise(t0, t1)
    n0, n1 = len(t0), len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = np.cumsum(gt0)
    C[0, 1:] = np.cumsum(gt1)
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            C[i, j] = min(
                C[i - 1, j] + gt0[i - 1],
                C[i, j - 1] + gt1[j - 1],
                C[i - 1, j - 1] + m[i - 1, j - 1],
            )
    return float(C[n0, n1])


def owd(t1: Array, t2: Array) -> float:
    """One-Way Distance of t2 from t1 (mean nearest-point distance)."""
    return float(_pairwise(t1, t2).min(axis=1).mean())


METRICS = {
    "deviation": deviation,
    "sspd": sspd,
    "hausdorff": hausdorff,
    "discrete_frechet": discrete_frechet,
    "dtw": dtw,
    "owd": owd,
}
