"""When a new global plan counts as a deviation from the previous one.

Two plans to the same goal are compared at their middle points. If these lie
further apart than a threshold, the robot has changed its route and not only
smoothed it.
"""

from __future__ import annotations

import numpy as np

#: distance between the middle points of two plans from which on they differ, in metres
DEVIATION_THRESHOLD_M = 1.0


def plan_deviation(previous: np.ndarray, current: np.ndarray) -> float:
    """Distance in metres between the middle points of two ``(N, 2)`` plans."""
    previous = np.asarray(previous, dtype=float).reshape(-1, 2)
    current = np.asarray(current, dtype=float).reshape(-1, 2)
    if len(previous) == 0 or len(current) == 0:
        return 0.0
    mid_previous = previous[int(0.5 * (len(previous) - 1))]
    mid_current = current[int(0.5 * (len(current) - 1))]
    return float(np.linalg.norm(mid_current - mid_previous))
