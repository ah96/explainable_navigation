"""The deviation of a local plan from the global plan, and the plan ahead of the robot.

``deviation`` is the one number every explanation method explains. It is a
vectorised port of the function of the same name in ``traj_dist.py`` of the
original ROS 1 code.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def deviation(local_plan: np.ndarray, global_plan: np.ndarray) -> float:
    """Sum over local-plan points of the min Euclidean distance to the global plan.

    The distance is taken to the points of the global plan, not to the segments
    between them, so a local plan that lies on the global plan has a small
    deviation, not exactly zero. Returns 0.0 if either plan is empty.
    """
    local_plan = np.asarray(local_plan, dtype=np.float64).reshape(-1, 2)
    global_plan = np.asarray(global_plan, dtype=np.float64).reshape(-1, 2)
    if local_plan.size == 0 or global_plan.size == 0:
        return 0.0
    return float(cdist(local_plan, global_plan).min(axis=1).sum())


def remaining_plan(global_plan: np.ndarray, x: float, y: float) -> np.ndarray:
    """The global plan from its point nearest to ``(x, y)`` onwards.

    A navigation stack publishes the plan from where the robot started. The part
    behind the robot is no longer a choice to explain, and the local planner
    would have to bend its path back to it.
    """
    global_plan = np.asarray(global_plan, dtype=np.float64).reshape(-1, 2)
    if len(global_plan) == 0:
        return global_plan
    nearest = int(np.argmin(np.hypot(global_plan[:, 0] - x, global_plan[:, 1] - y)))
    return global_plan[nearest:]
