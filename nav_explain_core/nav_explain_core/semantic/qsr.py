"""Qualitative spatial reasoning: where an object is, in words.

The robot is the *relatum* and each object is a *referent*. The single-cross
intrinsic calculus names one of four quadrants of the robot's heading frame:
``left/front``, ``right/front``, ``left/back``, ``right/back``.

A port of ``qsr_rt.py::getIntrinsicQsrValue`` of the original work.
"""

from __future__ import annotations

import math

from ..geometry import Pose2D, normalize_angle


def relative_angle(robot: Pose2D, obj_xy: tuple[float, float]) -> float:
    """Bearing of ``obj_xy`` relative to the robot's heading, in ``(-pi, pi]``."""
    bearing = math.atan2(obj_xy[1] - robot.y, obj_xy[0] - robot.x)
    return normalize_angle(bearing - robot.theta)


def single_cross_quadrant(angle: float) -> str:
    """Map a relative angle to one of the four intrinsic quadrants.

    Thresholds (from the original implementation):
        [0, pi/2)      -> left/front
        [pi/2, pi]     -> left/back
        [-pi/2, 0)     -> right/front
        [-pi, -pi/2)   -> right/back
    """
    a = normalize_angle(angle)
    if 0.0 <= a < math.pi / 2:
        return "left/front"
    if math.pi / 2 <= a <= math.pi:
        return "left/back"
    if -math.pi / 2 <= a < 0.0:
        return "right/front"
    return "right/back"


def qsr_for_object(robot: Pose2D, obj_xy: tuple[float, float]) -> str:
    """Quadrant of an object relative to the robot."""
    return single_cross_quadrant(relative_angle(robot, obj_xy))
