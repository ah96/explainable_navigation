"""Qualitative Spatial Reasoning calculi.

The robot is the *relatum*; each object is a *referent*. Two calculi are
provided:

* **single-cross** intrinsic calculus (4 quadrants of the robot's heading frame):
  ``left/front``, ``right/front``, ``left/back``, ``right/back``;
* **ternary** (distance + direction) calculus: a ``close``/``distant`` distance
  term (relative to a reference radius) combined with a ``front``/``left``/
  ``back``/``right`` direction term, e.g. ``close-front``.

Ports of ``qsr_rt.py::getIntrinsicQsrValue`` (single-cross) and
``qsr_class_copy.py::getTernaryQsrValue`` (ternary).
"""

from __future__ import annotations

import math

from ..geometry import Pose2D, normalize_angle

QUADRANTS = ("left/front", "right/front", "left/back", "right/back")
DIRECTIONS = ("front", "left", "back", "right")


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
    """Convenience: quadrant of an object relative to the robot."""
    return single_cross_quadrant(relative_angle(robot, obj_xy))


def ternary_direction(angle: float) -> str:
    """Direction term of the ternary calculus: front/left/back/right.

    Thresholds (from ``getTernaryQsrValue``):
        [-pi/4, pi/4]         -> front
        (pi/4, 3pi/4)         -> left
        |angle| >= 3pi/4      -> back
        (-3pi/4, -pi/4)       -> right
    """
    a = normalize_angle(angle)
    if -math.pi / 4 <= a <= math.pi / 4:
        return "front"
    if math.pi / 4 < a < 3 * math.pi / 4:
        return "left"
    if a >= 3 * math.pi / 4 or a <= -3 * math.pi / 4:
        return "back"
    return "right"


def ternary_qsr(robot: Pose2D, obj_xy: tuple[float, float], reference_radius: float) -> str:
    """Distance+direction relation, e.g. ``close-front`` or ``distant-left``.

    ``close`` iff the object is within ``reference_radius`` of the robot.
    """
    dist = math.hypot(obj_xy[0] - robot.x, obj_xy[1] - robot.y)
    distance_term = "close" if dist <= reference_radius else "distant"
    return f"{distance_term}-{ternary_direction(relative_angle(robot, obj_xy))}"
