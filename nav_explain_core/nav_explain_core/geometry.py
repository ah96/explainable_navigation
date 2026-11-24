"""Small geometric value types shared across the core.

Kept deliberately dependency-light (only numpy) so the core stays ROS-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    """A planar pose in a world/map frame (metres, radians)."""

    x: float
    y: float
    theta: float = 0.0


@dataclass(frozen=True)
class VelLimits:
    """Robot velocity/acceleration limits.

    Defaults mirror the TEB configuration of the original work, which is kept
    in ``nav_explain_sim/config/nav2_params.yaml``.
    """

    max_vel_x: float = 0.5
    max_vel_theta: float = 0.6
    acc_lim_x: float = 0.2


def yaw_from_quaternion(z: float, w: float) -> float:
    """Yaw (rad) from the z/w components of a quaternion with x=y=0.

    The original ROS1 code stored only ``orientation.z`` and ``orientation.w``
    for the (planar) costmap origin and robot pose, so this 2-component form is
    all that is needed.
    """
    return 2.0 * math.atan2(z, w)


def quaternion_from_yaw(theta: float) -> tuple[float, float]:
    """(z, w) quaternion components for a planar yaw."""
    return math.sin(theta / 2.0), math.cos(theta / 2.0)


def normalize_angle(angle: float) -> float:
    """Wrap an angle to the half-open interval ``(-pi, pi]``."""
    a = math.fmod(angle, 2.0 * math.pi)
    if a <= -math.pi:
        a += 2.0 * math.pi
    elif a > math.pi:
        a -= 2.0 * math.pi
    return a
