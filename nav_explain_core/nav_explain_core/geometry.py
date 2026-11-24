"""Poses, angles and planar transforms shared across the core."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    """A planar pose in a world/map frame (metres, radians)."""

    x: float
    y: float
    theta: float = 0.0


def yaw_from_quaternion(z: float, w: float) -> float:
    """Yaw (rad) from the z/w components of a quaternion with x=y=0."""
    return 2.0 * math.atan2(z, w)


def normalize_angle(angle: float) -> float:
    """Wrap an angle to the half-open interval ``(-pi, pi]``."""
    a = math.fmod(angle, 2.0 * math.pi)
    if a <= -math.pi:
        a += 2.0 * math.pi
    elif a > math.pi:
        a -= 2.0 * math.pi
    return a


def transform_points(points: np.ndarray, tx: float, ty: float, yaw: float) -> np.ndarray:
    """Express ``(N, 2)`` points of a source frame in a target frame.

    ``(tx, ty, yaw)`` is the pose of the source frame in the target frame, which
    is what a TF lookup from source to target returns.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + (tx, ty)


def transform_pose(pose: Pose2D, tx: float, ty: float, yaw: float) -> Pose2D:
    """Express a pose of a source frame in a target frame, see :func:`transform_points`."""
    (x, y), = transform_points([[pose.x, pose.y]], tx, ty, yaw)
    return Pose2D(float(x), float(y), normalize_angle(pose.theta + yaw))
