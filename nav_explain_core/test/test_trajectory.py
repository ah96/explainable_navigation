"""The deviation, the plan ahead of the robot and the planar transform."""

import math

import numpy as np

from nav_explain_core import trajectory as tj
from nav_explain_core.geometry import Pose2D, transform_points, transform_pose


def test_deviation_identical_zero():
    plan = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    assert tj.deviation(plan, plan) == 0.0


def test_deviation_offset_known():
    glob = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    local = np.array([[0.5, 0.0], [0.5, 1.0]])
    assert abs(tj.deviation(local, glob) - 1.0) < 1e-9


def test_deviation_empty():
    glob = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert tj.deviation(np.empty((0, 2)), glob) == 0.0


def test_remaining_plan_starts_at_the_point_nearest_to_the_robot():
    plan = np.column_stack([np.zeros(5), np.arange(5.0)])
    assert np.array_equal(tj.remaining_plan(plan, 0.3, 2.2), plan[2:])
    assert np.array_equal(tj.remaining_plan(plan, 0.0, -1.0), plan)  # robot before the start
    assert tj.remaining_plan(np.empty((0, 2)), 0.0, 0.0).shape == (0, 2)


def test_transform_expresses_points_and_poses_in_the_target_frame():
    # the source frame sits at (1, 2) of the target frame, turned by 90 degrees
    tx, ty, yaw = 1.0, 2.0, math.pi / 2
    assert np.allclose(transform_points([[1.0, 0.0], [0.0, 1.0]], tx, ty, yaw), [[1.0, 3.0], [0.0, 2.0]])
    pose = transform_pose(Pose2D(1.0, 0.0, math.pi), tx, ty, yaw)
    assert np.allclose((pose.x, pose.y, pose.theta), (1.0, 3.0, -math.pi / 2))
    assert transform_pose(Pose2D(0.5, -0.5, 0.3), 0.0, 0.0, 0.0) == Pose2D(0.5, -0.5, 0.3)
