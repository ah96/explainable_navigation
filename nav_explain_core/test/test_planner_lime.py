"""Planner determinism/behaviour, segmentation and perturbations."""

import numpy as np
import pytest

from nav_explain_core import CostmapInfo, Pose2D
from nav_explain_core.costmap import normalize_costmap, world_to_pixel, world_to_pixels
from nav_explain_core.perturbation import (
    apply_perturbation,
    full_2n_perturbations,
    one_off_perturbations,
    perturbations,
)
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.planner.base import PlanRequest
from nav_explain_core.segmentation import merge_until_n, slic_obstacle_segments, split_until_n


def _req(costmap):
    info = CostmapInfo(resolution=0.05, origin_x=-2.0, origin_y=-2.0, width=80, height=80)
    gy = np.linspace(0.0, 1.8, 30)
    plan = np.column_stack([np.zeros_like(gy), gy])
    return PlanRequest(costmap=costmap, costmap_info=info, global_plan=plan,
                       start_pose=Pose2D(0, 0, np.pi / 2), footprint_radius=0.275)


def test_planner_deterministic():
    cm = np.zeros((80, 80))
    cm[40:55, 35:45] = 99
    p = ElasticBandPlanner()
    assert np.array_equal(p.plan(_req(cm)).local_plan, p.plan(_req(cm)).local_plan)


def test_planner_bows_around_obstacle():
    free = np.zeros((80, 80))
    blocked = free.copy()
    blocked[40:55, 38:42] = 99
    p = ElasticBandPlanner()
    off = np.abs(p.plan(_req(free)).local_plan[:, 0]).max()
    on = np.abs(p.plan(_req(blocked)).local_plan[:, 0]).max()
    assert on > off + 0.05


def test_planner_follows_the_global_plan_in_free_space():
    req = _req(np.zeros((80, 80)))
    plan = ElasticBandPlanner().plan(req).local_plan
    assert np.abs(plan[:, 0]).max() < 1e-9  # the global plan runs along x = 0
    assert np.allclose(plan[0], (req.start_pose.x, req.start_pose.y))


def test_planner_reports_an_empty_plan_for_a_single_point():
    req = _req(np.zeros((80, 80)))
    short = PlanRequest(costmap=req.costmap, costmap_info=req.costmap_info,
                        global_plan=req.global_plan[:1], start_pose=req.start_pose)
    result = ElasticBandPlanner().plan(short)
    assert result.local_plan.shape == (0, 2) and result.meta["feasible"] is False


def test_world_to_pixels_agrees_with_the_scalar_version():
    info = CostmapInfo(resolution=0.05, origin_x=-2.0, origin_y=-1.0, width=80, height=80)
    points = np.array([[0.0, 0.0], [-1.99, 2.4], [1.234, -0.987], [-2.5, -1.5]])
    cols, rows = world_to_pixels(points, info)
    assert [(int(c), int(r)) for c, r in zip(cols, rows)] == [world_to_pixel(x, y, info) for x, y in points]


def test_segmentation_reaches_target():
    cm = np.zeros((80, 80))
    cm[30:55, 30:55] = 100
    seg = slic_obstacle_segments(normalize_costmap(cm), n_target=8)
    assert len([v for v in np.unique(seg) if v != 0]) == 8


def test_segmentation_of_free_space_is_one_segment():
    seg = slic_obstacle_segments(np.zeros((40, 40)), n_target=8)
    assert not seg.any()


def test_split_and_merge():
    seg = np.zeros((40, 40), dtype=np.int32)
    seg[5:35, 5:35] = 1
    assert len([v for v in np.unique(split_until_n(seg, 4)) if v]) == 4
    many = np.zeros((20, 60), dtype=np.int32)
    for k in range(1, 7):
        many[5:15, (k - 1) * 10:k * 10] = k
    assert len([v for v in np.unique(merge_until_n(many, 3)) if v]) == 3


def test_perturbations():
    assert one_off_perturbations(5).shape == (5, 5)
    assert (one_off_perturbations(5)[:, 0] == 1).all()
    data = full_2n_perturbations(4)
    assert data.shape == (8, 4) and (data[:, 0] == 1).all()
    assert (data[0] == 1).all()  # row 0 is the unperturbed instance
    assert full_2n_perturbations(1).shape == (1, 1)  # a costmap without obstacles
    assert np.array_equal(perturbations(4, "full_2n"), data)
    with pytest.raises(ValueError):
        perturbations(4, "some")


def test_apply_perturbation_hides_exactly_the_segments_turned_off():
    costmap = np.full((4, 6), 99.0)
    segments = np.repeat(np.array([[0, 0, 1, 1, 2, 2]]), 4, axis=0)
    out = apply_perturbation(costmap, segments, np.array([1, 0, 1]))
    assert (out[:, 2:4] == 0).all() and (out[:, :2] == 99).all() and (out[:, 4:] == 99).all()
    assert (costmap == 99).all()  # the input is not modified
