"""Planner determinism/behaviour, segmentation and perturbations."""

import numpy as np

from nav_explain_core import CostmapInfo, Pose2D
from nav_explain_core.costmap import normalize_costmap
from nav_explain_core.lime.perturbation import full_2n_perturbations, one_off_perturbations
from nav_explain_core.lime.segmentation import merge_until_n, slic_obstacle_segments, split_until_n
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.planner.base import PlanRequest


def _req(costmap):
    info = CostmapInfo(resolution=0.05, origin_x=-2.0, origin_y=-2.0, width=80, height=80)
    gy = np.linspace(0.0, 1.8, 30)
    plan = np.column_stack([np.zeros_like(gy), gy])
    return PlanRequest(costmap=costmap, costmap_info=info, global_plan=plan,
                       start_pose=Pose2D(0, 0, np.pi / 2), footprint_radius=0.275)


def test_planner_deterministic():
    cm = np.zeros((80, 80)); cm[40:55, 35:45] = 99
    p = ElasticBandPlanner()
    assert np.array_equal(p.plan(_req(cm)).local_plan, p.plan(_req(cm)).local_plan)


def test_planner_bows_around_obstacle():
    free = np.zeros((80, 80)); blocked = free.copy(); blocked[40:55, 38:42] = 99
    p = ElasticBandPlanner()
    off = np.abs(p.plan(_req(free)).local_plan[:, 0]).max()
    on = np.abs(p.plan(_req(blocked)).local_plan[:, 0]).max()
    assert on > off + 0.05


def test_segmentation_reaches_target():
    cm = np.zeros((80, 80)); cm[30:55, 30:55] = 100
    seg = slic_obstacle_segments(normalize_costmap(cm), n_target=8)
    assert len([l for l in np.unique(seg) if l != 0]) == 8


def test_split_and_merge():
    seg = np.zeros((40, 40), dtype=np.int32); seg[5:35, 5:35] = 1
    assert len([l for l in np.unique(split_until_n(seg, 4)) if l]) == 4
    many = np.zeros((20, 60), dtype=np.int32)
    for k in range(1, 7):
        many[5:15, (k - 1) * 10:k * 10] = k
    assert len([l for l in np.unique(merge_until_n(many, 3)) if l]) == 3


def test_perturbations():
    assert one_off_perturbations(5).shape == (5, 5)
    assert (one_off_perturbations(5)[:, 0] == 1).all()
    data = full_2n_perturbations(4)
    assert data.shape == (8, 4) and (data[:, 0] == 1).all()
