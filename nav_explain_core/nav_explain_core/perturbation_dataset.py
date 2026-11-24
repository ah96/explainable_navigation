"""Shared perturbation dataset for all costmap explanation methods.

Building the dataset means querying the black-box planner once per perturbation,
which is the expensive step. This module is the only place of the costmap
pipeline that queries the planner. LIME, SHAP, Anchors and counterfactual
explanations all read a :class:`PerturbationDataset`, so comparing the methods
costs one planner pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .costmap import CostmapInfo, normalize_costmap
from .geometry import Pose2D, VelLimits
from .perturbation import apply_perturbation, perturbations
from .planner.base import PlannerBackend, PlanRequest
from .segmentation import slic_obstacle_segments
from .trajectory import deviation

#: default number of obstacle segments (interpretable features) per costmap
DEFAULT_SEGMENTS = 8


@dataclass
class ScenarioInstance:
    """One local-navigation situation to be explained."""

    #: raw occupancy grid, values 0..100
    costmap: np.ndarray
    #: geometry of the grid
    costmap_info: CostmapInfo
    #: global plan, ``(N, 2)`` world coordinates
    global_plan: np.ndarray
    #: robot pose at the moment of the explanation
    start_pose: Pose2D
    #: radius of the circular robot footprint in metres
    footprint_radius: float = 0.275
    #: velocity and acceleration limits of the robot
    vel_limits: VelLimits = field(default_factory=VelLimits)
    #: short name, used for output folders and figure titles
    name: str = "scenario"
    #: one sentence on what the scenario shows
    description: str = ""


@dataclass
class PerturbationDataset:
    """Perturbations and the planner's deviation for each, for one scenario."""

    #: integer label map of the costmap, 0 = free space
    segments: np.ndarray
    #: unique segment labels; column ``j`` of ``data`` belongs to ``labels[j]``
    labels: np.ndarray
    #: ``(n_perturbations, n_features)`` binary matrix, row 0 is the instance
    data: np.ndarray
    #: ``(n_perturbations,)`` deviation of the local plan from the global plan
    deviations: np.ndarray
    #: the planner's local plan for the unperturbed instance, ``(M, 2)``
    local_plan: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    #: wall-clock seconds of the segmentation and of the planner pass
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def obstacle_columns(self) -> list[int]:
        """Feature columns that are obstacles (all except free space, label 0)."""
        return [j for j in range(self.n_features) if self.labels[j] != 0]


def plan_request(instance: ScenarioInstance, costmap: np.ndarray) -> PlanRequest:
    """The planner query for ``instance`` with its costmap replaced by ``costmap``."""
    return PlanRequest(
        costmap=costmap,
        costmap_info=instance.costmap_info,
        global_plan=instance.global_plan,
        start_pose=instance.start_pose,
        footprint_radius=instance.footprint_radius,
        vel_limits=instance.vel_limits,
    )


def build_perturbation_dataset(
    instance: ScenarioInstance,
    planner: PlannerBackend,
    n_target_segments: int = DEFAULT_SEGMENTS,
    mode: str = "full_2n",
) -> PerturbationDataset:
    """Segment the costmap, perturb it and query the planner once per perturbation.

    ``mode`` is ``"full_2n"`` (every on/off combination of the obstacle segments,
    needed by SHAP, Anchors and the counterfactual) or ``"one_off"`` (one
    perturbation per segment, enough for a fast LIME explanation).
    """
    t0 = time.perf_counter()
    costmap = normalize_costmap(instance.costmap)
    segments = slic_obstacle_segments(costmap, n_target=n_target_segments)
    labels = np.unique(segments)
    data = perturbations(len(labels), mode)
    segmentation_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    deviations = np.empty(len(data), dtype=np.float64)
    local_plan = np.empty((0, 2))
    for i, row in enumerate(data):
        perturbed = apply_perturbation(costmap, segments, row, labels=labels)
        result = planner.plan(plan_request(instance, perturbed))
        deviations[i] = deviation(result.local_plan, instance.global_plan)
        if i == 0:  # row 0 is the unperturbed instance
            local_plan = result.local_plan
    planner_s = time.perf_counter() - t1

    return PerturbationDataset(
        segments=segments,
        labels=labels,
        data=data,
        deviations=deviations,
        local_plan=local_plan,
        timings={"segmentation_s": segmentation_s, "planner_s": planner_s},
    )
