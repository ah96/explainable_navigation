"""Shared perturbation dataset for all costmap explanation methods.

Building the dataset means querying the black-box planner once per perturbation
(the expensive step). LIME, SHAP, Anchors and counterfactual explanations all
consume the *same* dataset, so the bringup "compare all methods" demo pays the
planner cost only once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .costmap import normalize_costmap
from .lime.perturbation import apply_perturbation, full_2n_perturbations, one_off_perturbations
from .lime.pipeline import ScenarioInstance
from .lime.segmentation import slic_obstacle_segments
from .planner.base import PlannerBackend, PlanRequest
from .trajectory import deviation


@dataclass
class PerturbationDataset:
    """Perturbations + per-perturbation deviation for one scenario."""

    segments: np.ndarray
    labels: np.ndarray            # unique segment ids; column j <-> labels[j]
    data: np.ndarray              # (n_perturbations, n_features) binary
    deviations: np.ndarray        # (n_perturbations,) planner deviation
    timings: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def obstacle_columns(self) -> list[int]:
        """Feature columns that are obstacles (all except free space, column 0)."""
        return [j for j in range(self.n_features) if self.labels[j] != 0]


def build_perturbation_dataset(
    instance: ScenarioInstance,
    planner: PlannerBackend,
    n_target_segments: int = 8,
    mode: str = "full_2n",
) -> PerturbationDataset:
    t0 = time.perf_counter()
    costmap = normalize_costmap(instance.costmap)
    segments = slic_obstacle_segments(costmap, n_target=n_target_segments)
    labels = np.unique(segments)
    seg_s = time.perf_counter() - t0

    data = full_2n_perturbations(len(labels)) if mode == "full_2n" else one_off_perturbations(len(labels))
    fudged = np.zeros_like(costmap)

    t1 = time.perf_counter()
    deviations = np.empty(len(data), dtype=np.float64)
    for i, row in enumerate(data):
        perturbed = apply_perturbation(costmap, segments, row, fudged)
        result = planner.plan(
            PlanRequest(
                costmap=perturbed,
                costmap_info=instance.costmap_info,
                global_plan=instance.global_plan,
                start_pose=instance.start_pose,
                footprint_radius=instance.footprint_radius,
                vel_limits=instance.vel_limits,
            )
        )
        deviations[i] = deviation(result.local_plan, instance.global_plan)
    planner_s = time.perf_counter() - t1

    return PerturbationDataset(
        segments=segments,
        labels=labels,
        data=data,
        deviations=deviations,
        timings={"segmentation_s": seg_s, "planner_s": planner_s},
    )
