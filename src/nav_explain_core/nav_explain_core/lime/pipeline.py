"""End-to-end LIME explanation of a local-path-planning decision.

``LimeCostmapExplainer.explain`` is the de-ROS'd equivalent of the original
``lime_rt_pub`` orchestration:

1. normalise the local costmap (lethal->99, else->0);
2. segment it into obstacle features (SLIC + split-until-N);
3. build perturbations (turn obstacle segments on/off);
4. for each perturbation, query the black-box planner and measure the deviation
   of the resulting local plan from the global plan (the LIME target);
5. fit the kernel-weighted Ridge surrogate;
6. colour each segment by its coefficient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .coloring import colorize_segments
from ..costmap import CostmapInfo, normalize_costmap
from ..geometry import Pose2D, VelLimits
from .perturbation import (
    apply_perturbation,
    full_2n_perturbations,
    one_off_perturbations,
)
from ..planner.base import PlannerBackend, PlanRequest
from .ridge_explainer import Explanation, RidgeExplainer
from .segmentation import slic_obstacle_segments
from ..trajectory import deviation


@dataclass
class ScenarioInstance:
    """One local-navigation situation to be explained."""

    costmap: np.ndarray  # raw occupancy grid (0..100)
    costmap_info: CostmapInfo
    global_plan: np.ndarray  # (N, 2) world coords
    start_pose: Pose2D
    footprint_radius: float = 0.275
    vel_limits: VelLimits = field(default_factory=VelLimits)
    name: str = "scenario"
    description: str = ""


@dataclass
class ExplanationResult:
    """Everything produced for one explanation."""

    segments: np.ndarray
    explanation: Explanation
    explanation_image: np.ndarray  # (H, W, 3) RGB in [0, 1]
    original_local_plan: np.ndarray
    original_deviation: float
    n_features: int
    n_perturbations: int
    timings: dict

    @property
    def coefficients(self) -> dict[int, float]:
        return self.explanation.coef


class LimeCostmapExplainer:
    """Explains a local planner's deviation from the global plan with LIME."""

    def __init__(
        self,
        planner: PlannerBackend,
        mode: str = "one_off",
        explainer: RidgeExplainer | None = None,
        n_target_segments: int = 8,
    ):
        if mode not in ("one_off", "full_2n"):
            raise ValueError("mode must be 'one_off' or 'full_2n'")
        self.planner = planner
        self.mode = mode
        self.explainer = explainer or RidgeExplainer()
        self.n_target_segments = n_target_segments

    def _plan_deviation(self, instance: ScenarioInstance, costmap: np.ndarray) -> tuple[np.ndarray, float]:
        req = PlanRequest(
            costmap=costmap,
            costmap_info=instance.costmap_info,
            global_plan=instance.global_plan,
            start_pose=instance.start_pose,
            footprint_radius=instance.footprint_radius,
            vel_limits=instance.vel_limits,
        )
        result = self.planner.plan(req)
        return result.local_plan, deviation(result.local_plan, instance.global_plan)

    def explain(self, instance: ScenarioInstance) -> ExplanationResult:
        timings: dict = {}

        t0 = time.perf_counter()
        costmap = normalize_costmap(instance.costmap)
        segments = slic_obstacle_segments(costmap, n_target=self.n_target_segments)
        labels = np.unique(segments)
        n_features = len(labels)
        timings["segmentation_s"] = time.perf_counter() - t0

        # Perturbation matrix over features (column 0 = free space, always on).
        if self.mode == "one_off":
            data = one_off_perturbations(n_features)
        else:
            data = full_2n_perturbations(n_features)

        # Black-box queries: deviation per perturbation.
        t1 = time.perf_counter()
        fudged = np.zeros_like(costmap)
        targets = np.empty(len(data), dtype=np.float64)
        original_local_plan = np.empty((0, 2))
        for i, row in enumerate(data):
            perturbed = apply_perturbation(costmap, segments, row, fudged)
            local_plan, dev = self._plan_deviation(instance, perturbed)
            targets[i] = dev
            if i == 0:
                original_local_plan = local_plan
        timings["planner_s"] = time.perf_counter() - t1

        # Surrogate fit + colouring.
        t2 = time.perf_counter()
        explanation = self.explainer.explain(
            data, targets, segment_labels=labels, num_features=n_features
        )
        image = colorize_segments(segments, explanation)
        timings["surrogate_s"] = time.perf_counter() - t2
        timings["total_s"] = time.perf_counter() - t0

        return ExplanationResult(
            segments=segments,
            explanation=explanation,
            explanation_image=image,
            original_local_plan=original_local_plan,
            original_deviation=float(targets[0]),
            n_features=n_features,
            n_perturbations=len(data),
            timings=timings,
        )
