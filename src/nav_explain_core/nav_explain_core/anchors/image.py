"""Anchors explanation for a local costmap.

Reuses the same segmentation, perturbation and black-box planner as the LIME
explainer, but instead of fitting an additive surrogate it searches for an
*anchor*: a minimal set of obstacle segments whose presence keeps the robot's
"deviate" decision. The instance's continuous deviation is binarised into a
decision (``deviates`` iff deviation >= ``decision_ratio`` * instance deviation)
so the anchor rule reads: "as long as these obstacles are present, I deviate."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..costmap import normalize_costmap
from ..lime.perturbation import apply_perturbation, full_2n_perturbations
from ..lime.pipeline import ScenarioInstance
from ..lime.segmentation import slic_obstacle_segments
from ..planner.base import PlannerBackend, PlanRequest
from ..trajectory import deviation
from .base import AnchorExplanation, find_anchor

ANCHOR_COLOR = (0.0, 0.55, 0.9)  # cyan/blue highlight for anchor segments
OTHER_COLOR = (0.4, 0.4, 0.4)
FREE_COLOR = (0.7, 0.7, 0.7)


@dataclass
class AnchorResult:
    segments: np.ndarray
    explanation: AnchorExplanation
    anchor_segments: list[int]
    image: np.ndarray  # (H, W, 3) highlight of the anchor
    original_deviation: float
    decision_threshold: float
    n_perturbations: int
    timings: dict = field(default_factory=dict)


class AnchorImageExplainer:
    def __init__(
        self,
        planner: PlannerBackend,
        n_target_segments: int = 8,
        tau: float = 0.9,
        decision_ratio: float = 0.5,
    ):
        self.planner = planner
        self.n_target_segments = n_target_segments
        self.tau = tau
        self.decision_ratio = decision_ratio

    def _deviation(self, instance: ScenarioInstance, costmap: np.ndarray) -> float:
        result = self.planner.plan(
            PlanRequest(
                costmap=costmap,
                costmap_info=instance.costmap_info,
                global_plan=instance.global_plan,
                start_pose=instance.start_pose,
                footprint_radius=instance.footprint_radius,
                vel_limits=instance.vel_limits,
            )
        )
        return deviation(result.local_plan, instance.global_plan)

    def explain(self, instance: ScenarioInstance) -> AnchorResult:
        t0 = time.perf_counter()
        costmap = normalize_costmap(instance.costmap)
        segments = slic_obstacle_segments(costmap, n_target=self.n_target_segments)
        labels = np.unique(segments)

        data = full_2n_perturbations(len(labels))
        fudged = np.zeros_like(costmap)
        devs = np.array(
            [self._deviation(instance, apply_perturbation(costmap, segments, row, fudged)) for row in data]
        )

        d0 = float(devs[0])
        threshold = self.decision_ratio * d0
        y = (devs >= threshold).astype(int)  # 1 = "deviates" like the instance

        explanation = find_anchor(data, y, tau=self.tau)
        anchor_segments = [int(labels[f]) for f in explanation.features]
        explanation.segment_labels = anchor_segments

        image = self._colorize(segments, anchor_segments)
        return AnchorResult(
            segments=segments,
            explanation=explanation,
            anchor_segments=anchor_segments,
            image=image,
            original_deviation=d0,
            decision_threshold=threshold,
            n_perturbations=len(data),
            timings={"total_s": time.perf_counter() - t0},
        )

    @staticmethod
    def _colorize(segments: np.ndarray, anchor_segments: list[int]) -> np.ndarray:
        h, w = segments.shape
        img = np.zeros((h, w, 3), dtype=np.float64)
        img[segments == 0] = FREE_COLOR
        for label in np.unique(segments):
            if label == 0:
                continue
            img[segments == label] = ANCHOR_COLOR if label in anchor_segments else OTHER_COLOR
        return img
