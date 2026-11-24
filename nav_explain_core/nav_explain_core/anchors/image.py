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

from ..perturbation_dataset import (
    DEFAULT_SEGMENTS,
    PerturbationDataset,
    ScenarioInstance,
    build_perturbation_dataset,
)
from ..planner.base import PlannerBackend
from .base import AnchorExplanation, find_anchor

ANCHOR_COLOR = (0.0, 0.55, 0.9)  # cyan/blue highlight for anchor segments
OTHER_COLOR = (0.4, 0.4, 0.4)
FREE_COLOR = (0.7, 0.7, 0.7)


@dataclass
class AnchorResult:
    """Everything produced for one anchor explanation."""

    #: integer label map of the costmap, 0 = free space
    segments: np.ndarray
    #: the anchor rule with its precision and coverage
    explanation: AnchorExplanation
    #: segment labels that form the anchor
    anchor_segments: list[int]
    #: ``(H, W, 3)`` RGB image that highlights the anchor
    image: np.ndarray
    #: deviation of the unperturbed instance
    original_deviation: float
    #: deviation from which a perturbation counts as "deviates"
    decision_threshold: float
    #: number of planner queries behind the explanation
    n_perturbations: int
    #: wall-clock seconds per stage
    timings: dict[str, float] = field(default_factory=dict)


def highlight_segments(
    segments: np.ndarray,
    chosen: list[int],
    chosen_color: tuple[float, float, float],
    other_color: tuple[float, float, float] = OTHER_COLOR,
    free_color: tuple[float, float, float] = FREE_COLOR,
) -> np.ndarray:
    """RGB image with the ``chosen`` segments in one colour and the rest in another."""
    img = np.empty((*segments.shape, 3), dtype=np.float64)
    img[:] = other_color
    img[segments == 0] = free_color
    img[np.isin(segments, chosen)] = chosen_color
    return img


class AnchorImageExplainer:
    """Finds the obstacle segments whose presence keeps the robot deviating."""

    def __init__(
        self,
        planner: PlannerBackend,
        n_target_segments: int = DEFAULT_SEGMENTS,
        tau: float = 0.9,
        decision_ratio: float = 0.5,
    ) -> None:
        self.planner = planner
        self.n_target_segments = n_target_segments
        self.tau = tau
        self.decision_ratio = decision_ratio

    def explain(self, instance: ScenarioInstance) -> AnchorResult:
        """Query the planner for ``instance`` and search the anchor."""
        dataset = build_perturbation_dataset(
            instance, self.planner, self.n_target_segments, mode="full_2n"
        )
        return self.explain_dataset(dataset)

    def explain_dataset(self, dataset: PerturbationDataset) -> AnchorResult:
        """Search the anchor on an existing full dataset (no planner query)."""
        t0 = time.perf_counter()
        d0 = float(dataset.deviations[0])
        threshold = self.decision_ratio * d0
        y = (dataset.deviations >= threshold).astype(int)  # 1 = "deviates" like the instance

        explanation = find_anchor(dataset.data, y, tau=self.tau)
        anchor_segments = [int(dataset.labels[f]) for f in explanation.features]
        explanation.segment_labels = anchor_segments

        timings = dict(dataset.timings)
        timings["search_s"] = time.perf_counter() - t0
        timings["total_s"] = sum(timings.values())
        return AnchorResult(
            segments=dataset.segments,
            explanation=explanation,
            anchor_segments=anchor_segments,
            image=highlight_segments(dataset.segments, anchor_segments, ANCHOR_COLOR),
            original_deviation=d0,
            decision_threshold=threshold,
            n_perturbations=len(dataset.data),
            timings=timings,
        )
