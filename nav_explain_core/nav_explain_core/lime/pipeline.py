"""End-to-end LIME explanation of a local-path-planning decision.

``LimeCostmapExplainer.explain`` is the ROS-free equivalent of the original
``lime_rt_pub`` orchestration:

1. normalise the local costmap (lethal->99, else->0);
2. segment it into obstacle features (SLIC + split-until-N);
3. build perturbations (turn obstacle segments on/off);
4. for each perturbation, query the black-box planner and measure the deviation
   of the resulting local plan from the global plan (the LIME target);
5. fit the kernel-weighted Ridge surrogate;
6. colour each segment by its coefficient.

Steps 1 to 4 are :func:`~nav_explain_core.perturbation_dataset.build_perturbation_dataset`.
A dataset that already exists can be explained with ``explain_dataset``, which
needs no further planner query.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..perturbation import MODES
from ..perturbation_dataset import (
    DEFAULT_SEGMENTS,
    PerturbationDataset,
    ScenarioInstance,
    build_perturbation_dataset,
)
from ..planner.base import PlannerBackend
from .coloring import colorize_segments
from .ridge_explainer import Explanation, RidgeExplainer


@dataclass
class ExplanationResult:
    """Everything produced for one explanation."""

    #: integer label map of the costmap, 0 = free space
    segments: np.ndarray
    #: the fitted surrogate
    explanation: Explanation
    #: ``(H, W, 3)`` RGB image in [0, 1]
    explanation_image: np.ndarray
    #: the planner's local plan for the unperturbed instance
    original_local_plan: np.ndarray
    #: deviation of that local plan from the global plan
    original_deviation: float
    #: number of interpretable features, free space included
    n_features: int
    #: number of planner queries behind the explanation
    n_perturbations: int
    #: wall-clock seconds per stage
    timings: dict[str, float]

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
        n_target_segments: int = DEFAULT_SEGMENTS,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.planner = planner
        self.mode = mode
        self.explainer = explainer or RidgeExplainer()
        self.n_target_segments = n_target_segments

    def explain(self, instance: ScenarioInstance) -> ExplanationResult:
        """Query the planner for ``instance`` and explain the result."""
        dataset = build_perturbation_dataset(
            instance, self.planner, self.n_target_segments, self.mode
        )
        return self.explain_dataset(dataset)

    def explain_dataset(self, dataset: PerturbationDataset) -> ExplanationResult:
        """Fit the surrogate on an existing dataset (no planner query)."""
        t0 = time.perf_counter()
        explanation = self.explainer.explain(
            dataset.data,
            dataset.deviations,
            segment_labels=dataset.labels,
            num_features=dataset.n_features,
        )
        image = colorize_segments(dataset.segments, explanation)
        timings = dict(dataset.timings)
        timings["surrogate_s"] = time.perf_counter() - t0
        timings["total_s"] = sum(timings.values())

        return ExplanationResult(
            segments=dataset.segments,
            explanation=explanation,
            explanation_image=image,
            original_local_plan=dataset.local_plan,
            original_deviation=float(dataset.deviations[0]),
            n_features=dataset.n_features,
            n_perturbations=len(dataset.data),
            timings=timings,
        )
