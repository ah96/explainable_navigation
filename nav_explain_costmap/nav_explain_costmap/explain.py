"""Unified dispatch across the four costmap explanation methods.

Each method turns the same scenario into a coloured explanation image plus a
short textual/structured summary, so they can be produced and compared uniformly.

SHAP, Anchors and the counterfactual need the full perturbation set. They share
one :class:`~nav_explain_core.PerturbationDataset`, i.e. one planner pass, when a
dataset is passed in. LIME runs in its fast ``one_off`` mode, which takes one
planner query per obstacle segment and is therefore built separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nav_explain_core import (
    DEFAULT_SEGMENTS,
    PerturbationDataset,
    ScenarioInstance,
    build_perturbation_dataset,
)
from nav_explain_core.anchors.image import AnchorImageExplainer, highlight_segments
from nav_explain_core.counterfactual import counterfactual
from nav_explain_core.lime import LimeCostmapExplainer, colorize_segments
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.planner.base import PlannerBackend
from nav_explain_core.shap import shapley_values

METHODS = ("lime", "shap", "anchors", "counterfactual")

#: how many segments the textual summary of LIME and SHAP names
TOP_SEGMENTS = 3

# colours of the counterfactual image: segments to remove, and the others
_CF_REMOVE = (0.9, 0.1, 0.1)
_CF_KEEP = (0.45, 0.45, 0.45)


@dataclass
class CostmapExplanation:
    """The result of one method, in the form shared by all four."""

    #: one of :data:`METHODS`
    method: str
    #: ``(H, W, 3)`` RGB image in [0, 1]
    image: np.ndarray
    #: integer label map of the costmap, 0 = free space
    segments: np.ndarray
    #: the planner's local plan for the unperturbed instance
    local_plan: np.ndarray
    #: the numbers of the explanation, by name
    summary: dict = field(default_factory=dict)
    #: the explanation as one sentence
    text: str = ""


def _nothing_to_explain(method: str, dataset: PerturbationDataset) -> CostmapExplanation:
    """The explanation of a costmap without obstacles: there is no feature to blame."""
    image = highlight_segments(dataset.segments, [], _CF_REMOVE)
    return CostmapExplanation(method, image, dataset.segments, dataset.local_plan, {},
                              "no obstacle in the costmap, nothing to explain")


def explain_costmap(
    instance: ScenarioInstance,
    method: str = "lime",
    planner: PlannerBackend | None = None,
    n_target_segments: int = DEFAULT_SEGMENTS,
    dataset: PerturbationDataset | None = None,
) -> CostmapExplanation:
    """Explain ``instance`` with one method.

    ``dataset`` is a full (``full_2n``) perturbation dataset of the same instance.
    Pass it to run several methods on one planner pass. Without it, the method
    builds what it needs.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    planner = planner or ElasticBandPlanner()

    if method == "lime":
        one_off = build_perturbation_dataset(instance, planner, n_target_segments, mode="one_off")
        if not one_off.obstacle_columns:
            return _nothing_to_explain(method, one_off)
        r = LimeCostmapExplainer().explain(one_off)
        text = "; ".join(
            f"segment {s} {'increases' if r.coefficients[s] > 0 else 'decreases'} deviation"
            for s in r.explanation.order[:TOP_SEGMENTS] if s != 0
        )
        return CostmapExplanation("lime", r.explanation_image, r.segments,
                                  r.original_local_plan, dict(r.coefficients), text)

    if dataset is None:
        dataset = build_perturbation_dataset(instance, planner, n_target_segments)
    if not dataset.obstacle_columns:
        return _nothing_to_explain(method, dataset)

    if method == "shap":
        sh = shapley_values(dataset)
        image = colorize_segments(dataset.segments, sh.values)
        text = "; ".join(f"segment {s}: {sh.values[s]:+.2f}" for s in sh.order[:TOP_SEGMENTS] if s != 0)
        return CostmapExplanation("shap", image, dataset.segments, dataset.local_plan,
                                  dict(sh.values), text)

    if method == "anchors":
        r = AnchorImageExplainer().explain(dataset)
        text = (f"anchor = segments {r.anchor_segments} "
                f"(precision {r.explanation.precision:.2f}, coverage {r.explanation.coverage:.2f}): "
                f"while these obstacles are present, the robot deviates")
        return CostmapExplanation("anchors", r.image, r.segments, dataset.local_plan,
                                  {"anchor_segments": r.anchor_segments,
                                   "precision": r.explanation.precision,
                                   "coverage": r.explanation.coverage}, text)

    # counterfactual
    cf = counterfactual(dataset)
    image = highlight_segments(dataset.segments, cf.removed_segments, _CF_REMOVE, _CF_KEEP)
    text = (f"remove segments {cf.removed_segments} -> deviation "
            f"{cf.resulting_deviation:.2f} (from {cf.instance_deviation:.2f}); "
            + ("target met" if cf.found else "target not reachable"))
    return CostmapExplanation("counterfactual", image, dataset.segments, dataset.local_plan,
                              {"removed_segments": cf.removed_segments,
                               "resulting_deviation": cf.resulting_deviation,
                               "instance_deviation": cf.instance_deviation,
                               "found": cf.found}, text)
