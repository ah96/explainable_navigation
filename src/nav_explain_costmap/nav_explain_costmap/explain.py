"""Unified dispatch across the four costmap explanation methods.

Each method turns the same scenario into a coloured explanation image plus a
short textual/structured summary, so they can be produced and compared uniformly.
LIME/SHAP/Anchors/counterfactual all share one perturbation dataset (one planner
pass) when run together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nav_explain_core import ScenarioInstance, build_perturbation_dataset
from nav_explain_core.anchors.image import AnchorImageExplainer
from nav_explain_core.counterfactual import counterfactual
from nav_explain_core.lime import LimeCostmapExplainer, colorize_segments
from nav_explain_core.lime.ridge_explainer import Explanation
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.planner.base import PlannerBackend
from nav_explain_core.shap import shapley_values

METHODS = ("lime", "shap", "anchors", "counterfactual")

# colouring for counterfactual "remove me" segments
_CF_REMOVE = (0.9, 0.1, 0.1)
_CF_KEEP = (0.45, 0.45, 0.45)
_FREE = (0.7, 0.7, 0.7)


@dataclass
class CostmapExplanation:
    method: str
    image: np.ndarray            # (H, W, 3) RGB in [0, 1]
    segments: np.ndarray
    local_plan: np.ndarray
    summary: dict = field(default_factory=dict)
    text: str = ""


def _cf_image(segments: np.ndarray, removed: list[int]) -> np.ndarray:
    h, w = segments.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[segments == 0] = _FREE
    for label in np.unique(segments):
        if label == 0:
            continue
        img[segments == label] = _CF_REMOVE if label in removed else _CF_KEEP
    return img


def explain_costmap(
    instance: ScenarioInstance,
    method: str = "lime",
    planner: PlannerBackend | None = None,
    n_target_segments: int = 8,
    dataset=None,
) -> CostmapExplanation:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    planner = planner or ElasticBandPlanner()

    if method == "lime":
        r = LimeCostmapExplainer(planner, mode="one_off", n_target_segments=n_target_segments).explain(instance)
        top = r.explanation.order[:3]
        text = "; ".join(
            f"segment {s} {'increases' if r.coefficients[s] > 0 else 'decreases'} deviation"
            for s in top if s != 0
        )
        return CostmapExplanation("lime", r.explanation_image, r.segments,
                                  r.original_local_plan, dict(r.coefficients), text)

    # shap / anchors / counterfactual share one dataset
    if dataset is None:
        dataset = build_perturbation_dataset(instance, planner, n_target_segments)

    if method == "shap":
        sh = shapley_values(dataset)
        expl = Explanation(coef=sh.values, order=sh.order, intercept=sh.base_value, score=0.0)
        image = colorize_segments(dataset.segments, expl)
        text = "; ".join(f"segment {s}: {sh.values[s]:+.2f}" for s in sh.order[:3] if s != 0)
        return CostmapExplanation("shap", image, dataset.segments, np.empty((0, 2)),
                                  dict(sh.values), text)

    if method == "anchors":
        r = AnchorImageExplainer(planner, n_target_segments=n_target_segments).explain(instance)
        text = (f"anchor = segments {r.anchor_segments} "
                f"(precision {r.explanation.precision:.2f}, coverage {r.explanation.coverage:.2f}): "
                f"while these obstacles are present, the robot deviates")
        return CostmapExplanation("anchors", r.image, r.segments, np.empty((0, 2)),
                                  {"anchor_segments": r.anchor_segments,
                                   "precision": r.explanation.precision,
                                   "coverage": r.explanation.coverage}, text)

    # counterfactual
    cf = counterfactual(dataset)
    image = _cf_image(dataset.segments, cf.removed_segments)
    text = (f"remove segments {cf.removed_segments} -> deviation "
            f"{cf.resulting_deviation:.2f} (from {cf.instance_deviation:.2f}); "
            + ("target met" if cf.found else "target not reachable"))
    return CostmapExplanation("counterfactual", image, dataset.segments, np.empty((0, 2)),
                              {"removed_segments": cf.removed_segments,
                               "resulting_deviation": cf.resulting_deviation,
                               "instance_deviation": cf.instance_deviation,
                               "found": cf.found}, text)
