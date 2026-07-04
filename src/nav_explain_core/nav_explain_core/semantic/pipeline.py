"""End-to-end visuo-textual explanation of a local navigational choice.

Pipeline (HRI 2023):

1. rasterise + inflate the ontology into a semantic map;
2. enumerate (object, affordance) interpretable features;
3. perturb each feature (apply its affordance) and ask the planner for the local
   plan, measuring its deviation from the global plan (the LIME target);
4. fit the kernel-weighted Ridge surrogate -> a coefficient per feature;
5. colour each object on the semantic map per feature (the *visual* explanation);
6. for the deviation-driving features, place the object with QSR and emit the
   descriptive + suggestive *textual* explanations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose2D, VelLimits
from ..planner.base import PlannerBackend, PlanRequest
from ..lime.ridge_explainer import Explanation, RidgeExplainer
from ..trajectory import deviation
from .affordance import build_affordance_perturbations, object_affordance_pairs
from .ontology import Ontology
from .qsr import qsr_for_object
from .semantic_map import (
    SemanticMapInfo,
    build_semantic_map,
    inflate,
    object_centroids,
    semantic_to_costmap,
)
from .templater import descriptive, suggestive

# RGB shades for the semantic explanation maps.
FREE_SHADE = (0.85, 0.85, 0.85)
OBJECT_SHADE = (0.55, 0.55, 0.55)


@dataclass
class VTInstance:
    """One navigation situation described by a semantic map + a global plan."""

    ontology: Ontology
    map_info: SemanticMapInfo
    global_plan: np.ndarray
    robot_pose: Pose2D
    footprint_radius: float = 0.275
    inflation_radius: float = 0.275
    vel_limits: VelLimits = field(default_factory=VelLimits)
    name: str = "scenario"


@dataclass
class VTResult:
    coefficients: dict[tuple[int, str], float]  # (object_id, affordance) -> coef
    explanation_maps: dict[tuple[int, str], np.ndarray]  # RGB per feature
    descriptive_text: list[str]
    suggestive_text: list[str]
    semantic_map: np.ndarray
    original_local_plan: np.ndarray
    original_deviation: float
    pairs: list[tuple[int, str]]
    runtime_s: float


def color_semantic_pair(
    semantic_map: np.ndarray,
    obj_id: int,
    coef: float,
    max_abs: float,
) -> np.ndarray:
    """RGB map with every object grey and ``obj_id`` coloured by its coefficient.

    Green = the object increases the deviation; red = it decreases it (paper
    convention). Intensity is proportional to ``|coef| / max_abs``.
    """
    h, w = semantic_map.shape
    img = np.full((h, w, 3), FREE_SHADE, dtype=np.float64)
    img[semantic_map > 0] = OBJECT_SHADE
    mask = semantic_map == obj_id
    intensity = abs(coef) / max_abs if max_abs > 0 else 0.0
    if coef > 0:
        img[mask] = (0.0, intensity, 0.0)
    elif coef < 0:
        img[mask] = (intensity, 0.0, 0.0)
    return img


class VisuoTextualExplainer:
    def __init__(
        self,
        planner: PlannerBackend,
        mode: str = "one_off",
        explainer: RidgeExplainer | None = None,
        text_top_k: int = 3,
        text_threshold: float = 1e-6,
    ):
        if mode not in ("one_off", "full_2n"):
            raise ValueError("mode must be 'one_off' or 'full_2n'")
        self.planner = planner
        self.mode = mode
        self.explainer = explainer or RidgeExplainer()
        self.text_top_k = text_top_k
        self.text_threshold = text_threshold

    def _plan_deviation(self, instance: VTInstance, costmap: np.ndarray):
        req = PlanRequest(
            costmap=costmap,
            costmap_info=instance.map_info.to_costmap_info(),
            global_plan=instance.global_plan,
            start_pose=instance.robot_pose,
            footprint_radius=instance.footprint_radius,
            vel_limits=instance.vel_limits,
        )
        result = self.planner.plan(req)
        return result.local_plan, deviation(result.local_plan, instance.global_plan)

    def explain(self, instance: VTInstance) -> VTResult:
        t0 = time.perf_counter()
        info = instance.map_info
        smap = build_semantic_map(instance.ontology, info)
        smap_infl = inflate(smap, info, instance.inflation_radius)
        ids_present = {int(v) for v in np.unique(smap) if v != 0}
        pairs = object_affordance_pairs(instance.ontology, ids_present)
        if not pairs:
            raise ValueError("no object-affordance pairs in this scenario")

        data, costmaps, labels = build_affordance_perturbations(
            smap_infl, info, instance.ontology, pairs, self.mode
        )

        targets = np.empty(len(costmaps), dtype=np.float64)
        original_local_plan = np.empty((0, 2))
        for i, cm in enumerate(costmaps):
            local_plan, dev = self._plan_deviation(instance, cm)
            targets[i] = dev
            if i == 0:
                original_local_plan = local_plan

        expl: Explanation = self.explainer.explain(
            data, targets, segment_labels=labels, num_features=len(labels)
        )
        # column label k (1..F) -> pair k-1
        coefficients = {pairs[k - 1]: expl.coef.get(k, 0.0) for k in range(1, len(labels))}

        max_abs = max((abs(v) for v in coefficients.values()), default=1.0) or 1.0
        explanation_maps = {
            pair: color_semantic_pair(smap, pair[0], coef, max_abs)
            for pair, coef in coefficients.items()
        }

        descriptive_text, suggestive_text = self._textual(
            instance, coefficients
        )

        return VTResult(
            coefficients=coefficients,
            explanation_maps=explanation_maps,
            descriptive_text=descriptive_text,
            suggestive_text=suggestive_text,
            semantic_map=smap,
            original_local_plan=original_local_plan,
            original_deviation=float(targets[0]),
            pairs=pairs,
            runtime_s=time.perf_counter() - t0,
        )

    def _textual(self, instance: VTInstance, coefficients):
        centroids = object_centroids(instance.ontology)
        labels = {o.id: o.label for o in instance.ontology}
        ranked = sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)
        descr, sugg = [], []
        for (obj_id, aff), coef in ranked[: self.text_top_k]:
            if abs(coef) < self.text_threshold:
                continue
            qsr = qsr_for_object(instance.robot_pose, centroids[obj_id])
            label = labels[obj_id]
            descr.append(descriptive(label, qsr, increases=coef > 0))
            sugg.append(suggestive(label, qsr, aff))
        return descr, sugg
