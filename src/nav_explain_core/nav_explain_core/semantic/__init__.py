"""Semantic-map, affordance, QSR and textual-explanation layer."""

from .affordance import (
    build_affordance_perturbations, object_affordance_pairs, perturb_for_pair,
)
from .ontology import ObjectEntry, Ontology, load_ontology
from .qsr import (
    QUADRANTS, DIRECTIONS, qsr_for_object, relative_angle,
    single_cross_quadrant, ternary_direction, ternary_qsr,
)
from .semantic_map import (
    SemanticMapInfo, build_semantic_map, inflate, object_centroids,
    semantic_to_costmap,
)
from .templater import descriptive, suggestive

__all__ = [
    "build_affordance_perturbations", "object_affordance_pairs", "perturb_for_pair",
    "ObjectEntry", "Ontology", "load_ontology",
    "QUADRANTS", "DIRECTIONS", "qsr_for_object", "relative_angle",
    "single_cross_quadrant", "ternary_direction", "ternary_qsr",
    "SemanticMapInfo", "build_semantic_map", "inflate", "object_centroids",
    "semantic_to_costmap", "descriptive", "suggestive",
]

from .pipeline import VisuoTextualExplainer, VTInstance, VTResult, color_semantic_pair  # noqa: E402

__all__ += ["VisuoTextualExplainer", "VTInstance", "VTResult", "color_semantic_pair"]
