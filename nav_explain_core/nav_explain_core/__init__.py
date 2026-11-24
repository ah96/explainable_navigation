"""nav_explain_core: framework-agnostic core for explainable robot navigation.

A single, ROS-free library shared by every ROS 2 package in the workspace. It
provides:

* a deterministic local-planner black box (``planner``) standing in for TEB;
* trajectory-distance metrics (``trajectory``) incl. the ``deviation`` target;
* the interpretable features of a costmap (``segmentation``) and their on/off
  combinations (``perturbation``);
* a shared perturbation dataset (``perturbation_dataset``) built once per
  scenario and consumed by every costmap explanation method;
* four costmap explanation methods: **LIME** (``lime``), **SHAP** (``shap``),
  **Anchors** (``anchors``) and **counterfactual** (``counterfactual``);
* the semantic / affordance / QSR / textual layer (``semantic``) for
  visuo-textual explanations;
* the point-cloud payload the live nodes publish (``pointcloud``).

Nothing here imports ROS, so it runs headless in a plain Python environment and
is unit-tested on its own.
"""

__version__ = "0.1.0"

from .costmap import (
    CostmapInfo,
    normalize_costmap,
    pixel_to_world,
    world_to_pixel,
    world_to_pixels,
)
from .geometry import Pose2D, VelLimits
from .lime import LimeCostmapExplainer
from .perturbation_dataset import (
    DEFAULT_SEGMENTS,
    PerturbationDataset,
    ScenarioInstance,
    build_perturbation_dataset,
)
from .planner import ElasticBandPlanner

__all__ = [
    "CostmapInfo", "normalize_costmap", "pixel_to_world", "world_to_pixel",
    "world_to_pixels", "Pose2D", "VelLimits",
    "LimeCostmapExplainer", "ScenarioInstance", "DEFAULT_SEGMENTS",
    "PerturbationDataset", "build_perturbation_dataset",
    "ElasticBandPlanner",
]
