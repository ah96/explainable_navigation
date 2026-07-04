"""nav_explain_core — framework-agnostic core for explainable robot navigation.

A single, ROS-free library shared by every ROS2 package in the workspace. It
provides:

* a deterministic local-planner black box (``planner``) standing in for TEB;
* trajectory-distance metrics (``trajectory``) incl. the ``deviation`` target;
* a shared perturbation dataset (``perturbation_dataset``) built once per
  scenario and consumed by every costmap explanation method;
* four costmap explanation methods — **LIME** (``lime``), **SHAP** (``shap``),
  **Anchors** (``anchors``) and **counterfactual** (``counterfactual``);
* the semantic / affordance / QSR / textual layer (``semantic``) for
  visuo-textual explanations.

Nothing here imports ROS, so it runs headless in a plain Python environment and
is unit-tested on its own.
"""

__version__ = "0.1.0"

from .costmap import CostmapInfo, normalize_costmap, pixel_to_world, world_to_pixel
from .geometry import Pose2D, VelLimits
from .lime import LimeCostmapExplainer, ScenarioInstance
from .perturbation_dataset import PerturbationDataset, build_perturbation_dataset
from .planner import ElasticBandPlanner

__all__ = [
    "CostmapInfo", "normalize_costmap", "pixel_to_world", "world_to_pixel",
    "Pose2D", "VelLimits",
    "LimeCostmapExplainer", "ScenarioInstance",
    "PerturbationDataset", "build_perturbation_dataset",
    "ElasticBandPlanner",
]
