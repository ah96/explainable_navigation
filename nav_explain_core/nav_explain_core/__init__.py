"""nav_explain_core: framework-agnostic core for explainable robot navigation.

A single, ROS-free library shared by every ROS 2 package in the workspace. It
provides:

* a deterministic local-planner black box (``planner``) standing in for TEB;
* the ``deviation`` of a local plan from the global plan (``trajectory``);
* the interpretable features of a costmap (``segmentation``) and their on/off
  combinations (``perturbation``);
* a shared perturbation dataset (``perturbation_dataset``) built once per
  scenario and consumed by every costmap explanation method;
* four costmap explanation methods: **LIME** (``lime``), **SHAP** (``shap``),
  **Anchors** (``anchors``) and **counterfactual** (``counterfactual``);
* the semantic map, affordance, QSR and text layer (``semantic``) for
  visuo-textual explanations;
* the explanation loop that depends on the robot's personality (``hixron``);
* the point-cloud payload the live nodes publish (``pointcloud``).

Nothing here imports ROS, so it runs headless in a plain Python environment and
is unit-tested on its own.
"""

__version__ = "0.1.0"

from .costmap import CostmapInfo, world_to_pixel, world_to_pixels
from .geometry import Pose2D
from .perturbation_dataset import (
    DEFAULT_SEGMENTS,
    PerturbationDataset,
    ScenarioInstance,
    build_perturbation_dataset,
)

__all__ = [
    "CostmapInfo", "world_to_pixel", "world_to_pixels", "Pose2D",
    "DEFAULT_SEGMENTS", "PerturbationDataset", "ScenarioInstance",
    "build_perturbation_dataset",
]
