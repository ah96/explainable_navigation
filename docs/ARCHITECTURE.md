# Architecture

A colcon workspace of ROS2 packages sharing one **ROS-free core**.

```
nav_explain_core  (pure Python: algorithms, no rclpy)
    ├── costmap.py, geometry.py          value/coordinate conventions
    ├── planner/                         PlannerBackend + deterministic ElasticBandPlanner
    ├── trajectory.py                    deviation target + metric library
    ├── perturbation_dataset.py          build once, shared by all costmap methods
    ├── lime/                            LIME segmentation/perturbation/ridge/coloring/pipeline
    ├── shap.py                          exact Shapley values
    ├── anchors/                         deterministic anchor search + image explainer
    ├── counterfactual.py                minimal removal set
    ├── dataset.py                       cache a perturbation dataset to .npz
    └── semantic/                        ontology, semantic map, affordance, QSR, templater,
                                         visuo-textual pipeline
        │
        ▼ depended on by
  nav_explain_costmap   nav_explain_semantic   nav_explain_perception
  nav_explain_sim       nav_explain_msgs       nav_explain_bringup
```

## The shared-dataset seam
The planner queries dominate runtime, so `build_perturbation_dataset()` runs the
planner once per scenario and LIME/SHAP/Anchors/counterfactual all consume the
same `PerturbationDataset`. The bringup demo builds it once and compares all four
methods with no extra planner cost.

## The PlannerBackend seam
The explanation black box is a `PlannerBackend` (`plan(PlanRequest) -> PlanResult`).
The default `ElasticBandPlanner` is deterministic and runs anywhere; a
`Ros2ControllerPlanner` stub is the documented route to a live Nav2/TEB controller
for exact-fidelity numbers.

## Headless vs. ROS
`nav_explain_core` imports no ROS and is unit-tested standalone. Each package's
`ros/` layer is guarded (`_ros_guard.py`): the node classes are only defined when
rclpy + the ML stack are both importable, so `colcon build` and the headless apps
never break when they aren't.
