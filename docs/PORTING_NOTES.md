# Porting notes (ROS1 origin → this workspace)

This workspace ports the *algorithms and logic* of an older ROS1 (catkin) research
repo into a clean, method-centric ROS2 codebase. The original repo is left
untouched; nothing here references specific conferences or papers.

## What was ported
- **LIME on the costmap** (segmentation, perturbation, weighted Ridge, coloring).
- **Visuo-textual** semantic explanation (LIME over object-affordance features +
  QSR + ontology + templated text).
- **Anchors** (`anchor_base.py`/`anchor_image.py`) — reimplemented deterministically.
- **Trajectory-distance metrics** (`traj_dist.py`) — full library, vectorised.
- **QSR** single-cross + ternary calculi (`qsr_rt.py`, `qsr_class_copy.py`).
- **YOLO perception** (`camera/detect_objects_rt.py`).

## Methods added for completeness
- **Exact SHAP** (Shapley values over the enumerated perturbation set).
- **Counterfactual** explanations.
Both reuse the shared perturbation dataset, giving four complementary costmap
methods instead of one.

## Cleanups / optimizations applied
- `deviation` / point-to-trajectory **vectorised** with `cdist` (was nested Python
  loops duplicated across three ROS1 files).
- The costmap conversion, coloring and `world_to_pixel` logic — duplicated 8×/3×/7×
  in the original — now live once in `nav_explain_core`.
- Fixed the `e_hausdorff` bug (a bare `return` returning `None`) and removed debug
  `print`s in the metrics.
- Removed hardcoded `~/amar_ws/...` paths, wildcard imports and dead/commented code;
  added type hints, docstrings and tests.
- The TEB black box (a lost C++ node driven by per-perturbation CSV files) is
  replaced by a deterministic, in-process elastic-band planner.

## The deterministic-planner caveat
The elastic-band planner reproduces TEB's behaviour **qualitatively** — the correct
*sign* of each obstacle's effect on deviation, hence the correct colours, rules and
suggestions — not TEB's exact numbers. Swap in a real Nav2/TEB controller via the
`PlannerBackend` seam for numeric fidelity (`nav_explain_sim`).
