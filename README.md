# explainable_navigation

A ROS2 (Humble) workspace of **methods for explaining a mobile robot's local
navigation decisions** — why it deviates from its plan, expressed visually and
textually. It ports and unifies the algorithms from an older ROS1 research
codebase into one clean, method-centric library shared by several ROS2 packages.

Everything runs **headless** (plain Python) for research/reproduction; the rclpy
nodes are an optional live layer that degrades gracefully when ROS/ML aren't both
available.

## The methods

The black box being explained is a deterministic local planner (an elastic-band
stand-in for TEB) whose output is the **deviation** of the local plan from the
global plan. Explanation methods:

| Method | Question it answers | Output |
|---|---|---|
| **LIME** | how much does each obstacle add to / remove from the deviation? | additive coefficients, green/red costmap |
| **SHAP** | same, but with exact game-theoretic attributions | Shapley values (sum to instance−baseline) |
| **Anchors** | which obstacles, while present, *guarantee* the robot deviates? | a minimal high-precision rule |
| **Counterfactual** | which obstacles, if removed, would stop the deviation? | a minimal removal set |
| **Visuo-textual** | why does the robot deviate, in words + a semantic map? | per-(object,affordance) maps + "please open the door …" |

Supporting logic: **Qualitative Spatial Reasoning** (single-cross + ternary
calculi), **trajectory-distance metrics** (SSPD, Hausdorff, Fréchet, DTW, …),
**semantic maps / affordances / ontology**, and **YOLO perception**.

## Packages

```
src/
  nav_explain_core         ROS-free shared library: planner, metrics, the four
                           costmap methods, the semantic/QSR/visuo-textual engine
  nav_explain_costmap      costmap explanations (LIME/SHAP/Anchors/counterfactual)
  nav_explain_semantic     visuo-textual explanations (semantic map + QSR + text)
  nav_explain_perception   YOLO object detection node
  nav_explain_sim          generic diff-drive robot, worlds, Nav2 params (Gazebo scaffold)
  nav_explain_msgs         Explanation / ObjectAffordance messages
  nav_explain_bringup      demos + launches tying it together
```

## Quickstart (headless, no ROS)

```bash
source ~/venvamar/bin/activate            # or any venv
pip install -r requirements.txt
pip install -e src/nav_explain_core

# costmap: run one method or compare all four
PYTHONPATH=src/nav_explain_costmap python -m nav_explain_costmap.apps.explain_scenario \
    --scenario doorway_blocked --method all --out outputs/

# LIME quality/robustness/runtime figures
python -m nav_explain_costmap.apps.reproduce_figures --scenario doorway_blocked --out outputs/figures/

# visuo-textual semantic explanation
PYTHONPATH="src/nav_explain_semantic:src/nav_explain_core" \
  python -m nav_explain_semantic.apps.explain_scenario --scenario cabinet_room --out outputs/

# one-shot showcase of everything
PYTHONPATH="src/nav_explain_core:src/nav_explain_costmap:src/nav_explain_semantic" \
  python -m nav_explain_bringup.apps.demo --out outputs/demo
```

Scenarios are conference-free and built in: `doorway`, `doorway_blocked`,
`clutter` (costmap) and `cabinet_room` (semantic).

## Build with colcon

```bash
# workspace lives under a colcon src/ already
ln -s $(pwd) ~/git/ros2_ws/src/explainable_navigation   # or copy
source /opt/ros/humble/setup.bash
cd ~/git/ros2_ws && colcon build && colcon test
```

## Tests

```bash
pip install -r requirements-dev.txt && pip install -e src/nav_explain_core
pytest src -q
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — workspace + core layout, the shared dataset seam
- [docs/METHODS.md](docs/METHODS.md) — each explanation method explained
- [docs/PORTING_NOTES.md](docs/PORTING_NOTES.md) — what changed from the ROS1 origin + the deterministic-planner caveat

## Notes & limitations

- The deterministic elastic-band planner reproduces TEB's behaviour
  *qualitatively* (the sign of each obstacle's effect on deviation), not its exact
  numbers. A `PlannerBackend` seam allows swapping in a real Nav2/TEB controller
  (`nav_explain_sim`, documented, not run by default).
- Nav2/Gazebo/TEB are optional and not required for the headless methods.

## License

MIT — see [LICENSE](LICENSE).
