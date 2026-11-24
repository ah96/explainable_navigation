# explainable_navigation: tutorial

For a reader who is new to the topic. Each section on code names its file.

## Contents

1. The project in five minutes
2. Background from zero
3. Motivation
4. The model: a black box and one number
5. The methods
6. Design choices and why
7. A tour of the code, in reading order
8. Rules the code keeps
9. Running it
10. The live nodes
11. Tests
12. Simulation assets
13. Where the code comes from
14. Open work

## 1. The project in five minutes

A mobile robot follows a global plan, a path from where it is to its goal. A local planner turns the next few metres of that path into the motion the robot executes, and it bends the path where obstacles are in the way. A person who watches the robot sees it leave the straight path and does not see why.

This repository answers the question "why does the robot deviate here?" in five ways. Four methods work on the local costmap, the grid of obstacles around the robot: LIME, SHAP, Anchors and a counterfactual. One method, called visuo-textual, works on a semantic map of named objects and answers with a coloured map and a sentence.

All five use the same idea. Change the scene a little, ask the planner again, and see how much the deviation changes.

Run `neb_demo` (section 9) to see every method on the built-in scenarios.

## 2. Background from zero

| Term | Meaning here |
|---|---|
| global plan | the path to the goal, a list of points in metres |
| local plan | the path the local planner returns for the next few metres |
| local costmap | a grid around the robot, each cell free or occupied |
| deviation | how far the local plan lies from the global plan, one number (section 4) |
| black box | the local planner, used only through its input and output |
| interpretable feature | a part of the scene a person can point at: an obstacle segment, or an object with an affordance |
| perturbation | a copy of the scene with some features switched off |
| surrogate | a simple model fitted to the answers of the black box |
| affordance | what can be done with an object: it can be moved, or it can be opened |
| QSR | qualitative spatial reasoning, which turns coordinates into words such as "left/front" |

## 3. Motivation

A deviation can look like an error to a person nearby. An explanation that names the cause ("the door in front of me is closed") makes the behaviour understandable, and a suggestion ("please open the door") lets the person remove the cause. The methods here produce both, without any change to the planner.

## 4. The model: a black box and one number

The planner is a function from a scene to a local plan. The interface is `PlannerBackend` in `nav_explain_core/nav_explain_core/planner/base.py`: a `PlanRequest` holds the costmap, its geometry, the global plan and the robot pose, and a `PlanResult` holds the local plan.

The one number every method explains is the **deviation**: for each point of the local plan, the distance to the nearest point of the global plan, summed over the local plan. It is `deviation` in `nav_explain_core/nav_explain_core/trajectory.py`. It is 0 when the robot follows its plan, and it grows when the robot detours.

To explain a costmap, the code prepares a **perturbation dataset** (`nav_explain_core/nav_explain_core/perturbation_dataset.py`):

1. The costmap is reduced to obstacle or free. Cells of value 99 or 100 are obstacles (`normalize_costmap` in `costmap.py`).
2. The obstacles are cut into segments, by default 8 (`segmentation.py`). Free space is one more segment, with label 0.
3. Perturbations are built (`perturbation.py`). Each is a row of ones and zeros, one entry per segment. A zero removes that segment from the costmap.
4. The planner is asked once per perturbation, and the deviation is stored.

There are two sets of perturbations. `one_off` removes one segment at a time, which takes 9 planner queries for 8 segments. `full_2n` contains every combination, which takes 256 queries for 8 segments.

## 5. The methods

### LIME (`nav_explain_core/nav_explain_core/lime/`)

LIME fits a linear surrogate to the dataset: a Ridge regression in which perturbations that are close to the real scene count more (`kernel.py`, `ridge_explainer.py`). Each segment receives a coefficient. A positive coefficient means that the segment increases the deviation, and the segment is drawn green. A negative coefficient means that it decreases the deviation, and the segment is drawn red. The colour is stronger for a larger coefficient (`coloring.py`). LIME is fast, since the `one_off` set is enough for it.

### SHAP (`nav_explain_core/nav_explain_core/shap.py`)

The Shapley value of a segment is its average contribution to the deviation over all subsets of the other segments. Since the `full_2n` set holds the deviation of every subset, the values are computed exactly, without sampling. They sum to the deviation with all obstacles minus the deviation with none.

### Anchors (`nav_explain_core/nav_explain_core/anchors/`)

An anchor is a small set of segments that fixes the decision. The decision is "the robot deviates", which holds for a perturbation when its deviation is at least half of the real one. The search (`base.py`) looks for the smallest set of segments such that, among the perturbations in which all of them are present, at least 90 percent deviate. That share is the precision. The result reads as a rule: while these obstacles are present, the robot deviates. The precision is computed exactly from the dataset, so the search uses no random sampling.

### Counterfactual (`nav_explain_core/nav_explain_core/counterfactual.py`)

The counterfactual is the smallest set of segments whose removal brings the deviation below a quarter of the real one. It reads as: if these obstacles were gone, the robot would follow its plan. If no set reaches the target, the function returns the set with the lowest deviation and marks the target as not reached.

### Visuo-textual (`nav_explain_core/nav_explain_core/semantic/`)

Here the scene is a semantic map: a grid in which each cell holds the number of the object that occupies it. The objects come from an ontology, a CSV file with position, size and affordances of each object (`ontology.py`, `semantic_map.py`).

A feature is a pair of object and affordance (`affordance.py`). Switching a feature off applies the affordance:

- movability: the object is moved away, and its cells become free,
- openability of a door: the door opens, and its cells become free,
- openability of another object, for example a cabinet: its open doors take more room, and the object grows by 0.35 m.

LIME then gives one coefficient per feature (`pipeline.py`). The visual explanation is one map per feature with the object coloured by its coefficient. The textual explanation takes the three features with the largest coefficients. QSR (`qsr.py`) states where the object is relative to the robot, and templates (`templater.py`) form two sentences: a descriptive one ("I deviate because of the door right/front of me.") and a suggestive one ("Dear human, please open the door right/front of me.").

### Qualitative spatial reasoning (`nav_explain_core/nav_explain_core/semantic/qsr.py`)

Two calculi are implemented. The single-cross calculus names one of four quadrants around the robot's heading: `left/front`, `right/front`, `left/back`, `right/back`. The ternary calculus combines distance (`close`, `distant`) with direction (`front`, `left`, `back`, `right`), for example `close-front`.

### Trajectory metrics (`nav_explain_core/nav_explain_core/trajectory.py`)

Besides the deviation, the file holds a small library of distances between two paths: SSPD, Hausdorff, discrete Fréchet, DTW, LCSS, EDR, ERP and OWD.

## 6. Design choices and why

- **The core has no ROS.** `nav_explain_core` is plain Python with NumPy, SciPy, scikit-learn and scikit-image. The methods can then be run, tested and studied on any computer, and ROS is needed only to connect them to a robot.
- **One dataset for all costmap methods.** The planner queries take almost all of the time. SHAP, Anchors and the counterfactual need the same `full_2n` set, so they read one `PerturbationDataset`, and comparing them costs one planner pass. LIME uses the `one_off` set, since that keeps a single explanation fast.
- **The planner is behind an interface.** A method knows only `PlannerBackend`. The built-in planner can be replaced by a real controller without a change to any method.
- **A deterministic planner.** `ElasticBandPlanner` (`planner/elastic_band.py`) moves the points of the path under three forces: smoothness, distance from obstacles, and a pull back to the global plan. It runs a fixed number of sweeps and uses no random number. The same scene therefore always gives the same explanation, which a planner with random restarts would not.
- **A fixed number of segments.** The segmentation splits or merges until it has the requested number. The number of planner queries is then known in advance, and explanations of different scenes can be compared.
- **Dependencies as system packages.** A ROS 2 node runs in the system Python, where `rclpy` is installed. Each `package.xml` therefore names the Ubuntu packages of the Python libraries, and `rosdep` installs them next to ROS.
- **Figures without a display.** The figure code uses Matplotlib's `Figure` class and writes files, so the tools run on a server.

## 7. A tour of the code, in reading order

All paths are below `nav_explain_core/nav_explain_core/` unless a package is named.

1. `costmap.py`, `geometry.py`: the grid, the conversion between metres and cells, poses and angles.
2. `trajectory.py`: the deviation.
3. `planner/base.py`, then `planner/elastic_band.py`: the interface and the built-in planner.
4. `segmentation.py`, `perturbation.py`: the features of a costmap and their combinations.
5. `perturbation_dataset.py`: `ScenarioInstance`, the input of every costmap method, and `build_perturbation_dataset`, the only place that queries the planner. `dataset.py` stores a dataset in a file.
6. `lime/`, `shap.py`, `anchors/`, `counterfactual.py`: the four methods, each a function of a dataset.
7. `semantic/`: ontology, semantic map, affordances, QSR, templates, and the visuo-textual pipeline.
8. `pointcloud.py`: the bytes of the coloured point cloud that the nodes publish.
9. `nav_explain_costmap/nav_explain_costmap/`: `scenarios.py` (three built-in costmaps), `explain.py` (one entry point for the four methods), `render.py` and `evaluation.py` (figures), `apps/` (command-line tools), `ros/node.py`.
10. `nav_explain_semantic/nav_explain_semantic/`: `scenarios.py` with the data in `nav_explain_semantic/data/`, `render.py`, `apps/`, `ros/node.py`.
11. `nav_explain_perception/nav_explain_perception/`: `detector.py` and `ros/node.py`.
12. `nav_explain_bringup/nav_explain_bringup/apps/demo.py`: the demo.

## 8. Rules the code keeps

- Only a `ros/` folder imports `rclpy`. A test never needs ROS.
- Row 0 of every perturbation matrix is the real scene, and column 0 is the feature that is never switched off. All methods rely on both.
- A cell index is `grid[row, col]`, where the row counts along y and the column along x. `world_to_pixel` returns `(col, row)`.
- A costmap method does not query the planner itself. It receives a dataset, or it calls `build_perturbation_dataset`. The visuo-textual pipeline has its own loop, since its perturbations are affordances and not segments.
- Numbers that decide a result are named constants or fields of a configuration class, for example `ElasticBandConfig`, not literals in the code.

## 9. Running it

Installation is described in `README.md`. The scenarios are built in:

| Scenario | Kind | Scene |
|---|---|---|
| `doorway` | costmap | a straight plan through a gap between two walls |
| `doorway_blocked` | costmap | the same, with a round obstacle that closes the gap |
| `clutter` | costmap | a wider gap with a box and a round obstacle in it |
| `cabinet_room` | semantic | a room with cabinet, table, chair and door, and a plan through the closed door |

What the tools write, each into `<out>/<scenario>/`:

| Tool | Files |
|---|---|
| `nec_explain` | `<method>.png`, `comparison.png` (for `--method all`), `summary.txt`, `summary.json` |
| `nec_figures` | `evaluation.png` with three panels: quality, `robustness`, runtime |
| `nes_explain` | `explanation_<object>_<affordance>.png`, `explanation_grid.png`, `coefficients.csv`, `explanations.txt`, `timing.json` |
| `neb_demo` | the files of `nec_explain` below `costmap/` and of `nes_explain` below `semantic/` |

In `evaluation.png`, quality is the variance minus the mean of the absolute coefficients, and `robustness` is the share of segment pairs that keep their order of importance when LIME sees only a part of the perturbations.

On a laptop, the planner pass over 256 perturbations takes between one and two seconds.

## 10. The live nodes

**Costmap node** (`nav_explain_costmap/nav_explain_costmap/ros/node.py`). It explains the local costmap of a running navigation stack.

| Parameter | Default | Meaning |
|---|---|---|
| `method` | `lime` | `lime`, `shap`, `anchors` or `counterfactual` |
| `costmap_topic` | `/local_costmap/costmap` | `nav_msgs/OccupancyGrid` |
| `global_plan_topic` | `/plan` | `nav_msgs/Path`, in the frame of the costmap |
| `odom_topic` | `/odom` | `nav_msgs/Odometry` |
| `min_period_s` | `1.0` | shortest time between two explanations, since one explanation takes longer than a costmap update |

It publishes `/nav_explain/image` (`sensor_msgs/Image`), `/nav_explain/layer` (`sensor_msgs/PointCloud2`, one coloured point per cell) and `/nav_explain/text` (`std_msgs/String`).

**Semantic node** (`nav_explain_semantic/nav_explain_semantic/ros/node.py`). It explains the built-in semantic scenario at the pose the odometry reports.

| Parameter | Default | Meaning |
|---|---|---|
| `scenario` | `cabinet_room` | name of a built-in scenario |
| `global_plan_topic` | `/plan` | `nav_msgs/Path` in the `map` frame. Until one arrives, the plan of the scenario is used. |
| `odom_topic` | `/odom` | `nav_msgs/Odometry` |
| `mode` | `one_off` | `one_off` or `full_2n` |
| `min_period_s` | `1.0` | shortest time between two explanations |

It publishes `/nav_explain/semantic_layer` (`sensor_msgs/PointCloud2`), `/nav_explain/semantic_text` (`std_msgs/String`) and `/nav_explain/semantic_markers` (`visualization_msgs/MarkerArray`).

**Perception node** (`nav_explain_perception/nav_explain_perception/ros/node.py`). It reads `sensor_msgs/Image` from the parameter `image_topic` (default `/camera/image_raw`), runs the YOLO model named by the parameter `model` (default `yolov5s`) and publishes `/nav_explain/detections` (`std_msgs/String`). PyTorch is not a declared dependency, since it is large and only this node needs it. Without PyTorch the node starts, warns and publishes nothing. The model is downloaded on first use.

## 11. Tests

The tests are in the `test/` folder of each package and need no ROS. `python3 -m pytest` runs them from a checkout, and `colcon test` runs them in a workspace.

They check, among other things: the planner gives the same plan twice and bends around an obstacle, the segmentation reaches the requested number of segments, the Shapley values sum to the difference they have to explain, an explanation from a shared dataset equals the explanation computed alone and needs no further planner query, each affordance changes the costmap as described, and the door is the cause in `cabinet_room`.

They do not check: the nodes, the content of a figure, and the YOLO detector with a real model.

## 12. Simulation assets

`nav_explain_sim` holds a generic differential-drive robot (`robot/`), two worlds (`worlds/`) and Nav2 parameters with the TEB controller (`config/nav2_params.yaml`). They were written for ROS 2 Humble with Gazebo Classic. Nothing in the repository uses them, and they have never been run from this repository.

They are not ported to ROS 2 Lyrical. On 2026-11-24 the Lyrical package repository contained neither the Gazebo Classic packages (`gazebo_ros_pkgs`) nor the TEB controller (`teb_local_planner`). It does contain Nav2 and the current Gazebo with its ROS bridge (`ros_gz`).

A port needs four things: the worlds and the robot plugins in the format of the current Gazebo, a Nav2 controller in place of TEB, a launch file, and an implementation of `Ros2ControllerPlanner` (`nav_explain_core/nav_explain_core/planner/ros2_controller.py`) that sends each perturbed costmap to that controller and reads back its plan.

## 13. Where the code comes from

The algorithms were ported from an older ROS 1 research code base, which is left as it is. The notes of that port:

- Ported: LIME on the costmap (segmentation, perturbation, weighted Ridge, colouring), the visuo-textual explanation, Anchors, the trajectory metrics, both QSR calculi, and the YOLO detector.
- Added: exact SHAP and the counterfactual, both on the shared dataset.
- Anchors was reimplemented. The original searched with random sampling. Since the dataset holds every perturbation, the precision is computed exactly here.
- The deviation is computed with array operations in place of nested loops.
- The costmap conversion, the colouring and the conversion from metres to cells existed in several copies and now exist once.
- The Hausdorff distance of the original returned no value, which is corrected.
- Paths to one user's home folder, wildcard imports and unused code were removed. Type hints, docstrings and tests were added.
- The original black box was a modified TEB planner in a C++ node, driven through CSV files. That node is not available. `ElasticBandPlanner` takes its place.

The planner is therefore the main difference to the original work. It reproduces the sign of the effect of each obstacle, and with it the colours, rules and suggestions. It does not reproduce the numbers of TEB.

## 14. Open work

- A planner backend that queries a real Nav2 controller, and with it the port of `nav_explain_sim` (section 12).
- In `doorway_blocked` the gap is closed and no path through it exists. The elastic band does not find the way around the walls. Its points scatter around the obstacle, which is visible in the figures. The deviation is large, as it should be, but the local plan shown is not a path a robot would drive.
- The nodes publish strings, images and point clouds. The messages `Explanation` and `ObjectAffordance` of `nav_explain_msgs` are defined and not yet published.
- The semantic node explains a built-in scenario. Building the semantic map from the detections of the perception node is not implemented.
- The nodes have been run with fed test data only, not with a navigation stack.
