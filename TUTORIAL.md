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
12. Using a simulation
13. The publications: data, recorded results and their reproduction
14. Where the code comes from
15. Open work

## 1. The project in five minutes

A mobile robot follows a global plan, a path from where it is to its goal. A local planner turns the next few metres of that path into the motion the robot executes, and it bends the path where obstacles are in the way. A person who watches the robot sees it leave the straight path and does not see why.

This repository answers the question "why does the robot deviate here?" in five ways. Four methods work on the local costmap, the grid of obstacles around the robot: LIME, SHAP, Anchors and a counterfactual. One method, called visuo-textual, works on a semantic map of named objects and answers with a coloured map and a sentence.

All five use the same idea. Change the scene a little, ask the planner again, and see how much the deviation changes.

A sixth method, HiXRoN, answers a different question, and it answers while the robot drives: what is around me, and which moved object made me change my global plan? How often and in how much detail the robot explains follows from its personality (section 5).

The repository also holds the data and the recorded results of four publications (section 13).

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

The one number every method explains is the **deviation**: for each point of the local plan, the distance to the nearest point of the global plan, summed over the local plan. It is `deviation` in `nav_explain_core/nav_explain_core/trajectory.py`. It is small when the robot follows its plan, and it grows when the robot detours. It is not exactly 0 on the plan, since the distance is taken to the points of the global plan and not to the lines between them.

To explain a costmap, the code prepares a **perturbation dataset** (`nav_explain_core/nav_explain_core/perturbation_dataset.py`):

1. The costmap is reduced to obstacle or free. Cells of value 99 or 100 are obstacles (`normalize_costmap` in `costmap.py`).
2. The obstacles are cut into segments, by default 8 (`segmentation.py`). Free space is one more segment, with label 0. A costmap without obstacles has no segment, and every method then answers that there is nothing to explain.
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

### HiXRoN: explanations that depend on the robot's personality (`nav_explain_core/nav_explain_core/hixron/`)

The methods above explain one moment. This one runs as a loop while the robot drives. It is the explanation loop of the HiXRoN framework of Halilovic and Krivic (2023).

- **Visual** (`layer.py`). A square window around the robot. The objects in it are coloured in five rings, from yellow (near, likely to become an obstacle) to green (far). The object whose move caused a deviation is red, for the next twelve explanations.
- **Textual** (`text.py`). A sentence that lists the objects around the robot, or names the object that was moved. A rich text places each object relative to the robot: "I am passing by table to my left and bookshelf to my right." A poor text only lists: "Around me are table and bookshelf."
- **Deviation** (`deviation.py`). A new global plan counts as a deviation when its middle point lies more than 1 m from the middle point of the previous plan, and an object has moved before.
- **Personality** (`personality.py`). One number, the extroversion between 0 and 1, sets the rest.

| Property | Value | Reason of the publication |
|---|---|---|
| representation threshold, in metres | 3.6 - 2.4 x extroversion | extroverts accept others closer to them. 3.6 m and 1.2 m are the edges of the social zone of Hall's social space. |
| size of the window | the threshold | an extroverted robot gives less detail |
| timing and duration, in passes of the loop | 20 - 20 x extroversion | extroverts pause less and speak shorter |
| detail of the text | rich below 0.5, poor from 0.5 on | extroverts use simpler sentences |
| representation | visual only while a human is nearer than the threshold, visual and textual otherwise | the robot holds back when a human is inside its zone |

One cycle of the loop (`replay.py`, class `Replay`): the robot forms an explanation, waits for the timing, gives it, and keeps it for the duration. A totally extroverted robot forms and gives an explanation on every pass.

### Qualitative spatial reasoning (`nav_explain_core/nav_explain_core/semantic/qsr.py`)

The single-cross calculus names one of four quadrants around the robot's heading: `left/front`, `right/front`, `left/back`, `right/back`.

## 6. Design choices and why

- **The core has no ROS.** `nav_explain_core` is plain Python with NumPy, SciPy, scikit-learn and scikit-image. The methods can then be run, tested and studied on any computer, and ROS is needed only to connect them to a robot.
- **Every costmap method is a function of a dataset.** The planner queries take almost all of the time. SHAP, Anchors and the counterfactual need the same `full_2n` set, so they read one `PerturbationDataset`, and comparing them costs one planner pass. LIME uses the `one_off` set, since that keeps a single explanation fast.
- **The planner is behind an interface.** A method knows only `PlannerBackend`. The built-in planner can be replaced by another one, for example by the controller of a navigation stack, without a change to any method.
- **A deterministic planner.** `ElasticBandPlanner` (`planner/elastic_band.py`) moves the points of the path under three forces: smoothness, distance from obstacles, and a pull back to the global plan. It runs a fixed number of sweeps and uses no random number. The same scene therefore always gives the same explanation, which a planner with random restarts would not.
- **A fixed number of segments.** The segmentation splits or merges until it has the requested number. The number of planner queries is then known in advance, and explanations of different scenes can be compared.
- **Dependencies as system packages.** A ROS 2 node runs in the system Python, where `rclpy` is installed. Each `package.xml` therefore names the Ubuntu packages of the Python libraries, and `rosdep` installs them next to ROS.
- **Figures without a display.** The figure code uses Matplotlib's `Figure` class and writes files, so the tools run on a server.

## 7. A tour of the code, in reading order

All paths are below `nav_explain_core/nav_explain_core/` unless a package is named.

1. `costmap.py`, `geometry.py`: the grid, the conversion between metres and cells, poses, angles and the transform between two frames.
2. `trajectory.py`: the deviation, and the part of a plan that lies ahead of the robot.
3. `planner/base.py`, then `planner/elastic_band.py`: the interface and the built-in planner.
4. `segmentation.py`, `perturbation.py`: the features of a costmap and their combinations.
5. `perturbation_dataset.py`: `ScenarioInstance`, the input of every costmap method, and `build_perturbation_dataset`, the only place that queries the planner.
6. `lime/`, `shap.py`, `anchors/`, `counterfactual.py`: the four methods, each a function of a dataset.
7. `semantic/`: ontology, semantic map, affordances, QSR, templates, and the visuo-textual pipeline, which queries the planner in its own loop.
8. `pointcloud.py`: the bytes of the coloured point cloud that the nodes publish.
9. `nav_explain_costmap/nav_explain_costmap/`: `scenarios.py` (three built-in costmaps), `explain.py` (one entry point for the four methods), `render.py` and `evaluation.py` (figures), `apps/` (command-line tools), `ros/node.py`.
10. `nav_explain_semantic/nav_explain_semantic/`: `scenarios.py` with the data in `nav_explain_semantic/data/`, `render.py`, `apps/`, `ros/node.py`.
11. `hixron/`: `personality.py`, `layer.py`, `text.py` and `deviation.py` (section 5), `route.py` (a global route on the semantic map), then `replay.py` (the loop, and the scripted run that drives it).
12. `nav_explain_hixron/nav_explain_hixron/`: `scenarios.py` (library and office, with the values of the original nodes), `results.py` and `studies.py` (the recorded results of the publications), `render.py`, `apps/`, `ros/node.py`. The data is in `nav_explain_hixron/data/`.
13. `nav_explain_perception/nav_explain_perception/`: `detector.py` and `ros/node.py`.
14. `nav_explain_bringup/nav_explain_bringup/apps/demo.py`: the demo.

## 8. Rules the code keeps

- Only a `ros/` folder imports `rclpy`. A test never needs ROS.
- Row 0 of every perturbation matrix is the real scene, and column 0 is the feature that is never switched off. All methods rely on both.
- A cell index is `grid[row, col]`, where the row counts along y and the column along x. `world_to_pixel` returns `(col, row)`.
- A costmap method does not query the planner. It receives a dataset from `build_perturbation_dataset`. The visuo-textual pipeline has its own loop, since its perturbations are affordances and not segments.
- Code that no command-line tool, node or launch file uses is removed. The `humble` branch holds what was removed.
- Numbers that decide a result are named constants or fields of a configuration class, for example `ElasticBandConfig`, not literals in the code.
- A scripted run uses no clock and no random number. Measured runtimes are written to their own file, so that every other file of a run is the same on every machine.
- A recorded result is read and never written. `results.py` needs only the standard library, so that the reported numbers can be recomputed anywhere.

## 9. Running it

Installation is described in `README.md`. The scenarios are built in:

| Scenario | Kind | Scene |
|---|---|---|
| `doorway` | costmap | a straight plan through a gap between two walls |
| `doorway_blocked` | costmap | the same, with a round obstacle that closes the gap |
| `clutter` | costmap | a wider gap with a box and a round obstacle in it |
| `cabinet_room` | semantic | a room with cabinet, table, chair and door, and a plan through the closed door |
| `library` | HiXRoN | a library. Two chairs are moved into the robot's way, one after the other, and the robot deviates twice. |
| `office` | HiXRoN | an office. A chair is moved into the robot's way, and later a human steps into its way and leaves again. |

What the tools write, each into `<out>/<scenario>/`:

| Tool | Files |
|---|---|
| `nec_explain` | `<method>.png`, `comparison.png` (for `--method all`), `summary.txt`, `summary.json` |
| `nec_figures` | `evaluation.png` with three panels: quality, `robustness`, runtime |
| `nes_explain` | `explanation_<object>_<affordance>.png`, `explanation_grid.png`, `coefficients.csv`, `explanations.txt`, `timing.json` |
| `neh_replay` | `explanations.csv` (one row per explanation), `overview.png` (the plans and the driven path), one `explanation_<cause>_tick<n>.png` per deviation or failure |
| `neh_replay --sweep` | `sweep.csv`, `sweep_runtimes.csv`, `eval_N_exps.png`, `eval_N_artefacts.png`, `eval_runtimes.png` |
| `neh_results` | `icsr2023_<machine>.csv` with its three figures, `icra2024_studies.csv`, `icra2024_study1_answers.png` |
| `neb_demo` | the files of `nec_explain` below `costmap/`, of `nes_explain` below `semantic/`, of `neh_replay` below `hixron_1.0/` and `hixron_0.0/`, and of `neh_results` below `results/` |

In `evaluation.png`, quality is the variance minus the mean of the absolute coefficients, and `robustness` is the share of segment pairs that keep their order of importance when LIME sees only a part of the perturbations.

On a laptop, the planner pass over 256 perturbations takes between one and two seconds.

## 10. The live nodes

The nodes connect the methods to a robot or to a simulation (section 12). All four take their input from topics, so they do not depend on where the data comes from.

**Costmap node** (`nav_explain_costmap/nav_explain_costmap/ros/node.py`). It explains the local costmap of a running navigation stack.

| Parameter | Default | Meaning |
|---|---|---|
| `method` | `lime` | `lime`, `shap`, `anchors` or `counterfactual` |
| `costmap_topic` | `/local_costmap/costmap` | `nav_msgs/OccupancyGrid` |
| `global_plan_topic` | `/plan` | `nav_msgs/Path` |
| `odom_topic` | `/odom` | `nav_msgs/Odometry` |
| `min_period_s` | `1.0` | shortest time between two explanations, since one explanation takes longer than a costmap update |

It publishes `/nav_explain/image` (`sensor_msgs/Image`), `/nav_explain/layer` (`sensor_msgs/PointCloud2`, one coloured point per cell, in the frame of the costmap) and `/nav_explain/text` (`std_msgs/String`).

For each explanation the node transforms the plan and the robot pose into the frame of the costmap with TF, and it drops the part of the plan that lies behind the robot. While TF does not know a needed transform, the node warns and waits.

**Semantic node** (`nav_explain_semantic/nav_explain_semantic/ros/node.py`). It explains the built-in semantic scenario at the pose the odometry reports.

| Parameter | Default | Meaning |
|---|---|---|
| `scenario` | `cabinet_room` | name of a built-in scenario |
| `global_plan_topic` | `/plan` | `nav_msgs/Path`. Until one arrives, the plan of the scenario is used. |
| `odom_topic` | `/odom` | `nav_msgs/Odometry` |
| `mode` | `one_off` | `one_off` or `full_2n` |
| `min_period_s` | `1.0` | shortest time between two explanations |

It publishes `/nav_explain/semantic_layer` (`sensor_msgs/PointCloud2`), `/nav_explain/semantic_text` (`std_msgs/String`) and `/nav_explain/semantic_markers` (`visualization_msgs/MarkerArray`), all in the `map` frame. The plan and the odometry are transformed into the `map` frame with TF.

**Perception node** (`nav_explain_perception/nav_explain_perception/ros/node.py`). It reads `sensor_msgs/Image` from the parameter `image_topic` (default `/camera/image_raw`), runs the YOLO model named by the parameter `model` (default `yolov5s`) and publishes `/nav_explain/detections` (`std_msgs/String`). PyTorch is not a declared dependency, since it is large and only this node needs it. Without PyTorch the node starts, warns and publishes nothing. The model is downloaded on first use.

**HiXRoN node** (`nav_explain_hixron/nav_explain_hixron/ros/node.py`). It runs the explanation loop of section 5 on a built-in scenario.

| Parameter | Default | Meaning |
|---|---|---|
| `scenario` | `library` | `library` or `office` |
| `extroversion` | `1.0` | 0 (introverted) to 1 (extroverted) |
| `global_plan_topic` | `/plan` | `nav_msgs/Path` |
| `odom_topic` | `/odom` | `nav_msgs/Odometry` |
| `tick_period_s` | `0.35` | time of one pass of the loop. The original node took about this long for a pass. |
| `ontology_origin_x`, `ontology_origin_y` | `0.0` | position of the origin of the ontology coordinates in the `map` frame. In the original runs the map frame started at the robot's spawn pose, and the values were 9.0 and -9.0. |

It publishes `/nav_explain/hixron_layer` (`sensor_msgs/PointCloud2`, the coloured objects only) and `/nav_explain/hixron_text` (`std_msgs/String`, empty while a human is nearer than the threshold), in the `map` frame. A moved object is reported to it on `/nav_explain/hixron_object_moved` as a `std_msgs/String` of the form `<gazebo_label> <x> <y>`, the new centre in the `map` frame. The original node read the moves from the simulator. Here the simulation of the user has to publish them, since the node does not depend on a simulator.

The launch files of the costmap and the semantic node take the topic parameters as arguments of the same name, and in addition `use_sim_time` (default `false`) and `rviz` (default `true`). The launch file of the HiXRoN node takes all parameters of its table except `tick_period_s`, and `use_sim_time`. It starts no RViz.

## 11. Tests

The tests are in the `test/` folder of each package and need no ROS. `python3 -m pytest` runs them from a checkout, and `colcon test` runs them in a workspace.

They check, among other things: the planner gives the same plan twice and bends around an obstacle, the segmentation reaches the requested number of segments and gives every obstacle cell a segment, the Shapley values sum to the difference they have to explain, the methods on a shared dataset need no further planner query, a costmap without obstacles gives the answer "nothing to explain", each affordance changes the costmap as described, and the door is the cause in `cabinet_room`.

For HiXRoN they check: the formulas of the personality, the colours of the layer, the sentences, the cycle of the loop, a gap as wide as the robot being closed, the two deviations in `library`, the deviation and the failure in `office`, that a run gives the same result twice, and that more extroverted robots explain more often and with less detail. For the publications they compare the recomputed numbers with the reported ones (section 13).

They do not check: the nodes, the content of a figure, and the YOLO detector with a real model.

## 12. Using a simulation

The repository contains no simulation. A user adds the simulation package of their choice to the workspace, and the nodes attach to it through topics. `README.md` gives the steps. This section gives the reasons.

**What the costmap node needs.** A navigation stack that publishes three topics: the local costmap as `nav_msgs/OccupancyGrid`, the global plan as `nav_msgs/Path`, and the odometry as `nav_msgs/Odometry`. The node treats cells of value 99 or 100 as obstacles and all other cells as free. This is how Nav2 publishes its costmap: 100 for a lethal cell, 99 for a cell in which the robot would touch an obstacle, lower values for the inflation around obstacles. The default topic names of the node are the names Nav2 uses.

**Frames.** Nav2 publishes the plan in the `map` frame and, with its default parameters, the local costmap in the `odom` frame. The nodes therefore transform with TF, and the simulation has to publish TF between these frames, as a localisation does. Only x, y and the rotation about the vertical axis are used.

**Time.** A simulation publishes its own clock. With `use_sim_time:=true` a node follows it, which keeps its TF lookups and its throttling in step with the simulation.

**What the semantic node needs.** It does not read a costmap. It knows the world from the ontology of a built-in scenario, so the simulated world has to contain the same objects at the same places. The positions and sizes are in `nav_explain_semantic/data/scenarios/cabinet_room/ontology.csv`, and the map geometry is in `map_info.csv` next to it. The node needs the odometry and the transform from the odometry frame to `map`.

**What is not accepted, and why.**

- A stack that offers its costmap only as `nav2_msgs/Costmap`. The node reads `nav_msgs/OccupancyGrid`.
- A robot that does not move in a plane. The costmap, the plan and the transforms are two-dimensional.
- On ROS 2 Lyrical, a package built on Gazebo Classic. On 2026-11-24 the Lyrical package repository contained no Gazebo Classic packages (`gazebo_ros_pkgs`). It contained Nav2, the current Gazebo with its ROS bridge (`ros_gz`), `nav2_minimal_tb3_sim`, `nav2_loopback_sim` and `turtlebot3_gazebo`.

**What a simulation does not change.** The explanation is computed with the built-in planner. The node takes the costmap and the plan from the simulation, and it asks `ElasticBandPlanner` what the robot would do in the perturbed scenes. The controller of the simulation is not queried. Querying it needs a class with the `plan` method of `PlannerBackend` (`nav_explain_core/nav_explain_core/planner/base.py`) that sends each perturbed costmap to that controller and reads back its plan. The explanation methods need the same request to give the same plan, which not every controller does.

Where the facts on Nav2 come from: the parameter file of `nav2_bringup` as packaged for ROS 2 Lyrical (frames, topic of the odometry), and the files `nav2_costmap_2d/src/costmap_2d_publisher.cpp` and `nav2_planner/src/planner_server.cpp` of the Nav2 repository, branch `main`, read on 2026-11-24 (cell values, topic names).

No simulation has been run with this repository. The nodes were checked with a built-in scenario published as test data, with the plan in one frame and the costmap in another.

## 13. The publications: data, recorded results and their reproduction

The repository holds what four publications were built on. `README.md` lists the publications and their folders. This section says what each folder holds, how a reported number is recomputed, and where the recomputed number differs from the reported one.

**What is in the folders.**

- `nav_explain_semantic/data/scenarios/cabinet_room/` (HRI 2023): `ontology.csv` and `map_info.csv`, which the scenario `cabinet_room` reads, and the map, the labels of the simulated models and the simulation world of the original run.
- `nav_explain_hixron/data/icsr2023/scenario/` (ICSR 2023): the same files for the library, with two simulation worlds.
- `nav_explain_hixron/data/icsr2023/results/`: the folders `laptop`, `laptop2` and `labpc`, each with the files `0.csv` to `10.csv` for the extroversion levels 0 to 1. A file has one row per pass of the explanation loop: the runtime of the visual and of the textual explanation in seconds, the running numbers of both, the objects shown, the words, and the running number of explained deviations.
- `nav_explain_hixron/data/icra2024/scenario/` (ICRA 2024 and TAHRI 2024): the office, with three simulation worlds.
- `nav_explain_hixron/data/icra2024/planning/`: the planning domain in PDDL, four problems (an introverted and an extroverted robot, each at a deviation and at a failure) and the four plans that were recorded for them, as graphs in the DOT format.
- `nav_explain_hixron/data/icra2024/statistics/`: the answers of the user studies, one file per group and one row per participant. `study1/try2` has the groups `v` (visual), `vt` (visual and textual) and `t` (textual) with 42 participants each. `study2` has the groups `i` (introverted robot) and `e` (extroverted robot) with 10 participants each. `study1/try1` is an earlier collection of the first study with 31, 36 and 31 participants.

The ontology of the library and of the office has five more columns than the one of `cabinet_room`: where the pose of the simulated model lies in the box of the object (`mass_centre`), the side a chair faces, whether it stands under a table, and where its label is drawn.

**Recomputed and reported numbers.** `neh_results` reads the recorded files and prints the numbers below. The tests in `nav_explain_hixron/test/test_recorded_results.py` compare them with the publications.

| Publication | Quantity | Reported | Recomputed |
|---|---|---|---|
| ICSR 2023 | mean runtime of a visual explanation, `laptop` | 348.90 ms | 348.90 ms |
| | mean runtime of a textual explanation | 0.34 ms | 0.33 ms |
| | both together | 349.24 ms | 349.24 ms |
| | deviations explained | two, and one by the totally introverted robot | the same |
| ICRA 2024, representation | participants, age | 84, mean 22.75, SD 4.23 | the same |
| | share of female and of male participants | 60.71 % and 38.10 % | 38.10 % and 60.71 % |
| | group 1 (visual), mean and SD | 4.24 and 0.99 | 4.24 and 1.00 |
| | group 2 (visual and textual), mean and SD | 4.62 and 0.58 | the same |
| | t-test | t = -5.6, p = 3.28e-08 | t = -5.60, p = 3.29e-08 |
| ICRA 2024, extroversion | participants, age, share of male participants | 20, mean 27.45, SD 5.47, 60 % | the same |
| | group 1 (introverted), mean and SD | 3.47 and 1.21 | the same |
| | group 2 (extroverted), mean and SD | 3.77 and 0.77 | 3.77 and 0.87 |

Five reported values differ from the files. Three differ in the last digit: the textual runtime (0.333 ms), the SD of group 1 (0.9989) and the p-value (3.286e-08). The SD of the extroverted group differs in one digit. The shares of female and male participants are exchanged.

The folders `laptop2` and `labpc` are further runs of the same experiment, with mean visual runtimes of 272.20 ms and 467.82 ms. The publication reports `laptop`.

For TAHRI 2024 no reported number can be recomputed. That publication reports 52 participants per group and nine statements. `study1/try1` has nine statements, and 31, 36 and 31 participants. Whether it is an earlier state of that data is not recorded in the files.

**The scripted run.** The experiments of ICSR 2023 ran in a simulator, with a navigation stack for ROS 1. `neh_replay` repeats them without both. The same as in the original run: the ontology and the map geometry, the start pose and the goal, which objects move and where to, the 8 s the robot waits at a failure, and the formulas of section 5. Different:

- The global route is the shortest path on the semantic map (`hixron/route.py`). Cells nearer to an object than 0.275 m, the robot radius of the original set-up, plus one cell are closed. The extra cell is there because the box of an object covers its cells only up to the resolution of the grid. Without it, the gap of 0.55 m that the moved chair leaves in the office stays open by one cell, and the robot does not deviate.
- The robot follows the route exactly, 0.08 m per pass of the loop. With this step the run of the library has 143 passes. The recorded run of the totally extroverted robot has 177.
- In the library a chair moves when the robot is nearer than 0.5 m to a trigger point, as in the original node. For the first chair the radius is 0.8 m. The scripted robot passes that point at 0.66 m, so the original radius would never be met.
- The humans of the library are the three standing models of `icsr_world_1.world`. Which of the two worlds the recorded runs used is not recorded.

The scripted run therefore repeats the procedure and not the numbers. Recorded on `laptop`, the number of visual explanations rises from 3 at extroversion 0 to 177 at extroversion 1, and the mean number of objects falls from 9.33 to 2.27. In the scripted run they go from 4 to 143 and from 14.25 to 2.59. The scripted robot explains both deviations at every level, also at extroversion 0.

**The planning files.** The publication computed the plans with the planner POPF in ROSPlan, for ROS 1. No temporal planner was run for this repository, so the plans are kept as recorded. `dot -Tpdf <plan>.dot` draws one.

**The study files.** Seven columns were removed from every file: nationality, country of residence, the two columns on education, the two on profession, and the job title. They describe a participant in free text, the groups are small, and the repository is public. Age, sex and all answers are kept, since the reported numbers need them.

## 14. Where the code comes from

The algorithms were ported from an older ROS 1 research code base, which is left as it is. The notes of that port:

- Ported: LIME on the costmap (segmentation, perturbation, weighted Ridge, colouring), the visuo-textual explanation, Anchors, the deviation, the single-cross QSR calculus, and the YOLO detector.
- Added: exact SHAP and the counterfactual, both on the shared dataset.
- Anchors was reimplemented. The original searched with random sampling. Since the dataset holds every perturbation, the precision is computed exactly here.
- The deviation is computed with array operations in place of nested loops.
- The costmap conversion, the colouring and the conversion from metres to cells existed in several copies and now exist once.
- Paths to one user's home folder, wildcard imports and unused code were removed. Type hints, docstrings and tests were added.
- The original black box was a modified TEB planner in a C++ node, driven through CSV files. That node is not available. `ElasticBandPlanner` takes its place.

The HiXRoN part was ported from the nodes of the ICSR 2023, ICRA 2024 and TAHRI 2024 experiments. The notes of that port:

- Ported: the personality formulas, the visual layer with its green and yellow colour scheme, the two kinds of text, the detection of a deviation, the cycle of the loop, and the scripts of the two scenarios.
- The colouring of the rings is computed with array operations in place of one slice per ring and colour channel.
- The side of an object follows the heading of the robot: an object straight ahead is "in front". The original named the quarter ahead "right" and the quarter to the left "front".
- The original could not produce the sentence for a moved object in front of the robot, since its condition tested "back" twice. In a rich text it listed objects in front of and behind the robot also as "to my right". Both are corrected.
- Timing and duration are rounded. The original cut off the decimals, which gave 3 in place of 4 passes at extroversion 0.8.
- The red object is counted once among the objects shown. The original counted it twice while it was inside the window.
- The failure sentence is taken from the texts of the original office node. That node showed a list of texts written by hand, each at a fixed time of the run. The other texts of the list are not generated here.
- Not ported: the two other colour schemes, the arrows that show in which direction an object can be moved, the labels and the path markers for RViz, and the dispatching of the plans by ROSPlan.

The planner is therefore the main difference to the original work. It reproduces the sign of the effect of each obstacle, and with it the colours, rules and suggestions. It does not reproduce the numbers of TEB.

Parts of the port that no method uses are not on this branch: the other trajectory distances (SSPD, Hausdorff, Fréchet, DTW, LCSS, EDR, ERP, OWD), the ternary QSR calculus, the two message types, and the simulation assets for Gazebo Classic with the TEB parameters. They are on the `humble` branch.

## 15. Open work

- A planner backend that queries the controller of a running navigation stack (section 12).
- The nodes have been run with test data only. A run with a simulation is the next check.
- In `doorway_blocked` the gap is closed and no path through it exists. The elastic band does not find the way around the walls. Its points scatter around the obstacle, which is visible in the figures. The deviation is large, as it should be, but the local plan shown is not a path a robot would drive.
- The nodes publish strings, images and point clouds, and no typed explanation message. The HiXRoN node also receives a moved object as a string.
- `nav_explain_hixron` has not been built or run with ROS 2 yet.
- The HiXRoN node has no input for a planning failure. Only the scripted run of `office` shows one.
- The recorded plans of ICRA 2024 have not been computed again with a temporal planner.
- The semantic node explains a built-in scenario. Building the semantic map from the detections of the perception node is not implemented.
