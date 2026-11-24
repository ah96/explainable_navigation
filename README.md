# explainable_navigation

Methods that explain why a mobile robot leaves its planned path. The robot's local planner is treated as a black box: it receives the local costmap and the global plan, and it returns a local plan. Each method perturbs the scene (it removes obstacles, moves or opens objects), asks the planner again, and reports which part of the scene causes the deviation of the local plan from the global plan. The result is a coloured map and a sentence. A sixth method explains while the robot drives: what is around it, and why its global plan has changed. The repository is a ROS 2 workspace of six packages around one library that needs no ROS. It also holds the data and the recorded results of four publications, with the tools that recompute them.

| Method | Question it answers | Output |
|---|---|---|
| LIME | How much does each obstacle add to or remove from the deviation? | one coefficient per obstacle, a green and red costmap |
| SHAP | The same, with exact game-theoretic attributions. | Shapley values that sum to the deviation caused by all obstacles |
| Anchors | Which obstacles, while present, make sure that the robot deviates? | a minimal rule with its precision |
| Counterfactual | Which obstacles would have to go for the deviation to stop? | a minimal set of obstacles to remove |
| Visuo-textual | Why does the robot deviate, in words and on a semantic map? | one map per object and affordance, and sentences such as "please open the door" |
| HiXRoN | What is around the robot, and which moved object made it change its global plan? | a heat map of the nearby objects and a sentence, as often and as detailed as the robot's extroversion sets |

## Branches

The repository has one branch, `lyrical`, for ROS 2 Lyrical on Ubuntu 26.04. The section "Status" lists what was checked.

The state for ROS 2 Humble is the first commit of the repository (`45ccafd`). It still holds what this branch removed as unused. `git checkout 45ccafd` shows it.

## What it produces

`neb_demo` runs every method on the built-in scenarios and prints:

```
[doorway_blocked/lime] segment 3 increases deviation; segment 1 decreases deviation; segment 6 increases deviation
[doorway_blocked/shap] segment 3: +1.69; segment 4: +0.52; segment 8: +0.37
[doorway_blocked/anchors] anchor = segments [1] (precision 0.91, coverage 0.50): while these obstacles are present, the robot deviates
[doorway_blocked/counterfactual] remove segments [1, 3, 4, 5, 6, 7, 8] -> deviation 0.71 (from 4.06); target met
[cabinet_room] 5 features, deviation=10.819, runtime=0.113s
  -> I deviate because of the door right/front of me.
  -> Dear human, please open the door right/front of me.
[library] extroversion 1.0: 143 visual, 122 textual, 2.59 objects, 5.98 words, 2 of 2 deviations and failures explained
  tick 19 (deviation): The chair was moved.
  tick 54 (deviation): The chair was moved.
ICSR 2023 experiments, machine 'laptop':
  mean runtime: visual 348.90 ms, textual 0.33 ms, together 349.24 ms
```

It also writes one figure per method, a comparison of the four costmap methods, and the numbers as CSV and JSON.

## Quick start

In a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone -b lyrical https://github.com/ah96/explainable_navigation.git
cd ~/ros2_ws
source /opt/ros/lyrical/setup.bash
rosdep install --from-paths src --ignore-src -y
colcon build
source install/setup.bash
ros2 run nav_explain_bringup neb_demo --out outputs/demo
```

`rosdep` installs NumPy, SciPy, scikit-learn, scikit-image, Matplotlib and OpenCV as Ubuntu packages. They have to be system packages, because the ROS 2 nodes run in the system Python, where `rclpy` is installed.

Without ROS, from a checkout of this repository:

```bash
sudo apt install python3-numpy python3-scipy python3-sklearn python3-skimage python3-matplotlib python3-pytest
python3 -m pytest
export PYTHONPATH=$PWD/nav_explain_core:$PWD/nav_explain_costmap:$PWD/nav_explain_semantic:$PWD/nav_explain_hixron:$PWD/nav_explain_bringup
python3 -m nav_explain_bringup.apps.demo --out outputs/demo
```

In a virtual environment, `pip install -r requirements-dev.txt` replaces the `apt` line.

## Main commands

With ROS, a command is `ros2 run <package> <command>`. Without ROS it is `python3 -m <module>`. Times are from a laptop, and those marked with * from a server (Xeon E5-2660 v3), without ROS.

| Command | Package, module | What it does | Time |
|---|---|---|---|
| `neb_demo` | `nav_explain_bringup`, `nav_explain_bringup.apps.demo` | every method on the built-in scenarios, and the recorded results | 12 s * |
| `nec_explain --scenario clutter --method all` | `nav_explain_costmap`, `nav_explain_costmap.apps.explain_scenario` | one costmap method or all four, with figures | 4 s |
| `nec_figures` | `nav_explain_costmap`, `nav_explain_costmap.apps.reproduce_figures` | quality, `robustness` and runtime of LIME | 3 s |
| `nes_explain` | `nav_explain_semantic`, `nav_explain_semantic.apps.explain_scenario` | the visuo-textual explanation | 2 s |
| `neh_replay --scenario library` | `nav_explain_hixron`, `nav_explain_hixron.apps.replay` | one scripted run of `library` or `office`, with the HiXRoN explanations | 3 s * |
| `neh_replay --sweep` | the same | the run at the eleven extroversion levels, with the three evaluation figures | 4 s * |
| `neh_results` | `nav_explain_hixron`, `nav_explain_hixron.apps.results` | the reported numbers, recomputed from the recorded files | 3 s * |
| `nep_detect --image <file>` | `nav_explain_perception`, `nav_explain_perception.apps.detect_image` | YOLO detections in one image, needs PyTorch | not measured |
| `colcon test` | workspace | the tests of all packages | not measured again |
| `python3 -m pytest` | checkout | the same tests without a build | 19 s * |

## Publications, their data and their results

| Publication | Data in this repository | Recorded results | Recomputed by |
|---|---|---|---|
| Halilovic and Lindner (2023), "Visuo-Textual Explanations of a Robot's Navigational Choices", HRI 2023 Companion | `nav_explain_semantic/data/scenarios/cabinet_room/`: ontology, map geometry, map, simulation world | none | `nes_explain` runs the method on this scenario |
| Halilovic and Krivic (2023), "The Influence of a Robot's Personality on Real-Time Explanations of Its Navigation", ICSR 2023 | `nav_explain_hixron/data/icsr2023/scenario/`: the library | `results/`: one file per extroversion level, from three machines | `neh_results`, and `neh_replay --sweep` repeats the experiment as a scripted run |
| Halilovic and Krivic (2024), "Planning of Explanations for Robot Navigation", ICRA 2024 | `nav_explain_hixron/data/icra2024/scenario/`: the office. `planning/`: the planning domain, four problems, four recorded plans | `statistics/`: the answers of two user studies | `neh_results` |
| Halilovic, Chandrayan and Krivic (2024), "Exploring the Impact of Explanation Representation on User Satisfaction in Robot Navigation", TAHRI 2024 | the office of ICRA 2024. Its files were identical and are kept once. | none that match the publication, see `TUTORIAL.md` | `neh_replay --scenario office` runs the scenario |

`neh_results` gives the reported runtimes of ICSR 2023 (348.90 ms and 349.24 ms) and the reported means and the t-test of ICRA 2024 (4.24, 4.62, t = -5.60). Five reported values differ from the files: three in the last digit, one in another digit, and the shares of female and male participants are exchanged. `TUTORIAL.md`, section 13, lists every value side by side.

The scripted run stands in for the simulation of the original work. It repeats the procedure of the experiment and shows each effect in the same direction. It does not give the recorded numbers, which depend on the simulator and on the machine.

## Adding a simulation

The repository ships no simulation. A simulation is a package of your choice that you add to the workspace. The live nodes attach to it through topics, so nothing in this repository has to be changed and no package here depends on it.

Which simulation packages are accepted:

| Node | The simulation has to provide | Fits without arguments |
|---|---|---|
| costmap node | a navigation stack that publishes the local costmap (`nav_msgs/OccupancyGrid`, cells of value 99 or 100 are obstacles), the global plan (`nav_msgs/Path`) and the odometry (`nav_msgs/Odometry`) | Nav2, which publishes them as `/local_costmap/costmap`, `/plan` and `/odom` |
| semantic node | a world built like a built-in scenario, the odometry, and the TF transform from the odometry frame to `map`. A navigation stack is optional. | a world with the objects of `nav_explain_semantic/data/scenarios/cabinet_room/ontology.csv` |
| perception node | a camera that publishes `sensor_msgs/Image` in `rgb8` or `bgr8` | a camera on `/camera/image_raw` |
| HiXRoN node | a world built like `library` or `office`, the global plan, the odometry, the TF transform to `map`, and a message for every object that is moved | a navigation stack that publishes `/plan` and `/odom` |

- The simulator itself does not matter, since the nodes see only the topics: Gazebo, another physics simulator, or a simulator without physics.
- The three inputs may be in different frames, as in Nav2 (plan in `map`, local costmap in `odom`). The nodes transform them with TF, so the simulation has to publish TF between these frames.
- Not accepted: a stack that publishes its costmap only as `nav2_msgs/Costmap`, a robot that does not move in a plane, and, on ROS 2 Lyrical, a package that needs Gazebo Classic, which is not in the Lyrical package repository.

How to add one:

1. Clone the simulation package into `src/` of the workspace next to this repository, or install it with `apt`. For ROS 2 Lyrical, the package repository offers for example `nav2_bringup` (its `tb3_simulation_launch.py` starts Gazebo and Nav2 with a TurtleBot3) and `turtlebot3_gazebo`.
2. Run `rosdep install --from-paths src --ignore-src -y` and `colcon build` again.
3. Start the simulation and its navigation stack, and send a goal, so that a global plan is published.
4. Start a node of this repository, with the simulated clock:

```bash
ros2 launch nav_explain_costmap costmap_explainer.launch.py use_sim_time:=true method:=shap
ros2 launch nav_explain_semantic semantic_explainer.launch.py use_sim_time:=true
ros2 run nav_explain_perception nav_explain_perception_node --ros-args -p image_topic:=/camera/image_raw
ros2 launch nav_explain_hixron hixron_explainer.launch.py use_sim_time:=true extroversion:=0.7
```

If the simulation uses other topic names, pass `costmap_topic:=`, `global_plan_topic:=` and `odom_topic:=` to the launch file. Each launch file also starts RViz, unless `rviz:=false` is given. The explanation appears on `/nav_explain/text` and, as a coloured layer, in RViz. All topics and parameters are listed in `TUTORIAL.md`.

The explanation is computed with the built-in planner on the costmap and the plan that the simulation publishes. The controller of the simulation is not queried.

## What is here

```
nav_explain_core/        the library, no ROS: planner, features, perturbation dataset,
                         LIME, SHAP, Anchors, counterfactual, semantic layer
nav_explain_costmap/     costmap scenarios, figures, evaluation, live node, launch, RViz
nav_explain_semantic/    semantic scenario with its data, figures, live node, launch, RViz
nav_explain_hixron/      library and office scenarios with the data and the recorded results of the
                         publications, scripted run, recomputed results, live node, launch
nav_explain_perception/  YOLO detector and its node
nav_explain_bringup/     the demo that runs everything
TUTORIAL.md              background, methods, design choices, code tour, nodes, open work
pytest.ini               lets the tests run from a checkout
requirements.txt         the Python packages, for an environment without ROS
```

## Ground rules

- `nav_explain_core` imports no ROS, so that every method runs and is tested without a robot.
- The built-in planner is deterministic and uses no random number, so that the same scene always gives the same explanation.
- For the costmap methods, only `build_perturbation_dataset` queries the planner. Methods that are compared share one dataset, since the planner queries are the expensive part.
- The command-line tools write to the folder given with `--out`, by default `outputs/`, which git ignores.
- The repository holds only code that a command-line tool, a node or a launch file uses.
- The data of the publications is kept as it was recorded, also where no tool reads it: simulation worlds, maps and planning files. Recorded results are never edited. A number reported from them is recomputed by `neh_results`, and a test compares it with the publication.
- The study files hold no nationality, country, education, profession or job title of a participant, since the repository is public.

## Status

Checked on 2026-11-24 with ROS 2 Lyrical on Ubuntu 26.04 (Python 3.14, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, scikit-image 0.26, Matplotlib 3.10), before `nav_explain_hixron` was added:

- `colcon build` finishes without warnings, and `colcon test` passes with 57 tests.
- The tests and all command-line tools except `nep_detect` run with and without a ROS environment.
- The three nodes start and stop cleanly. Fed with a built-in scenario as test data, the costmap node publishes its three topics for each of the four methods, also with the plan in another frame than the costmap and with a costmap without obstacles. The semantic node publishes its three topics. Both launch files pass their arguments to the node.

Checked on 2026-11-24 without ROS, on Ubuntu 24.04 (Python 3.12, NumPy 2.4, SciPy 1.17, scikit-learn 1.9, scikit-image 0.26, Matplotlib 3.10), with `nav_explain_hixron`:

- `python3 -m pytest` passes with 92 tests.
- `neb_demo`, `neh_replay` on both scenarios and as a sweep, and `neh_results` run and write their files.

Not checked:

- **`nav_explain_hixron` has not been built or run with ROS 2.** `colcon build`, `colcon test`, its node and its launch file are still to be checked.
- **No simulation and no navigation stack was run with this branch.** The section "Adding a simulation" follows from the interfaces of the nodes and from the defaults of Nav2 as packaged for ROS 2 Lyrical.
- `rosdep install` and the `pip` route. The Python packages of the check were the Ubuntu packages that `rosdep` names.
- The RViz configurations on a display.
- The YOLO detector. PyTorch was not installed, so only the code around the model is tested.
- The simulation worlds of the publications. They are for Gazebo Classic, which ROS 2 Lyrical does not offer.
- The planning files. The recorded plans were not computed again, since that needs a temporal planner. A test checks that every action of a recorded plan is an action of the domain.

Not done:

- The built-in elastic-band planner imitates the TEB planner of the original work qualitatively: the sign of the effect of each obstacle, not the numbers.
- The scripted run of `nav_explain_hixron` repeats the procedure of the recorded experiments, not their numbers.
- The tests cover the library and the scenarios. They do not start a node and do not look at a figure.

## License

MIT, see [LICENSE](LICENSE).
