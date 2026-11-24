# explainable_navigation

Methods that explain why a mobile robot leaves its planned path. The robot's local planner is treated as a black box: it receives the local costmap and the global plan, and it returns a local plan. Each method perturbs the scene (it removes obstacles, moves or opens objects), asks the planner again, and reports which part of the scene causes the deviation of the local plan from the global plan. The result is a coloured map and a sentence. The repository is a ROS 2 workspace of seven packages around one library that needs no ROS.

| Method | Question it answers | Output |
|---|---|---|
| LIME | How much does each obstacle add to or remove from the deviation? | one coefficient per obstacle, a green and red costmap |
| SHAP | The same, with exact game-theoretic attributions. | Shapley values that sum to the deviation caused by all obstacles |
| Anchors | Which obstacles, while present, make sure that the robot deviates? | a minimal rule with its precision |
| Counterfactual | Which obstacles would have to go for the deviation to stop? | a minimal set of obstacles to remove |
| Visuo-textual | Why does the robot deviate, in words and on a semantic map? | one map per object and affordance, and sentences such as "please open the door" |

## Branches

| Branch | For | State |
|---|---|---|
| `lyrical` | ROS 2 Lyrical on Ubuntu 26.04 | this branch, see "Status" for what was checked |
| `humble` | ROS 2 Humble | the earlier state of the repository, kept unchanged, without the changes of this branch |

## What it produces

`neb_demo` runs every method on the built-in scenarios and prints:

```
[doorway_blocked/lime] segment 3 increases deviation; segment 1 decreases deviation; segment 6 increases deviation
[doorway_blocked/shap] segment 3: +1.69; segment 4: +0.52; segment 8: +0.37
[doorway_blocked/anchors] anchor = segments [1] (precision 0.91, coverage 0.50): while these obstacles are present, the robot deviates
[doorway_blocked/counterfactual] remove segments [1, 3, 4, 5, 6, 7, 8] -> deviation 0.71 (from 4.06); target met
[cabinet_room] 5 features, deviation=10.819, runtime=0.111s
  -> I deviate because of the door right/front of me.
  -> Dear human, please open the door right/front of me.
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
export PYTHONPATH=$PWD/nav_explain_core:$PWD/nav_explain_costmap:$PWD/nav_explain_semantic:$PWD/nav_explain_bringup
python3 -m nav_explain_bringup.apps.demo --out outputs/demo
```

In a virtual environment, `pip install -r requirements-dev.txt` replaces the `apt` line.

## Main commands

With ROS, a command is `ros2 run <package> <command>`. Without ROS it is `python3 -m <module>`. Times are from a laptop.

| Command | Package, module | What it does | Time |
|---|---|---|---|
| `neb_demo` | `nav_explain_bringup`, `nav_explain_bringup.apps.demo` | every method on the built-in scenarios | 5 s |
| `nec_explain --scenario clutter --method all` | `nav_explain_costmap`, `nav_explain_costmap.apps.explain_scenario` | one costmap method or all four, with figures | 4 s |
| `nec_figures` | `nav_explain_costmap`, `nav_explain_costmap.apps.reproduce_figures` | quality, `robustness` and runtime of LIME | 3 s |
| `nes_explain` | `nav_explain_semantic`, `nav_explain_semantic.apps.explain_scenario` | the visuo-textual explanation | 2 s |
| `nep_detect --image <file>` | `nav_explain_perception`, `nav_explain_perception.apps.detect_image` | YOLO detections in one image, needs PyTorch | not measured |
| `colcon test` | workspace | the tests of all packages | 12 s |
| `python3 -m pytest` | checkout | the same tests without a build | 9 s |

The live nodes, each with RViz unless `rviz:=false` is given:

| Command | What it does |
|---|---|
| `ros2 launch nav_explain_costmap costmap_explainer.launch.py method:=shap` | explains the local costmap of a running navigation stack |
| `ros2 launch nav_explain_semantic semantic_explainer.launch.py` | explains the built-in semantic scenario at the robot's current pose |
| `ros2 run nav_explain_perception nav_explain_perception_node` | publishes the detections in a camera image as text |

Topics and parameters are listed in `TUTORIAL.md`.

## What is here

```
nav_explain_core/        the library, no ROS: planner, features, perturbation dataset,
                         LIME, SHAP, Anchors, counterfactual, semantic layer
nav_explain_costmap/     costmap scenarios, figures, evaluation, live node, launch, RViz
nav_explain_semantic/    semantic scenario with its data, figures, live node, launch, RViz
nav_explain_perception/  YOLO detector and its node
nav_explain_bringup/     the demo that runs everything
nav_explain_msgs/        two message types
nav_explain_sim/         simulation assets written for ROS 2 Humble, not ported
TUTORIAL.md              background, methods, design choices, code tour, nodes, open work
pytest.ini               lets the tests run from a checkout
requirements.txt         the Python packages, for an environment without ROS
```

## Ground rules

- `nav_explain_core` imports no ROS, so that every method runs and is tested without a robot.
- The built-in planner is deterministic and uses no random number, so that the same scene always gives the same explanation.
- For the costmap methods, only `build_perturbation_dataset` queries the planner. Methods that are compared share one dataset, since the planner queries are the expensive part.
- The command-line tools write to the folder given with `--out`, by default `outputs/`, which git ignores.

## Status

Checked on 2026-11-24 with ROS 2 Lyrical on Ubuntu 26.04 (Python 3.14, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, scikit-image 0.26, Matplotlib 3.10):

- `colcon build` finishes without warnings, and `colcon test` passes with 57 tests.
- The tests and all command-line tools except `nep_detect` run with and without a ROS environment.
- The three nodes start and stop cleanly. Fed with a built-in scenario, the costmap node publishes its three topics for each of the four methods, and the semantic node publishes its three topics. Both launch files start their node with `rviz:=false`.

Not checked:

- `rosdep install` and the `pip` route. The Python packages of the check were the Ubuntu packages that `rosdep` names.
- The RViz configurations on a display, and the nodes with a real navigation stack.
- The YOLO detector. PyTorch was not installed, so only the code around the model is tested.

Not done:

- The built-in elastic-band planner imitates the TEB planner of the original work qualitatively: the sign of the effect of each obstacle, not the numbers.
- `Ros2ControllerPlanner`, the connection to a real Nav2 controller, is a stub.
- `nav_explain_sim` is written for Gazebo Classic and the TEB controller. Neither is in the package repository of ROS 2 Lyrical, and the assets are not ported.
- No node publishes the messages of `nav_explain_msgs` yet. The nodes publish images, point clouds, markers and strings.
- The tests cover the library and the scenarios. They do not start a node and do not look at a figure.

## License

MIT, see [LICENSE](LICENSE).
