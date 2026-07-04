# Full-fidelity simulation path (optional)

The headless reproduction (`lpe_explain_scenario`, `lpe_reproduce_figures`) needs
none of this. This directory is for reproducing the **exact** original setup: a
robot in Gazebo navigating with Nav2 + the **TEB** local planner, with LIME
querying that real planner instead of the built-in elastic-band approximation.

## 1. Install the heavy dependencies (not present by default)

```bash
sudo apt update
sudo apt install \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
  ros-humble-teb-local-planner ros-humble-xacro
```

## 2. Assets in this directory

| File | Purpose |
|------|---------|
| `nav2_params.yaml` | Nav2 controller_server using the TEB plugin, derived from the original TIAGo `teb/local_planner.yaml` + costmap configs. |
| `robot/generic_diffdrive.urdf.xacro` + `.gazebo.xacro` | A generic TIAGo-like differential-drive robot (circular footprint r = 0.275 m, planar laser). No PAL/TIAGo packages required. |
| `worlds/doorway.world` | A minimal Gazebo Classic world matching scenario C1. |
| `bringup.launch.py` | Scaffolded bringup: Gazebo + Nav2 + the explainer node with `planner_backend:=ros2_controller`. |

## 3. Bring it up

```bash
colcon build --packages-select nav_explain_sim
source install/setup.bash
# adapt the commented Gazebo/Nav2 includes in bringup.launch.py first
ros2 launch nav_explain_sim/sim/bringup.launch.py
```

## 4. How LIME uses the real planner

With the full stack running, the node is launched with
`planner_backend:=ros2_controller`, which routes each LIME perturbation through
`nav_explain_sim/ros/ros2_controller_planner.py` to the live Nav2/TEB
controller. Implement that backend's `plan()` to publish the perturbed costmap
and global path to the controller and read back its `nav_msgs/Path`. This is the
route to numerically matching the paper's TEB-based results; the built-in
`ElasticBandPlanner` reproduces the behaviour *qualitatively* without any of the
above installs.
