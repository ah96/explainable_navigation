"""Launch the costmap explainer node and RViz.

The node needs a navigation stack, real or simulated, that publishes the local
costmap, the global plan and the odometry. The topic arguments name them, and
their defaults are the names Nav2 uses. With a simulation, pass
``use_sim_time:=true``, so that the node follows the simulated clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory("nav_explain_costmap")
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument("method", default_value="lime",
                              description="lime, shap, anchors or counterfactual"),
        DeclareLaunchArgument("costmap_topic", default_value="/local_costmap/costmap",
                              description="local costmap, nav_msgs/OccupancyGrid"),
        DeclareLaunchArgument("global_plan_topic", default_value="/plan",
                              description="global plan, nav_msgs/Path"),
        DeclareLaunchArgument("odom_topic", default_value="/odom",
                              description="robot pose, nav_msgs/Odometry"),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="true with a simulation that publishes /clock"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="nav_explain_costmap", executable="nav_explain_costmap_node",
             name="nav_explain_costmap", output="screen",
             parameters=[{"method": LaunchConfiguration("method"),
                          "costmap_topic": LaunchConfiguration("costmap_topic"),
                          "global_plan_topic": LaunchConfiguration("global_plan_topic"),
                          "odom_topic": LaunchConfiguration("odom_topic"),
                          "use_sim_time": use_sim_time}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(pkg, "rviz", "costmap.rviz")],
             parameters=[{"use_sim_time": use_sim_time}],
             condition=IfCondition(LaunchConfiguration("rviz")), output="screen"),
    ])
