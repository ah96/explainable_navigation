"""Launch the costmap explainer node and RViz.

The node needs a navigation stack that publishes the local costmap, the global
plan and the odometry.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("nav_explain_costmap")
    return LaunchDescription([
        DeclareLaunchArgument("method", default_value="lime"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="nav_explain_costmap", executable="nav_explain_costmap_node",
             name="nav_explain_costmap", output="screen",
             parameters=[{"method": LaunchConfiguration("method")}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(pkg, "rviz", "costmap.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz")), output="screen"),
    ])
