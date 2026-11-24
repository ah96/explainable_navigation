"""Launch the visuo-textual explainer node and RViz.

The node loads its ontology and semantic map from a built-in scenario (argument
``scenario``) and needs only odometry to run, so no simulator is required.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("nav_explain_semantic")
    rviz_config = os.path.join(pkg, "rviz", "semantic.rviz")

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="cabinet_room"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="nav_explain_semantic",
            executable="nav_explain_semantic_node",
            name="nav_explain_semantic",
            output="screen",
            parameters=[{"scenario": LaunchConfiguration("scenario")}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
