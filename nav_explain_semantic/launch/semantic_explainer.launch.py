"""Launch the visuo-textual explainer node and RViz.

The node loads its ontology and semantic map from a built-in scenario (argument
``scenario``) and needs only the odometry of a robot, real or simulated, that
moves in a world built like that scenario. With a simulation, pass
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
    pkg = get_package_share_directory("nav_explain_semantic")
    rviz_config = os.path.join(pkg, "rviz", "semantic.rviz")
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="cabinet_room"),
        DeclareLaunchArgument("global_plan_topic", default_value="/plan",
                              description="global plan, nav_msgs/Path"),
        DeclareLaunchArgument("odom_topic", default_value="/odom",
                              description="robot pose, nav_msgs/Odometry"),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="true with a simulation that publishes /clock"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="nav_explain_semantic",
            executable="nav_explain_semantic_node",
            name="nav_explain_semantic",
            output="screen",
            parameters=[{"scenario": LaunchConfiguration("scenario"),
                         "global_plan_topic": LaunchConfiguration("global_plan_topic"),
                         "odom_topic": LaunchConfiguration("odom_topic"),
                         "use_sim_time": use_sim_time}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
