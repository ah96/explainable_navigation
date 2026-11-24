"""Launch the HiXRoN explainer node.

The node loads its ontology and semantic map from a built-in scenario (argument
``scenario``) and needs the global plan and the odometry of a robot, real or
simulated, that moves in a world built like that scenario. With a simulation,
pass ``use_sim_time:=true``, so that the node follows the simulated clock.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def number(name: str) -> ParameterValue:
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="library"),
        DeclareLaunchArgument("extroversion", default_value="1.0",
                              description="0 (introverted) to 1 (extroverted)"),
        DeclareLaunchArgument("global_plan_topic", default_value="/plan",
                              description="global plan, nav_msgs/Path"),
        DeclareLaunchArgument("odom_topic", default_value="/odom",
                              description="robot pose, nav_msgs/Odometry"),
        DeclareLaunchArgument("ontology_origin_x", default_value="0.0",
                              description="origin of the ontology coordinates in the map frame"),
        DeclareLaunchArgument("ontology_origin_y", default_value="0.0"),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="true with a simulation that publishes /clock"),
        Node(
            package="nav_explain_hixron",
            executable="nav_explain_hixron_node",
            name="nav_explain_hixron",
            output="screen",
            parameters=[{"scenario": LaunchConfiguration("scenario"),
                         "extroversion": number("extroversion"),
                         "global_plan_topic": LaunchConfiguration("global_plan_topic"),
                         "odom_topic": LaunchConfiguration("odom_topic"),
                         "ontology_origin_x": number("ontology_origin_x"),
                         "ontology_origin_y": number("ontology_origin_y"),
                         "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"),
                                                        value_type=bool)}],
        ),
    ])
