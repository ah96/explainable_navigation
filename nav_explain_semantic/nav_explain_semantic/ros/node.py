"""Live node that explains a navigational choice over a semantic map.

Loads the ontology and the map geometry of a built-in scenario (parameter
``scenario``) and subscribes to the global plan and the odometry. On an odometry
update it runs :class:`VisuoTextualExplainer` and publishes

* ``/nav_explain/semantic_layer`` (``sensor_msgs/PointCloud2``): the coloured
  explanation map of the feature with the largest coefficient, in the ``map`` frame,
* ``/nav_explain/semantic_text`` (``std_msgs/String``): the descriptive and the
  suggestive sentences,
* ``/nav_explain/semantic_markers`` (``visualization_msgs/MarkerArray``): the
  suggestive sentences as text above the robot, for RViz.

Odometry arrives faster than an explanation is computed, so updates are throttled
by the parameter ``min_period_s``. Until a plan is received on the plan topic the
node uses the plan that the scenario defines.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from nav_explain_core import Pose2D
from nav_explain_core.geometry import yaw_from_quaternion
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.pointcloud import POINT_FIELDS, POINT_STEP, pack_xyzrgb
from nav_explain_core.semantic import VisuoTextualExplainer, VTInstance, VTResult

from ..scenarios import DEFAULT_SCENARIO, load_scenario

#: frame of the semantic map and of everything the node publishes
MAP_FRAME = "map"
#: height of the first text marker above the robot, and the spacing of the next ones (m)
TEXT_HEIGHT, TEXT_SPACING = 1.0, 0.3


class VisuoTextualExplainerNode(Node):
    """Explains the deviation at the current robot pose, as a map layer and as text."""

    def __init__(self) -> None:
        super().__init__("nav_explain_semantic")
        self.declare_parameter("scenario", DEFAULT_SCENARIO)
        self.declare_parameter("global_plan_topic", "/plan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("mode", "one_off")
        self.declare_parameter("min_period_s", 1.0)

        self.instance: VTInstance = load_scenario(self.get_parameter("scenario").value)
        self.explainer = VisuoTextualExplainer(
            ElasticBandPlanner(), mode=self.get_parameter("mode").value)
        self.min_period_s = float(self.get_parameter("min_period_s").value)
        self._last = -float("inf")

        self.create_subscription(Path, self.get_parameter("global_plan_topic").value, self._on_plan, 1)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 1)
        self.pub_layer = self.create_publisher(PointCloud2, "/nav_explain/semantic_layer", 1)
        self.pub_text = self.create_publisher(String, "/nav_explain/semantic_text", 1)
        self.pub_markers = self.create_publisher(MarkerArray, "/nav_explain/semantic_markers", 1)
        self.get_logger().info(f"nav_explain_semantic ready (scenario={self.instance.name}).")

    def _on_plan(self, msg: Path) -> None:
        if msg.poses:
            self.instance.global_plan = np.array(
                [[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    def _on_odom(self, msg: Odometry) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last < self.min_period_s:
            return
        self._last = now
        p = msg.pose.pose
        self.instance.robot_pose = Pose2D(
            p.position.x, p.position.y, yaw_from_quaternion(p.orientation.z, p.orientation.w))
        self._publish(self.explainer.explain(self.instance))

    def _publish(self, result: VTResult) -> None:
        if not result.coefficients:
            return
        top = max(result.coefficients, key=lambda k: abs(result.coefficients[k]))
        self.pub_layer.publish(self._cloud(result.explanation_maps[top]))
        self.pub_text.publish(
            String(data="\n".join(result.descriptive_text + result.suggestive_text)))
        self.pub_markers.publish(self._markers(result))

    def _cloud(self, rgb: np.ndarray) -> PointCloud2:
        payload, n_points = pack_xyzrgb(rgb, self.instance.map_info.to_costmap_info())
        cloud = PointCloud2()
        cloud.header.frame_id = MAP_FRAME
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.height = 1
        cloud.width = n_points
        cloud.fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                        for name, offset in POINT_FIELDS]
        cloud.is_bigendian = False
        cloud.point_step = POINT_STEP
        cloud.row_step = POINT_STEP * n_points
        cloud.is_dense = True
        cloud.data = payload
        return cloud

    def _markers(self, result: VTResult) -> MarkerArray:
        arr = MarkerArray()
        pose = self.instance.robot_pose
        for i, text in enumerate(result.suggestive_text):
            m = Marker()
            m.header.frame_id = MAP_FRAME
            m.ns = "nav_explain_semantic"
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position = Point(x=pose.x, y=pose.y, z=TEXT_HEIGHT + TEXT_SPACING * i)
            m.scale.z = 0.3
            m.color.a = 1.0
            m.color.r = m.color.g = m.color.b = 1.0
            m.text = text
            arr.markers.append(m)
        return arr


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    try:
        rclpy.spin(VisuoTextualExplainerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
