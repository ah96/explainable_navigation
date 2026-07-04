"""Optional rclpy node wrapping the visuo-textual explainer.

Loads the ontology + semantic-map geometry from a scenario directory (parameter
``scenario_dir``); subscribes to the global plan and odometry; on each plan/odom
update it runs :class:`VisuoTextualExplainer` and publishes:

* ``/vtne/semantic_layer``    (``PointCloud2``) -- the coloured explanation map of
  the most deviation-driving feature, in the ``map`` frame;
* ``/vtne/explanation_text``  (``String``)      -- the descriptive + suggestive text;
* ``/vtne/explanation_markers`` (``MarkerArray``) -- the same text as RViz labels.

Imports cleanly without ROS/ML: the node class is only defined when both are
available, and ``main`` reports a clear message otherwise.
"""

from __future__ import annotations

from ._ros_guard import HAVE_ROS, unavailable_reason

if HAVE_ROS:  # pragma: no cover - requires a full ROS2 + ML environment
    import struct

    import numpy as np
    import rclpy
    from geometry_msgs.msg import Point
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray

    from nav_explain_core import Pose2D
    from nav_explain_core.geometry import yaw_from_quaternion
    from nav_explain_core.planner import ElasticBandPlanner
    from nav_explain_core.semantic import VisuoTextualExplainer, VTInstance
    from ..scenarios import load_scenario

    class VisuoTextualExplainerNode(Node):
        def __init__(self):
            super().__init__("visuo_textual_explainer")
            self.declare_parameter("scenario", "cabinet_chair_table")
            self.declare_parameter("global_plan_topic", "/plan")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("mode", "one_off")

            self.instance: VTInstance = load_scenario(self.get_parameter("scenario").value)
            self.explainer = VisuoTextualExplainer(
                ElasticBandPlanner(), mode=self.get_parameter("mode").value)
            self._plan = self.instance.global_plan

            self.create_subscription(Path, self.get_parameter("global_plan_topic").value, self._on_plan, 1)
            self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 1)
            self.pub_layer = self.create_publisher(PointCloud2, "/vtne/semantic_layer", 1)
            self.pub_text = self.create_publisher(String, "/vtne/explanation_text", 1)
            self.pub_markers = self.create_publisher(MarkerArray, "/vtne/explanation_markers", 1)
            self.get_logger().info(
                f"visuo_textual_explainer ready (scenario={self.instance.name}); "
                "publishing on /odom updates.")

        def _on_plan(self, msg):
            if msg.poses:
                self._plan = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

        def _on_odom(self, msg):
            p = msg.pose.pose
            self.instance.robot_pose = Pose2D(
                p.position.x, p.position.y,
                yaw_from_quaternion(p.orientation.z, p.orientation.w))
            self.instance.global_plan = self._plan
            result = self.explainer.explain(self.instance)
            self._publish(result)

        def _publish(self, result):
            if not result.coefficients:
                return
            # top feature -> colour its explanation map into a PointCloud2
            top = max(result.coefficients, key=lambda k: abs(result.coefficients[k]))
            self.pub_layer.publish(self._cloud(result.explanation_maps[top]))

            text = "\n".join(result.descriptive_text + result.suggestive_text)
            self.pub_text.publish(String(data=text))
            self.pub_markers.publish(self._markers(result))

        def _cloud(self, rgb):
            info = self.instance.map_info
            pts = []
            h, w = rgb.shape[:2]
            for row in range(h):
                for col in range(w):
                    r, g, b = (rgb[row, col] * 255).astype(int)
                    x = info.origin_x + (col + 0.5) * info.resolution
                    y = info.origin_y + (row + 0.5) * info.resolution
                    packed = struct.unpack("f", struct.pack("I", (r << 16) | (g << 8) | b))[0]
                    pts.append((x, y, 0.0, packed))
            cloud = PointCloud2()
            cloud.header.frame_id = "map"
            cloud.height = 1
            cloud.width = len(pts)
            cloud.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            cloud.is_bigendian = False
            cloud.point_step = 16
            cloud.row_step = 16 * len(pts)
            cloud.is_dense = True
            cloud.data = b"".join(struct.pack("ffff", *p) for p in pts)
            return cloud

        def _markers(self, result):
            arr = MarkerArray()
            for i, text in enumerate(result.suggestive_text):
                m = Marker()
                m.header.frame_id = "map"
                m.ns = "vtne_text"
                m.id = i
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                m.pose.position = Point(x=self.instance.robot_pose.x,
                                        y=self.instance.robot_pose.y, z=1.0 + 0.3 * i)
                m.scale.z = 0.3
                m.color.a = 1.0
                m.color.r = m.color.g = m.color.b = 1.0
                m.text = text
                arr.markers.append(m)
            return arr


def main(args=None):
    if not HAVE_ROS:
        print("visuo_textual_explainer node unavailable:", unavailable_reason())
        print("The headless apps (vtne_explain_scenario, vtne_reproduce_figures) "
              "do not need ROS and are the supported path.")
        return 0
    rclpy.init(args=args)
    node = VisuoTextualExplainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
