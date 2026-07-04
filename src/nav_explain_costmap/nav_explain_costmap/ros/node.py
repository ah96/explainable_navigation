"""Optional rclpy node exposing costmap explanations live.

Subscribes to the local costmap, global plan and odometry; on each costmap update
it runs the configured method and publishes the coloured explanation image, a
coloured PointCloud2 layer, and (for LIME/SHAP) the per-segment coefficients.

Imports cleanly without ROS/ML: the node class is only defined when both are
present, and ``main`` reports a clear message otherwise.
"""

from __future__ import annotations

from ._ros_guard import HAVE_ROS, unavailable_reason

if HAVE_ROS:  # pragma: no cover - requires ROS2 + ML
    import struct

    import numpy as np
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from std_msgs.msg import Float32MultiArray, String

    from nav_explain_core import CostmapInfo, Pose2D, ScenarioInstance
    from nav_explain_core.geometry import yaw_from_quaternion
    from nav_explain_core.planner import ElasticBandPlanner

    from ..explain import explain_costmap

    class CostmapExplainerNode(Node):
        def __init__(self):
            super().__init__("nav_explain_costmap")
            self.declare_parameter("method", "lime")
            self.declare_parameter("costmap_topic", "/local_costmap/costmap")
            self.declare_parameter("global_plan_topic", "/plan")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("min_period_s", 1.0)
            self.method = self.get_parameter("method").value
            self.planner = ElasticBandPlanner()
            self._plan = None
            self._pose = Pose2D(0.0, 0.0, 0.0)
            self._last = 0.0
            self.create_subscription(OccupancyGrid, self.get_parameter("costmap_topic").value, self._on_costmap, 1)
            self.create_subscription(Path, self.get_parameter("global_plan_topic").value, self._on_plan, 1)
            self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 1)
            self.pub_img = self.create_publisher(Image, "/nav_explain/image", 1)
            self.pub_layer = self.create_publisher(PointCloud2, "/nav_explain/layer", 1)
            self.pub_text = self.create_publisher(String, "/nav_explain/text", 1)
            self.get_logger().info(f"nav_explain_costmap ready (method={self.method}).")

        def _on_plan(self, msg):
            self._plan = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

        def _on_odom(self, msg):
            p = msg.pose.pose
            self._pose = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation.z, p.orientation.w))

        def _on_costmap(self, msg):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last < float(self.get_parameter("min_period_s").value) or self._plan is None:
                return
            self._last = now
            info = CostmapInfo(msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y,
                               msg.info.width, msg.info.height)
            grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
            inst = ScenarioInstance(costmap=grid, costmap_info=info, global_plan=self._plan,
                                    start_pose=self._pose, name="live")
            expl = explain_costmap(inst, self.method, self.planner)
            self._publish(expl, info, msg.header)

        def _publish(self, expl, info, header):
            img = (expl.image * 255).astype(np.uint8)
            m = Image(); m.header = header; m.height, m.width = img.shape[:2]
            m.encoding = "rgb8"; m.step = 3 * m.width; m.data = img.tobytes()
            self.pub_img.publish(m)
            self.pub_text.publish(String(data=f"[{expl.method}] {expl.text}"))
            self.pub_layer.publish(self._cloud(expl.image, info, header))

        def _cloud(self, rgb, info, header):
            pts = []
            h, w = rgb.shape[:2]
            for r in range(h):
                for c in range(w):
                    R, G, B = (rgb[r, c] * 255).astype(int)
                    if R == 0 and G == 0 and B == 0:
                        continue
                    x = info.origin_x + (c + 0.5) * info.resolution
                    y = info.origin_y + (r + 0.5) * info.resolution
                    packed = struct.unpack("f", struct.pack("I", (int(R) << 16) | (int(G) << 8) | int(B)))[0]
                    pts.append((x, y, 0.0, packed))
            cloud = PointCloud2(); cloud.header = header; cloud.height = 1; cloud.width = len(pts)
            cloud.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                            for n, o in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))]
            cloud.is_bigendian = False; cloud.point_step = 16; cloud.row_step = 16 * len(pts)
            cloud.is_dense = True; cloud.data = b"".join(struct.pack("ffff", *p) for p in pts)
            return cloud


def main(args=None):
    if not HAVE_ROS:
        print("nav_explain_costmap node unavailable:", unavailable_reason())
        print("The headless apps (nec_explain, nec_figures) don't need ROS.")
        return 0
    rclpy.init(args=args)
    node = CostmapExplainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
