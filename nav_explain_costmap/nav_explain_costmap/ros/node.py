"""Live node that explains the local costmap of a running navigation stack.

Subscribes to the local costmap, the global plan and the odometry. On a costmap
update it runs the configured method and publishes

* ``/nav_explain/image`` (``sensor_msgs/Image``): the coloured explanation,
* ``/nav_explain/layer`` (``sensor_msgs/PointCloud2``): the same colours as one
  point per costmap cell, in the frame of the costmap, for RViz,
* ``/nav_explain/text`` (``std_msgs/String``): the explanation as one sentence.

An explanation takes longer than a costmap period, so updates are throttled by
the parameter ``min_period_s``.

The three inputs may be in different frames, as they are in Nav2 (plan in
``map``, local costmap and odometry in ``odom``). The plan and the robot pose are
transformed into the frame of the costmap with TF. Planar transforms are assumed.
"""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener

from nav_explain_core import CostmapInfo, Pose2D, ScenarioInstance
from nav_explain_core.geometry import transform_points, transform_pose, yaw_from_quaternion
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.pointcloud import POINT_FIELDS, POINT_STEP, pack_xyzrgb
from nav_explain_core.trajectory import remaining_plan

from ..explain import METHODS, CostmapExplanation, explain_costmap


def xyzrgb_cloud(payload: bytes, n_points: int, header: Header) -> PointCloud2:
    """Wrap a payload of :func:`nav_explain_core.pointcloud.pack_xyzrgb` in a message."""
    cloud = PointCloud2()
    cloud.header = header
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


class CostmapExplainerNode(Node):
    """Explains each (throttled) local costmap with one of the four methods."""

    def __init__(self) -> None:
        super().__init__("nav_explain_costmap")
        self.declare_parameter("method", "lime")
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("global_plan_topic", "/plan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("min_period_s", 1.0)

        self.method = self.get_parameter("method").value
        if self.method not in METHODS:
            raise ValueError(f"parameter 'method' must be one of {METHODS}, got {self.method!r}")
        self.min_period_s = float(self.get_parameter("min_period_s").value)
        self.planner = ElasticBandPlanner()
        self._plan: np.ndarray | None = None
        self._plan_frame = ""
        self._pose: Pose2D | None = None
        self._pose_frame = ""
        self._last = -float("inf")
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)

        self.create_subscription(OccupancyGrid, self.get_parameter("costmap_topic").value, self._on_costmap, 1)
        self.create_subscription(Path, self.get_parameter("global_plan_topic").value, self._on_plan, 1)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 1)
        self.pub_img = self.create_publisher(Image, "/nav_explain/image", 1)
        self.pub_layer = self.create_publisher(PointCloud2, "/nav_explain/layer", 1)
        self.pub_text = self.create_publisher(String, "/nav_explain/text", 1)
        self.get_logger().info(f"nav_explain_costmap ready (method={self.method}).")

    def _on_plan(self, msg: Path) -> None:
        if msg.poses:
            self._plan = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
            self._plan_frame = msg.header.frame_id

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self._pose = Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation.z, p.orientation.w))
        self._pose_frame = msg.header.frame_id

    def _lookup(self, target: str, source: str) -> tuple[float, float, float] | None:
        """Pose ``(x, y, yaw)`` of the frame ``source`` in the frame ``target``, from TF.

        Data without a frame, or already in the target frame, needs no transform.
        ``None`` means that TF does not know the transform (yet).
        """
        if not source or source == target:
            return 0.0, 0.0, 0.0
        try:
            t = self._tf.lookup_transform(target, source, Time()).transform
        except TransformException as exc:
            self.get_logger().warning(f"no transform from '{source}' to '{target}': {exc}",
                                      throttle_duration_sec=5.0)
            return None
        return t.translation.x, t.translation.y, yaw_from_quaternion(t.rotation.z, t.rotation.w)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._plan is None or self._pose is None or now - self._last < self.min_period_s:
            return
        plan_tf = self._lookup(msg.header.frame_id, self._plan_frame)
        pose_tf = self._lookup(msg.header.frame_id, self._pose_frame)
        if plan_tf is None or pose_tf is None:
            return
        self._last = now
        pose = transform_pose(self._pose, *pose_tf)
        plan = remaining_plan(transform_points(self._plan, *plan_tf), pose.x, pose.y)
        info = CostmapInfo(msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y,
                           msg.info.width, msg.info.height)
        grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        inst = ScenarioInstance(costmap=grid, costmap_info=info, global_plan=plan,
                                start_pose=pose, name="live")
        expl = explain_costmap(inst, self.method, self.planner)
        self._publish(expl, info, msg.header)

    def _publish(self, expl: CostmapExplanation, info: CostmapInfo, header: Header) -> None:
        img = (expl.image * 255).astype(np.uint8)
        m = Image()
        m.header = header
        m.height, m.width = img.shape[:2]
        m.encoding = "rgb8"
        m.step = 3 * m.width
        m.data = img.tobytes()
        self.pub_img.publish(m)
        self.pub_text.publish(String(data=f"[{expl.method}] {expl.text}"))
        # black cells are segments without a colour, they are left out of the layer
        payload, n_points = pack_xyzrgb(expl.image, info, skip_black=True)
        self.pub_layer.publish(xyzrgb_cloud(payload, n_points, header))


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    try:
        rclpy.spin(CostmapExplainerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
