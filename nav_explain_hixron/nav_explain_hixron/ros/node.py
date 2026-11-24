"""Live node that explains the robot's surroundings and the changes of its global plan.

Loads the ontology and the map geometry of a built-in scenario (parameter
``scenario``) and runs the explanation loop of a robot with the given
``extroversion``. It subscribes to the global plan, the odometry and a topic
that tells which object has moved, and publishes

* ``/nav_explain/hixron_layer`` (``sensor_msgs/PointCloud2``): the coloured
  objects of the visual explanation, in the ``map`` frame,
* ``/nav_explain/hixron_text`` (``std_msgs/String``): the textual explanation.
  It is empty while a human is nearer than the robot's representation threshold.

The loop makes one pass every ``tick_period_s`` seconds. The original node took
about 0.35 s for a pass, which is the default.

A moved object arrives on ``/nav_explain/hixron_object_moved`` as a
``std_msgs/String`` of the form ``<gazebo_label> <x> <y>``, the new centre in the
``map`` frame. A new global plan that differs enough from the previous one after
such a move is explained as a deviation.

The ontology of a scenario has its own coordinates. ``ontology_origin_x`` and
``ontology_origin_y`` give the position of their origin in the ``map`` frame.
In the original runs the map frame started at the robot's spawn pose, and the
values were 9.0 and -9.0. A plan or an odometry in another frame than ``map`` is
transformed into it with TF. Planar transforms are assumed.
"""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from nav_explain_core import CostmapInfo, Pose2D
from nav_explain_core.geometry import transform_points, transform_pose, yaw_from_quaternion
from nav_explain_core.hixron import Explanation, ExplanationPolicy, Replay
from nav_explain_core.hixron.layer import FREE_GREY, OBJECT_GREY
from nav_explain_core.pointcloud import POINT_FIELDS, POINT_STEP, pack_xyzrgb

from ..scenarios import DEFAULT_SCENARIO, load_scenario

#: frame of everything the node publishes
MAP_FRAME = "map"


class HixronExplainerNode(Node):
    """Explains what is around the robot and why its global plan changed."""

    def __init__(self) -> None:
        super().__init__("nav_explain_hixron")
        self.declare_parameter("scenario", DEFAULT_SCENARIO)
        self.declare_parameter("extroversion", 1.0)
        self.declare_parameter("global_plan_topic", "/plan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("tick_period_s", 0.35)
        self.declare_parameter("ontology_origin_x", 0.0)
        self.declare_parameter("ontology_origin_y", 0.0)

        spec = load_scenario(self.get_parameter("scenario").value)
        self.replay = Replay(spec.ontology, spec.map_info,
                             ExplanationPolicy(float(self.get_parameter("extroversion").value)),
                             spec.humans)
        self._labels = {o.gazebo_label: o.id for o in spec.ontology}
        self._origin = (float(self.get_parameter("ontology_origin_x").value),
                        float(self.get_parameter("ontology_origin_y").value))
        info = spec.map_info
        self._cloud_info = CostmapInfo(info.resolution, info.origin_x + self._origin[0],
                                       info.origin_y + self._origin[1], info.width, info.height)
        self._pose: Pose2D | None = None
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)

        self.create_subscription(Path, self.get_parameter("global_plan_topic").value, self._on_plan, 1)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 1)
        self.create_subscription(String, "/nav_explain/hixron_object_moved", self._on_moved, 10)
        self.pub_layer = self.create_publisher(PointCloud2, "/nav_explain/hixron_layer", 1)
        self.pub_text = self.create_publisher(String, "/nav_explain/hixron_text", 1)
        self.create_timer(float(self.get_parameter("tick_period_s").value), self._on_tick)
        self.get_logger().info(
            f"nav_explain_hixron ready (scenario={spec.name}, "
            f"extroversion={self.replay.policy.extroversion}).")

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

    def _on_plan(self, msg: Path) -> None:
        tf = self._lookup(MAP_FRAME, msg.header.frame_id)
        if not msg.poses or tf is None:
            return
        plan = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.replay.set_plan(transform_points(plan, *tf) - self._origin)

    def _on_odom(self, msg: Odometry) -> None:
        tf = self._lookup(MAP_FRAME, msg.header.frame_id)
        if tf is None:
            return
        p = msg.pose.pose
        pose = transform_pose(
            Pose2D(p.position.x, p.position.y, yaw_from_quaternion(p.orientation.z, p.orientation.w)), *tf)
        self._pose = Pose2D(pose.x - self._origin[0], pose.y - self._origin[1], pose.theta)

    def _on_moved(self, msg: String) -> None:
        parts = msg.data.split()
        if len(parts) != 3 or parts[0] not in self._labels:
            self.get_logger().warning(f"ignored '{msg.data}': expected '<gazebo_label> <x> <y>'")
            return
        try:
            x, y = float(parts[1]), float(parts[2])
        except ValueError:
            self.get_logger().warning(f"ignored '{msg.data}': x and y must be numbers")
            return
        self.replay.move_object(self._labels[parts[0]], (x - self._origin[0], y - self._origin[1]))

    def _on_tick(self) -> None:
        if self._pose is None:
            return
        explanation = self.replay.tick(self._pose)
        if explanation is not None:
            self.pub_layer.publish(self._cloud(explanation))
            self.pub_text.publish(String(data=explanation.text))

    def _cloud(self, explanation: Explanation) -> PointCloud2:
        rgb = explanation.layer.rgb.astype(np.float64) / 255.0
        grey = np.all(explanation.layer.rgb == FREE_GREY, axis=2) | \
            np.all(explanation.layer.rgb == OBJECT_GREY, axis=2)
        rgb[grey] = 0.0  # only the coloured objects are published, so that the map below stays visible
        payload, n_points = pack_xyzrgb(rgb, self._cloud_info, skip_black=True)
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


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    try:
        rclpy.spin(HixronExplainerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
