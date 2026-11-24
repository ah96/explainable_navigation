"""Live node: subscribe to an image topic, publish the YOLO detections as text.

Publishes ``/nav_explain/detections`` (``std_msgs/String``), one line of
``label:confidence`` pairs per image. PyTorch is optional. Without it the node
starts, warns once and publishes nothing, so a launch file that includes it
still comes up.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ..detector import DEFAULT_MODEL, YoloDetector, torch_available


class PerceptionNode(Node):
    """Runs the detector on every image of the camera topic."""

    def __init__(self) -> None:
        super().__init__("nav_explain_perception")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("model", DEFAULT_MODEL)
        self.detector = YoloDetector(self.get_parameter("model").value) if torch_available() else None
        self.pub = self.create_publisher(String, "/nav_explain/detections", 1)
        self.create_subscription(Image, self.get_parameter("image_topic").value, self._on_image, 1)
        if self.detector is None:
            self.get_logger().warning("torch unavailable; node will not produce detections.")

    def _on_image(self, msg: Image) -> None:
        if self.detector is None:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        rgb = arr[..., :3] if msg.encoding.startswith("rgb") else arr[..., 2::-1]
        dets = self.detector.detect(rgb)
        self.pub.publish(String(data="; ".join(f"{d.label}:{d.confidence:.2f}" for d in dets)))


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    try:
        rclpy.spin(PerceptionNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
