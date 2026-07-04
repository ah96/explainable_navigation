"""Optional rclpy node: subscribe to an image topic, publish YOLO detections as text."""

from __future__ import annotations

from ._ros_guard import HAVE_RCLPY, unavailable_reason

if HAVE_RCLPY:  # pragma: no cover - requires ROS2 (+ torch for real detections)
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from ..detector import YoloDetector, torch_available

    class PerceptionNode(Node):
        def __init__(self):
            super().__init__("nav_explain_perception")
            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("model", "yolov5s")
            self.detector = YoloDetector(self.get_parameter("model").value) if torch_available() else None
            self.pub = self.create_publisher(String, "/nav_explain/detections", 1)
            self.create_subscription(Image, self.get_parameter("image_topic").value, self._on_image, 1)
            if self.detector is None:
                self.get_logger().warn("torch unavailable; node will not produce detections.")

        def _on_image(self, msg):
            if self.detector is None:
                return
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            rgb = arr[..., :3] if msg.encoding.startswith("rgb") else arr[..., 2::-1]
            dets = self.detector.detect(rgb)
            self.pub.publish(String(data="; ".join(f"{d.label}:{d.confidence:.2f}" for d in dets)))


def main(args=None):
    if not HAVE_RCLPY:
        print("nav_explain_perception node unavailable:", unavailable_reason())
        return 0
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
