"""nav_explain_perception — object detection for semantic explanations.

Wraps a YOLO detector so a live robot can populate the semantic map with detected
objects (feeding the visuo-textual explainer). Torch/YOLO and rclpy are optional:
the module imports cleanly and reports clearly when they are unavailable.
"""

__version__ = "0.1.0"

from .detector import Detection, YoloDetector, torch_available

__all__ = ["Detection", "YoloDetector", "torch_available"]
