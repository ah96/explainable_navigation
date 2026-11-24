"""YOLO object detector (a ROS-free port of the original camera node).

Lazily loads a YOLOv5 model via ``torch.hub`` and returns detections for an
image. Kept framework-agnostic: no ROS, and torch is imported lazily so importing
this module never fails when torch is absent (``torch_available()`` reports it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: the YOLOv5 variant loaded when none is named
DEFAULT_MODEL = "yolov5s"
#: detections below this confidence are dropped
DEFAULT_CONFIDENCE = 0.4


@dataclass(frozen=True)
class Detection:
    """One detected object in an image."""

    #: class name of the detector, e.g. "chair"
    label: str
    #: confidence in [0, 1]
    confidence: float
    #: bounding box in pixels: xmin, ymin, xmax, ymax
    box: tuple[float, float, float, float]

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def torch_available() -> bool:
    """Whether PyTorch can be imported in this interpreter."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class YoloDetector:
    """Thin wrapper around a YOLOv5 model with lazy loading."""

    def __init__(self, model_name: str = DEFAULT_MODEL, confidence: float = DEFAULT_CONFIDENCE) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model = None

    def load(self):  # noqa: ANN201 - a torch.hub model, torch is not imported at module level
        """Load the model on first use and return it."""
        if self._model is None:
            import torch  # lazy; may download weights on first use

            self._model = torch.hub.load("ultralytics/yolov5", self.model_name, verbose=False)
            self._model.conf = self.confidence
        return self._model

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run detection on an RGB image array; returns a list of Detections."""
        model = self.load()
        results = model(image_rgb)
        rows = results.pandas().xyxy[0]
        return [
            Detection(
                label=str(r["name"]),
                confidence=float(r["confidence"]),
                box=(float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])),
            )
            for _, r in rows.iterrows()
        ]
