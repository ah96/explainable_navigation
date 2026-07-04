"""YOLO object detector (a de-ROS'd port of the original camera node).

Lazily loads a YOLOv5 model via ``torch.hub`` and returns detections for an
image. Kept framework-agnostic: no ROS, and torch is imported lazily so importing
this module never fails when torch is absent (``torch_available()`` reports it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class YoloDetector:
    """Thin wrapper around a YOLOv5 model with lazy loading."""

    def __init__(self, model_name: str = "yolov5s", confidence: float = 0.4):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None

    def load(self):
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
        out = []
        for _, r in rows.iterrows():
            out.append(
                Detection(
                    label=str(r["name"]),
                    confidence=float(r["confidence"]),
                    box=(float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])),
                )
            )
        return out
