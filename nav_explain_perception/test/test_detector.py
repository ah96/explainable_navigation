"""Detector wrapper (no network / model download required)."""

from nav_explain_perception import Detection, YoloDetector, torch_available


def test_detection_centroid():
    d = Detection("chair", 0.9, (0.0, 0.0, 2.0, 4.0))
    assert d.centroid == (1.0, 2.0)


def test_detector_constructs_without_loading():
    det = YoloDetector("yolov5s")
    assert det._model is None  # lazy: no download at construction


def test_torch_available_is_bool():
    assert isinstance(torch_available(), bool)
