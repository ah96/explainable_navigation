"""Capability probing so the ROS layer degrades gracefully.

The headless reproduction needs only the ML stack; the live node needs *both*
rclpy and the ML stack importable in the same interpreter. On this project's
machine they live in different interpreters (rclpy in system Python, the ML libs
in a venv), so the node may be unavailable even though the package builds. We
detect that here and report a clear message rather than crashing.
"""

from __future__ import annotations

HAVE_RCLPY = False
HAVE_ML = False
_IMPORT_ERROR = ""

try:  # pragma: no cover - environment dependent
    import rclpy  # noqa: F401

    HAVE_RCLPY = True
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR += f"rclpy: {exc}; "

try:  # pragma: no cover - environment dependent
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    import skimage  # noqa: F401

    HAVE_ML = True
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR += f"ml-deps: {exc}; "

HAVE_ROS = HAVE_RCLPY and HAVE_ML


def unavailable_reason() -> str:
    parts = []
    if not HAVE_RCLPY:
        parts.append("rclpy not importable (source your ROS2 install)")
    if not HAVE_ML:
        parts.append(
            "ML deps (numpy/scikit-learn/scikit-image) not importable in this "
            "interpreter -- install requirements.txt where rclpy is visible, "
            "e.g. a venv created with --system-site-packages"
        )
    return "; ".join(parts) or "unknown"
