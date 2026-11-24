"""Pack an RGB grid image into the payload of a coloured point cloud.

The live nodes show an explanation in RViz as a ``sensor_msgs/PointCloud2`` with
one point per grid cell. The payload is plain bytes, so it is built here, without
ROS, and can be tested like the rest of the core. A node only wraps the bytes in
the message.

Layout of one point (16 bytes, little-endian): ``x``, ``y``, ``z`` as float32 and
``rgb`` as the packed integer ``0x00RRGGBB`` in the bytes of a float32, which is
the convention RViz reads with its ``RGB8`` colour transformer.
"""

from __future__ import annotations

import numpy as np

from .costmap import CostmapInfo

#: bytes per point: x, y, z and rgb, four bytes each
POINT_STEP = 16
#: ``(name, byte offset)`` of the four float32 fields of a point
POINT_FIELDS = (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))

_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])


def pack_xyzrgb(rgb: np.ndarray, info: CostmapInfo, skip_black: bool = False) -> tuple[bytes, int]:
    """Return ``(payload, n_points)`` for an ``(H, W, 3)`` image with values in [0, 1].

    Each cell becomes a point at its cell centre, at height zero. With
    ``skip_black`` the cells that are exactly black are left out, so that
    segments without a colour do not hide the map below them.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    height, width = rgb.shape[:2]
    channels = (rgb * 255).astype(np.uint32)
    packed = (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]

    rows, cols = np.mgrid[:height, :width]
    points = np.zeros(height * width, dtype=_POINT_DTYPE)
    points["x"] = (info.origin_x + (cols + 0.5) * info.resolution).ravel()
    points["y"] = (info.origin_y + (rows + 0.5) * info.resolution).ravel()
    points["rgb"] = packed.ravel()
    if skip_black:
        points = points[points["rgb"] != 0]
    return points.tobytes(), len(points)
