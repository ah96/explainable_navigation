"""Local-costmap representation and coordinate conversions.

This is the single place that owns the costmap value convention and the
world <-> pixel transforms, so the rest of the core never has to reason about
y-flips or origin offsets.

Value convention (mirrors ``lime_rt_sub.py`` in the original repo):

* raw ROS ``OccupancyGrid`` values are 0..100 (and -1 for unknown);
* ``normalize_costmap`` maps lethal cells (100) to 99 and everything <= 98 to 0,
  so the segmentation sees a clean "obstacle (99) vs free (0)" image.

Pixel/grid convention: ``grid[row, col]`` with ``row`` indexing the y axis and
``col`` indexing the x axis, exactly like ``nav_msgs/OccupancyGrid`` data laid
out row-major from the origin. ``world_to_pixel`` therefore returns ``(col, row)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Costmap cell values after normalization.
FREE = 0
LETHAL = 99
# A cell counts as an obstacle (for segmentation and the planner's distance
# field) if it is at least half lethal.
OBSTACLE_LEVEL = LETHAL / 2.0


@dataclass(frozen=True)
class CostmapInfo:
    """Geometry of a costmap grid (matches ``nav_msgs/MapMetaData``)."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: dict) -> CostmapInfo:
        return cls(
            resolution=float(d["resolution"]),
            origin_x=float(d["origin_x"]),
            origin_y=float(d["origin_y"]),
            width=int(d["width"]),
            height=int(d["height"]),
        )

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "width": self.width,
            "height": self.height,
        }


def normalize_costmap(raw: np.ndarray) -> np.ndarray:
    """Map a raw occupancy grid to the {0, 99} obstacle convention.

    100 (lethal) -> 99, anything <= 98 (free / inflated / unknown) -> 0.
    Returns a float array (a copy; the input is not modified).
    """
    cm = np.asarray(raw, dtype=np.float64).copy()
    cm[cm == 100] = LETHAL
    cm[cm <= 98] = FREE
    return cm


def world_to_pixel(x: float, y: float, info: CostmapInfo) -> tuple[int, int]:
    """World (x, y) in metres -> integer (col, row) grid indices."""
    col = int((x - info.origin_x) / info.resolution)
    row = int((y - info.origin_y) / info.resolution)
    return col, row


def world_to_pixels(points: np.ndarray, info: CostmapInfo) -> tuple[np.ndarray, np.ndarray]:
    """Array form of :func:`world_to_pixel`: ``(N, 2)`` points -> ``(cols, rows)``.

    Truncates toward zero exactly like the scalar version, so both agree on
    every point.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cols = ((points[:, 0] - info.origin_x) / info.resolution).astype(np.intp)
    rows = ((points[:, 1] - info.origin_y) / info.resolution).astype(np.intp)
    return cols, rows


def pixel_to_world(col: int, row: int, info: CostmapInfo) -> tuple[float, float]:
    """Integer (col, row) grid indices -> world (x, y) of the cell centre."""
    x = info.origin_x + (col + 0.5) * info.resolution
    y = info.origin_y + (row + 0.5) * info.resolution
    return x, y


def in_bounds(col: int, row: int, info: CostmapInfo) -> bool:
    """Whether the cell ``(col, row)`` lies inside the grid."""
    return 0 <= col < info.width and 0 <= row < info.height
