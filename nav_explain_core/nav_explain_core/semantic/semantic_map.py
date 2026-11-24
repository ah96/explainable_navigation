"""Rasterise the ontology into a semantic occupancy map.

Each object's axis-aligned bounding box is filled with its integer ID (0 = free
space). The map can then be treated as an obstacle costmap by the local planner,
and as a label map for colouring per-object explanations. Mirrors
``hixron_subscriber.py::create_global_semantic_map`` and its inflation step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..costmap import CostmapInfo
from .ontology import ObjectEntry, Ontology

# A planner-visible obstacle value for occupied semantic cells.
OBSTACLE_VALUE = 100


@dataclass(frozen=True)
class SemanticMapInfo:
    """Geometry of a semantic map, as stored in a scenario's ``map_info.csv``."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int

    @classmethod
    def from_csv_row(cls, row: dict) -> SemanticMapInfo:
        return cls(
            resolution=float(row["map_costmap_resolution"]),
            origin_x=float(row["map_costmap_origin_pos_x"]),
            origin_y=float(row["map_costmap_origin_pos_y"]),
            width=int(row["map_costmap_width"]),
            height=int(row["map_costmap_height"]),
        )

    def to_costmap_info(self) -> CostmapInfo:
        return CostmapInfo(self.resolution, self.origin_x, self.origin_y, self.width, self.height)


def _box_pixels(obj: ObjectEntry, info: SemanticMapInfo) -> tuple[int, int, int, int]:
    hx, hy = obj.half_extents
    col0 = int((obj.cx - hx - info.origin_x) / info.resolution)
    col1 = int((obj.cx + hx - info.origin_x) / info.resolution)
    row0 = int((obj.cy - hy - info.origin_y) / info.resolution)
    row1 = int((obj.cy + hy - info.origin_y) / info.resolution)
    col0, col1 = sorted((col0, col1))
    row0, row1 = sorted((row0, row1))
    return (
        max(0, col0), min(info.width, col1 + 1),
        max(0, row0), min(info.height, row1 + 1),
    )


def build_semantic_map(ontology: Ontology, info: SemanticMapInfo) -> np.ndarray:
    """Integer label map: ``0`` free, ``object.id`` inside each object's box."""
    smap = np.zeros((info.height, info.width), dtype=np.int32)
    for obj in ontology:
        c0, c1, r0, r1 = _box_pixels(obj, info)
        if c1 > c0 and r1 > r0:
            smap[r0:r1, c0:c1] = obj.id
    return smap


def inflate(semantic_map: np.ndarray, info: SemanticMapInfo, inflation_radius: float = 0.275) -> np.ndarray:
    """Grow each object's region by the inflation radius (robot footprint)."""
    pad = int(round(inflation_radius / info.resolution))
    if pad <= 0:
        return semantic_map.copy()
    out = semantic_map.copy()
    # Inflate larger IDs last so a deterministic precedence is kept on overlap.
    for oid in sorted(v for v in np.unique(semantic_map) if v != 0):
        grown = ndimage.binary_dilation(semantic_map == oid, iterations=pad)
        out[grown & (out == 0)] = oid
    return out


def semantic_to_costmap(semantic_map: np.ndarray) -> np.ndarray:
    """Binary obstacle costmap (``OBSTACLE_VALUE`` where any object is present)."""
    cm = np.zeros(semantic_map.shape, dtype=np.int16)
    cm[semantic_map > 0] = OBSTACLE_VALUE
    return cm


def object_centroids(ontology: Ontology) -> dict[int, tuple[float, float]]:
    """World-frame centroid per object id."""
    return {o.id: (o.cx, o.cy) for o in ontology}
