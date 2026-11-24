"""The visual explanation: a heat map of the objects around the robot.

The layer is a square window around the robot. Objects in it are coloured by
their distance to the robot in five square rings, from yellow (near, the object
is likely to become an obstacle) to green (far). An object that caused a
deviation is red. Free space and walls keep their grey, so that the layer
shows only what the robot attends to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..semantic.semantic_map import SemanticMapInfo

#: grey of free space and of objects outside the layer
FREE_GREY = 120
OBJECT_GREY = 180
#: colours of the five rings, from the robot outwards (yellow to green)
RING_COLOURS = ((227, 242, 19), (206, 215, 15), (124, 220, 15), (108, 196, 8), (98, 176, 9))
#: colour of the object that caused a deviation
HIGHLIGHT_COLOUR = (255, 0, 0)


@dataclass(frozen=True)
class LayerResult:
    """A visual explanation of one moment."""

    #: ``(H, W, 3)`` image of the whole map, ``uint8``. It is left out when two
    #: results are compared, since an array has no single truth value.
    rgb: np.ndarray = field(compare=False)
    #: ids of the objects inside the window, walls included
    object_ids: tuple[int, ...]
    #: the window in cells: first row, last row, first column, last column
    window: tuple[int, int, int, int]
    #: id of the highlighted object, or ``None``
    highlight_id: int | None


def _ring_index(offset: np.ndarray, extent: int) -> np.ndarray:
    """Ring 0 to 4 of each offset from the window centre, 5 outside the window."""
    ring = np.full(offset.shape, len(RING_COLOURS), dtype=int)
    for j in range(len(RING_COLOURS), 0, -1):
        bound = int(0.1 * j * extent)
        ring[(offset >= -bound) & (offset < bound)] = j - 1
    return ring


def window_cells(info: SemanticMapInfo, robot_xy: tuple[float, float],
                 half_size: float) -> tuple[int, int, int, int]:
    """The window around the robot in cells, clipped to the map."""
    col0 = int((robot_xy[0] - half_size - info.origin_x) / info.resolution)
    col1 = int((robot_xy[0] + half_size - info.origin_x) / info.resolution)
    row0 = int((robot_xy[1] - half_size - info.origin_y) / info.resolution)
    row1 = int((robot_xy[1] + half_size - info.origin_y) / info.resolution)
    return (max(0, row0), min(info.height - 1, row1), max(0, col0), min(info.width - 1, col1))


def explanation_layer(label_map: np.ndarray, info: SemanticMapInfo, robot_xy: tuple[float, float],
                      half_size: float, wall_ids: frozenset[int] = frozenset(),
                      highlight_id: int | None = None) -> LayerResult:
    """Colour the objects in the window around the robot.

    ``wall_ids`` stay grey also inside the window. ``highlight_id`` is coloured
    red wherever it is, since the cause of a deviation is shown even when the
    robot has left it behind.
    """
    row0, row1, col0, col1 = window_cells(info, robot_xy, half_size)
    inside = label_map[row0:row1, col0:col1]
    object_ids = tuple(int(i) for i in np.unique(inside) if i != 0)

    rgb = np.full(label_map.shape + (3,), FREE_GREY, dtype=np.uint8)
    rgb[label_map > 0] = OBJECT_GREY
    centre_row = int(0.5 * (row0 + row1) + 1)
    centre_col = int(0.5 * (col0 + col1) + 1)
    rows, cols = np.indices(label_map.shape)
    ring = np.maximum(_ring_index(rows - centre_row, row1 - row0),
                      _ring_index(cols - centre_col, col1 - col0))
    coloured = (label_map > 0) & ~np.isin(label_map, list(wall_ids))
    for j, colour in enumerate(RING_COLOURS):
        rgb[coloured & (ring == j)] = colour
    if highlight_id is not None:
        rgb[label_map == highlight_id] = HIGHLIGHT_COLOUR
    return LayerResult(rgb=rgb, object_ids=object_ids, window=(row0, row1, col0, col1),
                       highlight_id=highlight_id)
