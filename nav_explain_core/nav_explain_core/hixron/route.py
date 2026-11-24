"""A global route on the semantic map.

The replay of a scenario needs the global plan a navigation stack would
compute. Here it is the shortest path on the grid of the semantic map, with
every cell nearer to an object than the robot's radius closed. A path on a grid
runs only along the eight directions of its cells, so it is then pulled
straight wherever the straight line stays in open cells. The search is
deterministic, so the same world always gives the same route.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

from ..semantic.semantic_map import SemanticMapInfo

#: radius by which objects are grown, the inflation radius of the original set-up, in metres
INFLATION_RADIUS_M = 0.275
#: spacing of the points of a returned route, in metres
ROUTE_SPACING_M = 0.05


def _cell(xy: tuple[float, float], info: SemanticMapInfo) -> tuple[int, int]:
    col = int((xy[0] - info.origin_x) / info.resolution)
    row = int((xy[1] - info.origin_y) / info.resolution)
    return min(max(row, 0), info.height - 1), min(max(col, 0), info.width - 1)


def closed_cells(label_map: np.ndarray, info: SemanticMapInfo,
                 inflation_radius: float = INFLATION_RADIUS_M) -> np.ndarray:
    """Boolean map of the cells the robot's centre cannot occupy.

    A cell is closed if its centre is nearer to the centre of an occupied cell
    than the inflation radius plus one cell. The extra cell is there because the
    box of an object covers its cells only up to the resolution of the grid.
    Without it, a gap as wide as the robot stays open by one cell.
    """
    occupied = label_map > 0
    if not occupied.any():
        return occupied
    distance = ndimage.distance_transform_edt(~occupied) * info.resolution
    return distance <= inflation_radius + info.resolution


def _line_is_open(closed: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """True if every cell on the straight line between two cells is open."""
    n = int(np.ceil(2 * max(abs(b[0] - a[0]), abs(b[1] - a[1])))) + 1
    rows = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return not closed[rows, cols].any()


def pull_straight(cells: np.ndarray, closed: np.ndarray) -> np.ndarray:
    """Drop the cells of a grid path that a straight line through open cells can skip."""
    kept = [0]
    while kept[-1] < len(cells) - 1:
        i = kept[-1]
        j = len(cells) - 1
        while j > i + 1 and not _line_is_open(closed, cells[i], cells[j]):
            j -= 1
        kept.append(j)
    return cells[kept]


def resample(points: np.ndarray, spacing: float = ROUTE_SPACING_M) -> np.ndarray:
    """Points every ``spacing`` metres along the polyline through ``points``."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], spacing)
    t = np.append(t, s[-1])
    return np.column_stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])])


def plan_route(label_map: np.ndarray, info: SemanticMapInfo, start_xy: tuple[float, float],
               goal_xy: tuple[float, float], inflation_radius: float = INFLATION_RADIUS_M,
               ) -> np.ndarray | None:
    """Shortest route from start to goal as ``(N, 2)`` points, or ``None`` if there is none.

    The cells around the start and around the goal that are closed only by the
    inflation are opened: the robot stands at the start, and a goal may lie
    right in front of an object. Cells inside an object stay closed.
    """
    closed = closed_cells(label_map, info, inflation_radius)
    start, goal = _cell(start_xy, info), _cell(goal_xy, info)
    rows, cols = np.ogrid[:info.height, :info.width]
    reach = int(np.ceil(inflation_radius / info.resolution)) + 1
    for end in (start, goal):
        if closed[end]:
            near = (rows - end[0]) ** 2 + (cols - end[1]) ** 2 <= reach**2
            closed = closed & ~(near & (label_map == 0))
    if closed[start] or closed[goal]:
        return None
    costs = np.where(closed, -1.0, 1.0)
    try:
        cells, _ = route_through_array(costs, start, goal, fully_connected=True, geometric=True)
    except ValueError:  # no path through the open cells
        return None
    cells = pull_straight(np.asarray(cells, dtype=int), closed).astype(float)
    xy = np.column_stack([info.origin_x + (cells[:, 1] + 0.5) * info.resolution,
                          info.origin_y + (cells[:, 0] + 0.5) * info.resolution])
    xy[0], xy[-1] = start_xy, goal_xy
    return resample(xy)
