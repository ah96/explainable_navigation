"""Built-in synthetic costmap scenarios.

A robot follows a straight global plan through a doorway; the three scenarios
vary the obstacles. Fully programmatic (no external data): ``doorway`` (clear),
``doorway_blocked`` (a round obstacle in the gap) and ``clutter`` (a box + round
obstacle forcing a detour). Convention matches ``nav_explain_core.costmap``:
increasing row = increasing world y (forward).
"""

from __future__ import annotations

import numpy as np

from nav_explain_core import CostmapInfo, Pose2D, ScenarioInstance

GRID = 160
RES = 0.025


def _base() -> tuple[np.ndarray, CostmapInfo, Pose2D, np.ndarray]:
    """An empty costmap with the robot a quarter of the way up and a straight plan ahead."""
    cm = np.zeros((GRID, GRID), dtype=np.int16)
    robot_col, robot_row = GRID // 2, GRID // 4
    info = CostmapInfo(resolution=RES, origin_x=-robot_col * RES,
                       origin_y=-robot_row * RES, width=GRID, height=GRID)
    start = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)
    ys = np.linspace(0.0, (GRID - robot_row - 4) * RES, 40)
    plan = np.column_stack([np.zeros_like(ys), ys])
    return cm, info, start, plan


def doorway() -> ScenarioInstance:
    """A clear doorway between two walls."""
    cm, info, start, plan = _base()
    cm[70:150, 40:70] = 100
    cm[70:150, 90:120] = 100
    return ScenarioInstance(costmap=cm, costmap_info=info, global_plan=plan, start_pose=start,
                            name="doorway")


def doorway_blocked() -> ScenarioInstance:
    """The doorway, closed by a round obstacle."""
    inst = doorway()
    cm = inst.costmap.copy()
    rr, cc = np.ogrid[:GRID, :GRID]
    cm[(rr - 100) ** 2 + (cc - 80) ** 2 <= 12**2] = 100
    return ScenarioInstance(costmap=cm, costmap_info=inst.costmap_info, global_plan=inst.global_plan,
                            start_pose=inst.start_pose, name="doorway_blocked")


def clutter() -> ScenarioInstance:
    """A wider gap with a box and a round obstacle in it, which force a detour."""
    cm, info, start, plan = _base()
    cm[70:150, 30:55] = 100
    cm[70:150, 105:130] = 100
    cm[95:115, 72:80] = 100
    rr, cc = np.ogrid[:GRID, :GRID]
    cm[(rr - 105) ** 2 + (cc - 92) ** 2 <= 9**2] = 100
    return ScenarioInstance(costmap=cm, costmap_info=info, global_plan=plan, start_pose=start,
                            name="clutter")


SCENARIOS = {"doorway": doorway, "doorway_blocked": doorway_blocked, "clutter": clutter}


def load_scenario(name: str) -> ScenarioInstance:
    """Build the scenario of that name."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'; available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]()
