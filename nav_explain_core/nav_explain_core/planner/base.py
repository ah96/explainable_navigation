"""The black-box local-planner interface.

LIME treats the local planner as an opaque function ``f(situation) -> local
plan``. We capture that contract in :class:`PlannerBackend` so the explanation
pipeline is agnostic to *which* planner is behind it:

* :class:`~nav_explain_core.planner.elastic_band.ElasticBandPlanner` -- the
  default, pure-Python, fully deterministic backend that runs anywhere.
* :class:`~nav_explain_core.planner.ros2_controller.Ros2ControllerPlanner` -- a
  stub that marks where a live Nav2 controller would be queried instead.

Keeping this seam explicit is what lets the headless reproduction and the full
ROS 2 simulation share one explanation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..costmap import CostmapInfo
from ..geometry import Pose2D, VelLimits


@dataclass(frozen=True)
class PlanRequest:
    """Everything the black-box planner needs to produce one local plan."""

    costmap: np.ndarray  # 2D, {0 free, 99 lethal} convention
    costmap_info: CostmapInfo
    global_plan: np.ndarray  # (N, 2) world coords
    start_pose: Pose2D
    footprint_radius: float = 0.275
    vel_limits: VelLimits = field(default_factory=VelLimits)


@dataclass(frozen=True)
class PlanResult:
    """The planner's output."""

    local_plan: np.ndarray  # (M, 2) world coords; empty => no feasible plan
    meta: dict = field(default_factory=dict)


class PlannerBackend(Protocol):
    """A deterministic local planner queried as the LIME black box."""

    def plan(self, request: PlanRequest) -> PlanResult:  # pragma: no cover - protocol
        ...
