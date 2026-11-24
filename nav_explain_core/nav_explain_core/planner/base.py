"""The black-box local-planner interface.

LIME treats the local planner as an opaque function ``f(situation) -> local
plan``. We capture that contract in :class:`PlannerBackend` so the explanation
pipeline is agnostic to *which* planner is behind it:

* :class:`~nav_explain_core.planner.elastic_band.ElasticBandPlanner` -- the
  default, pure-Python, fully deterministic backend that runs anywhere.
Any object with this ``plan`` method can take its place, for example one that
queries the controller of a navigation stack. The explanation methods need the
planner to be deterministic: the same request has to give the same plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..costmap import CostmapInfo
from ..geometry import Pose2D


@dataclass(frozen=True)
class PlanRequest:
    """Everything the black-box planner needs to produce one local plan."""

    costmap: np.ndarray  # 2D, {0 free, 99 lethal} convention
    costmap_info: CostmapInfo
    global_plan: np.ndarray  # (N, 2) world coords
    start_pose: Pose2D
    footprint_radius: float = 0.275


@dataclass(frozen=True)
class PlanResult:
    """The planner's output."""

    local_plan: np.ndarray  # (M, 2) world coords; empty => no plan


class PlannerBackend(Protocol):
    """A deterministic local planner queried as the LIME black box."""

    def plan(self, request: PlanRequest) -> PlanResult:  # pragma: no cover - protocol
        ...
