"""A PlannerBackend that defers to a live ROS 2 controller.

This is the seam for querying a real local planner instead of the built-in
elastic band. It is a documented stub: nothing in this repository implements or
runs it. Wiring it up requires a running Nav2 controller server, see the
section on the simulation assets in ``TUTORIAL.md``.

The intended implementation forwards each :class:`PlanRequest` to the controller
(publishing the perturbed costmap and global path, then reading back the
controller's local plan) and converts the returned ``nav_msgs/Path`` into the
``(M, 2)`` array expected by :class:`PlanResult`.
"""

from __future__ import annotations

from typing import Any

from .base import PlanRequest, PlanResult


class Ros2ControllerPlanner:
    """Forward the black-box queries to a live Nav2 controller (not implemented)."""

    def __init__(self, controller_node: Any = None) -> None:
        self.controller_node = controller_node

    def plan(self, request: PlanRequest) -> PlanResult:  # pragma: no cover - stub
        raise NotImplementedError(
            "Ros2ControllerPlanner is a stub: it needs a running Nav2 controller "
            "server and an implementation of this method (see TUTORIAL.md). "
            "Use the default ElasticBandPlanner instead."
        )
