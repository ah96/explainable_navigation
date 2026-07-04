"""A PlannerBackend that defers to a live ROS2 controller (e.g. Nav2 TEB).

This is the seam for *exact* reproduction of the original TEB behaviour. It is a
documented stub: wiring it up requires a running Nav2 controller server (or the
``teb_local_planner`` ROS2 plugin) which is not installed in the default
environment. See ``sim/README_FULLSIM.md``.

The intended implementation forwards each :class:`PlanRequest` to the controller
(publishing the perturbed costmap and global path, then reading back the
controller's local plan) and converts the returned ``nav_msgs/Path`` into the
``(M, 2)`` array expected by :class:`PlanResult`.
"""

from __future__ import annotations

from .base import PlanRequest, PlanResult


class Ros2ControllerPlanner:
    """Forward LIME's black-box queries to a live Nav2/TEB controller."""

    def __init__(self, controller_node=None):
        self.controller_node = controller_node

    def plan(self, request: PlanRequest) -> PlanResult:  # pragma: no cover - stub
        raise NotImplementedError(
            "Ros2ControllerPlanner requires a running Nav2 controller server. "
            "Bring up the full simulation (see sim/README_FULLSIM.md) and wire "
            "this backend to the controller's compute action, or use the default "
            "ElasticBandPlanner for the headless reproduction."
        )
