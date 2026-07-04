"""A deterministic elastic-band local planner.

The original work queried a (forked) TEB local planner as the black box. TEB is
itself a *timed elastic band* optimiser that trades off three things: staying
clear of obstacles, adhering to the global plan, and keeping the path smooth.
Here we reproduce that behaviour with a small, fully deterministic elastic-band
relaxation so the whole explanation pipeline runs with no ROS, no Gazebo and no
TEB installed.

Why deterministic? The relaxation is a pure ``numpy``/``scipy.ndimage`` pipeline
with no RNG, no threads and no wall-clock-dependent branching, run for a fixed
number of iterations. Identical ``PlanRequest`` inputs therefore produce
bit-identical ``local_plan`` outputs (asserted by ``test_planner_determinism``).
This determinism is what LIME requires of its black box.

The three forces mirror TEB's weights:

* **smoothness** (elastic contraction toward neighbours),
* **clearance** (bounded repulsion away from obstacles within an influence band,
  analogous to ``weight_obstacle`` / ``inflation_dist``),
* **via-point** (pull back toward the original global-plan node, analogous to
  ``weight_viapoint``).

We aim for *qualitative* fidelity to TEB -- the same sign of deviation per
obstacle, hence the same green/red colouring -- not numeric equality. The
``Ros2ControllerPlanner`` backend is the route to exact TEB numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..costmap import CostmapInfo, OBSTACLE_LEVEL, world_to_pixel
from .base import PlanRequest, PlanResult


@dataclass(frozen=True)
class ElasticBandConfig:
    """Tunable, but fixed-at-construction, planner gains.

    Defaults are chosen to bow paths around obstacles in a TEB-like way while
    staying numerically stable (the clearance term is bounded, so it never blows
    up near an obstacle). ``inflation_dist`` mirrors the TEB config value.
    """

    iterations: int = 60  # fixed -> deterministic (TEB no_inner * no_outer = 20..)
    node_spacing: float = 0.15  # m, ~ dt_ref(0.3) * max_vel_x(0.5)
    # Obstacles only deflect the band once they intrude within this margin of the
    # robot footprint. Keeping it tight means the band follows the global plan in
    # open space (deviation ~ 0) and only detours around a genuine obstruction --
    # so removing an obstruction reduces deviation, as in the paper.
    clearance_margin: float = 0.2  # m, extra clearance beyond the footprint radius
    w_smooth: float = 0.1
    w_clearance: float = 1.0
    w_viapoint: float = 0.5
    step: float = 1.0
    max_step: float = 0.05  # m, max per-node displacement per iteration
    lateral_probe: float = 0.4  # m, how far to probe sideways for clearance


class ElasticBandPlanner:
    """Deterministic elastic-band backend implementing :class:`PlannerBackend`."""

    def __init__(self, config: ElasticBandConfig | None = None):
        self.config = config or ElasticBandConfig()

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
        """Resample a polyline to roughly uniform arc-length ``spacing``."""
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 2:
            return points
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-9:
            return points[:1]
        n = max(2, int(round(total / spacing)) + 1)
        targets = np.linspace(0.0, total, n)
        xs = np.interp(targets, s, points[:, 0])
        ys = np.interp(targets, s, points[:, 1])
        return np.column_stack([xs, ys])

    @staticmethod
    def _clip_to_window(global_plan: np.ndarray, info: CostmapInfo) -> np.ndarray:
        """Keep the prefix of the global plan that lies inside the costmap."""
        x0 = info.origin_x
        y0 = info.origin_y
        x1 = x0 + info.width * info.resolution
        y1 = y0 + info.height * info.resolution
        inside = (
            (global_plan[:, 0] >= x0)
            & (global_plan[:, 0] < x1)
            & (global_plan[:, 1] >= y0)
            & (global_plan[:, 1] < y1)
        )
        if not inside.any():
            return global_plan
        # contiguous prefix from the first inside point
        idx = np.where(inside)[0]
        return global_plan[idx[0] : idx[-1] + 1]

    def _fields(self, costmap: np.ndarray, info: CostmapInfo):
        """Distance-to-obstacle field (metres) and its unit gradient (away)."""
        free = costmap < OBSTACLE_LEVEL
        dist_px = distance_transform_edt(free)
        dist_m = dist_px * info.resolution
        # gradient of distance points away from obstacles
        gy, gx = np.gradient(dist_px)
        norm = np.hypot(gx, gy)
        norm[norm < 1e-9] = 1.0
        return dist_m, gx / norm, gy / norm

    # -- main entry point ------------------------------------------------------
    def plan(self, request: PlanRequest) -> PlanResult:
        cfg = self.config
        info = request.costmap_info
        costmap = np.asarray(request.costmap, dtype=np.float64)

        gplan = np.asarray(request.global_plan, dtype=np.float64)
        gplan = self._clip_to_window(gplan, info)
        if len(gplan) < 2:
            return PlanResult(local_plan=np.empty((0, 2)), meta={"feasible": False})

        band = self._resample(gplan, cfg.node_spacing)
        original = band.copy()  # via-point reference
        # Pin the first node to the robot pose.
        band[0] = (request.start_pose.x, request.start_pose.y)
        original[0] = band[0]

        dist_m, ux, uy = self._fields(costmap, info)
        d0 = request.footprint_radius + cfg.clearance_margin

        def sample(pt):
            col, row = world_to_pixel(pt[0], pt[1], info)
            col = min(max(col, 0), info.width - 1)
            row = min(max(row, 0), info.height - 1)
            return dist_m[row, col], np.array([ux[row, col], uy[row, col]])

        probe = cfg.lateral_probe
        for _ in range(cfg.iterations):
            new = band.copy()
            for i in range(1, len(band) - 1):
                p = band[i]
                smooth = band[i - 1] + band[i + 1] - 2.0 * p
                d, grad = sample(p)
                push = max(0.0, (d0 - d) / d0)  # bounded: 0 at band edge, 1 at obstacle

                clearance = np.zeros(2)
                if push > 0.0:
                    # Probe perpendicular to the travel direction and steer toward
                    # the side with more clearance (breaks the head-on symmetry
                    # that a pure gradient field cannot resolve). Ties go left so
                    # the planner stays deterministic.
                    tangent = band[i + 1] - band[i - 1]
                    tnorm = np.linalg.norm(tangent)
                    if tnorm > 1e-9:
                        tangent = tangent / tnorm
                        normal = np.array([-tangent[1], tangent[0]])  # left of travel
                        d_left, _ = sample(p + normal * probe)
                        d_right, _ = sample(p - normal * probe)
                        side = 1.0 if d_left >= d_right else -1.0
                        clearance = (normal * side + 0.5 * grad) * push
                    else:
                        clearance = grad * push

                via = original[i] - p
                delta = cfg.step * (
                    cfg.w_smooth * smooth
                    + cfg.w_clearance * clearance
                    + cfg.w_viapoint * via
                )
                mag = float(np.linalg.norm(delta))
                if mag > cfg.max_step:
                    delta *= cfg.max_step / mag
                new[i] = p + delta
            band = new

        # Mark feasibility: did the final band stay clear of the lethal core?
        clearances = np.array([sample(p)[0] for p in band])
        feasible = bool(np.all(clearances >= request.footprint_radius * 0.5))
        return PlanResult(
            local_plan=band,
            meta={
                "feasible": feasible,
                "iterations": cfg.iterations,
                "min_clearance": float(clearances.min()),
            },
        )
