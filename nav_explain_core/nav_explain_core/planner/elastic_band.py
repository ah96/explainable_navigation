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
bit-identical ``local_plan`` outputs (asserted by ``test_planner_deterministic``).
This determinism is what LIME requires of its black box.

The three forces mirror TEB's weights:

* **smoothness** (elastic contraction toward neighbours),
* **clearance** (bounded repulsion away from obstacles within an influence band,
  analogous to ``weight_obstacle`` / ``inflation_dist``),
* **via-point** (pull back toward the original global-plan node, analogous to
  ``weight_viapoint``).

We aim for *qualitative* fidelity to TEB -- the same sign of deviation per
obstacle, hence the same green/red colouring -- not numeric equality. The
``PlannerBackend`` interface is where a real controller can take its place.

Every iteration updates all band nodes at once from the previous band (a Jacobi
sweep), so the update is written as array operations over the nodes. The planner
is called once per perturbation, which makes this loop the hot path of every
explanation method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..costmap import OBSTACLE_LEVEL, CostmapInfo, world_to_pixels
from .base import PlanRequest, PlanResult

# Below this length (metres) a segment or a tangent counts as degenerate.
_EPS = 1e-9


@dataclass(frozen=True)
class ElasticBandConfig:
    """Tunable, but fixed-at-construction, planner gains.

    Defaults are chosen to bow paths around obstacles in a TEB-like way while
    staying numerically stable (the clearance term is bounded, so it never blows
    up near an obstacle).
    """

    #: number of relaxation sweeps; fixed, so the planner is deterministic
    iterations: int = 60
    #: spacing of the band nodes in metres, about dt_ref (0.3) * max_vel_x (0.5)
    node_spacing: float = 0.15
    #: extra clearance beyond the footprint radius, in metres. Obstacles deflect
    #: the band only inside this margin, so the band follows the global plan in
    #: open space (deviation near 0) and detours only around a real obstruction.
    #: Removing an obstruction therefore reduces the deviation.
    clearance_margin: float = 0.2
    #: weight of the smoothness force
    w_smooth: float = 0.1
    #: weight of the clearance force
    w_clearance: float = 1.0
    #: weight of the pull back to the global plan
    w_viapoint: float = 0.5
    #: step size of one sweep
    step: float = 1.0
    #: largest displacement of a node in one sweep, in metres
    max_step: float = 0.05
    #: how far to probe sideways for clearance, in metres
    lateral_probe: float = 0.4


class ElasticBandPlanner:
    """Deterministic elastic-band backend implementing :class:`PlannerBackend`."""

    def __init__(self, config: ElasticBandConfig | None = None) -> None:
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
        if total < _EPS:
            return points[:1]
        n = max(2, int(round(total / spacing)) + 1)
        targets = np.linspace(0.0, total, n)
        xs = np.interp(targets, s, points[:, 0])
        ys = np.interp(targets, s, points[:, 1])
        return np.column_stack([xs, ys])

    @staticmethod
    def _clip_to_window(global_plan: np.ndarray, info: CostmapInfo) -> np.ndarray:
        """Keep the global plan from its first to its last point inside the costmap.

        A plan with no point inside the costmap is returned unchanged.
        """
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
        idx = np.where(inside)[0]
        return global_plan[idx[0] : idx[-1] + 1]

    @staticmethod
    def _fields(costmap: np.ndarray, info: CostmapInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance-to-obstacle field (metres) and its unit gradient (away)."""
        free = costmap < OBSTACLE_LEVEL
        dist_px = distance_transform_edt(free)
        dist_m = dist_px * info.resolution
        # the gradient of the distance points away from obstacles
        gy, gx = np.gradient(dist_px)
        norm = np.hypot(gx, gy)
        norm[norm < _EPS] = 1.0
        return dist_m, gx / norm, gy / norm

    @staticmethod
    def _cells(points: np.ndarray, info: CostmapInfo) -> tuple[np.ndarray, np.ndarray]:
        """Grid ``(rows, cols)`` of world points, clamped to the costmap."""
        cols, rows = world_to_pixels(points, info)
        np.clip(cols, 0, info.width - 1, out=cols)
        np.clip(rows, 0, info.height - 1, out=rows)
        return rows, cols

    # -- main entry point ------------------------------------------------------
    def plan(self, request: PlanRequest) -> PlanResult:
        cfg = self.config
        info = request.costmap_info
        costmap = np.asarray(request.costmap, dtype=np.float64)

        gplan = np.asarray(request.global_plan, dtype=np.float64)
        gplan = self._clip_to_window(gplan, info)
        if len(gplan) < 2:
            return PlanResult(local_plan=np.empty((0, 2)))

        band = self._resample(gplan, cfg.node_spacing)
        # Pin the first node to the robot pose.
        band[0] = (request.start_pose.x, request.start_pose.y)
        original = band.copy()  # via-point reference

        dist_m, ux, uy = self._fields(costmap, info)
        d0 = request.footprint_radius + cfg.clearance_margin
        probe = cfg.lateral_probe

        for _ in range(cfg.iterations):
            prev, p, nxt = band[:-2], band[1:-1], band[2:]
            smooth = prev + nxt - 2.0 * p

            rows, cols = self._cells(p, info)
            # bounded push: 0 at the edge of the influence band, 1 at the obstacle
            push = np.maximum(0.0, (d0 - dist_m[rows, cols]) / d0)

            clearance = np.zeros_like(p)
            near = push > 0.0
            if near.any():
                pn = p[near]
                grad = np.column_stack([ux[rows[near], cols[near]], uy[rows[near], cols[near]]])
                tangent = nxt[near] - prev[near]
                tnorm = np.hypot(tangent[:, 0], tangent[:, 1])
                moving = tnorm > _EPS
                tangent = tangent / np.where(moving, tnorm, 1.0)[:, None]
                normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # left of travel
                # Probe perpendicular to the travel direction and steer toward
                # the side with more clearance. This breaks the head-on symmetry
                # that a pure gradient field cannot resolve. Ties go left, so
                # the planner stays deterministic.
                d_left = dist_m[self._cells(pn + normal * probe, info)]
                d_right = dist_m[self._cells(pn - normal * probe, info)]
                side = np.where(d_left >= d_right, 1.0, -1.0)
                steer = np.where(moving[:, None], normal * side[:, None] + 0.5 * grad, grad)
                clearance[near] = steer * push[near, None]

            via = original[1:-1] - p
            delta = cfg.step * (
                cfg.w_smooth * smooth + cfg.w_clearance * clearance + cfg.w_viapoint * via
            )
            mag = np.hypot(delta[:, 0], delta[:, 1])
            too_far = mag > cfg.max_step
            scale = np.where(too_far, cfg.max_step / np.where(too_far, mag, 1.0), 1.0)
            new = band.copy()
            new[1:-1] = p + delta * scale[:, None]
            band = new

        return PlanResult(local_plan=band)
