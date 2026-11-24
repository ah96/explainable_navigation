"""Matplotlib rendering of a costmap explanation (kept out of the pure core).

The figures are built with ``matplotlib.figure.Figure`` and written to files, so
no display and no ``pyplot`` backend are needed.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nav_explain_core import ScenarioInstance, world_to_pixel, world_to_pixels

from .explain import CostmapExplanation


def _draw(ax: Axes, instance: ScenarioInstance, expl: CostmapExplanation,
          plan_width: float, local_size: float, robot_size: float) -> None:
    """Draw the explanation image with the global plan, local plan and robot on top."""
    info = instance.costmap_info
    ax.imshow(expl.image, origin="lower", interpolation="nearest")
    gc, gr = world_to_pixels(instance.global_plan, info)
    ax.plot(gc, gr, "-", color="blue", linewidth=plan_width, label="global plan")
    if expl.local_plan.size:
        lc, lr = world_to_pixels(expl.local_plan, info)
        ax.plot(lc, lr, "o", color="yellow", markersize=local_size, label="local plan")
    rc, rr = world_to_pixel(instance.start_pose.x, instance.start_pose.y, info)
    ax.plot([rc], [rr], "o", color="white", markersize=robot_size, markeredgecolor="k", label="robot")
    ax.set_xlim(0, info.width)
    ax.set_ylim(0, info.height)
    ax.axis("off")


def save_explanation_figure(instance: ScenarioInstance, expl: CostmapExplanation, path: Path | str) -> None:
    """Save the coloured explanation with global/local plans + robot overlaid."""
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    _draw(ax, instance, expl, plan_width=2, local_size=3, robot_size=9)
    ax.set_title(f"{expl.method.upper()}: {instance.name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=130)


def save_comparison_grid(instance: ScenarioInstance, expls: list[CostmapExplanation], path: Path | str) -> None:
    """Save all methods' explanations side by side for comparison."""
    n = len(expls)
    fig = Figure(figsize=(5 * n, 5))
    axes = fig.subplots(1, n, squeeze=False)
    for ax, expl in zip(axes[0], expls):
        _draw(ax, instance, expl, plan_width=1.5, local_size=2, robot_size=7)
        ax.set_title(expl.method.upper())
    fig.suptitle(f"Explanation methods: {instance.name}")
    fig.tight_layout()
    fig.savefig(Path(path), dpi=120)
