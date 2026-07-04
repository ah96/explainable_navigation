"""Matplotlib rendering of a costmap explanation (kept out of the pure core)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nav_explain_core import ScenarioInstance, world_to_pixel

from .explain import CostmapExplanation


def save_explanation_figure(instance: ScenarioInstance, expl: CostmapExplanation, path: Path | str) -> None:
    """Save the coloured explanation with global/local plans + robot overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = instance.costmap_info

    def to_px(points):
        cols, rows = [], []
        for x, y in np.asarray(points, float).reshape(-1, 2):
            c, r = world_to_pixel(x, y, info)
            cols.append(c); rows.append(r)
        return cols, rows

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(expl.image, origin="lower", interpolation="nearest")
    gc, gr = to_px(instance.global_plan)
    ax.plot(gc, gr, "-", color="blue", linewidth=2, label="global plan")
    if expl.local_plan.size:
        lc, lr = to_px(expl.local_plan)
        ax.plot(lc, lr, "o", color="yellow", markersize=3, label="local plan")
    rc, rr = world_to_pixel(instance.start_pose.x, instance.start_pose.y, info)
    ax.plot([rc], [rr], "o", color="white", markersize=9, markeredgecolor="k", label="robot")
    ax.set_title(f"{expl.method.upper()} — {instance.name}")
    ax.set_xlim(0, info.width); ax.set_ylim(0, info.height)
    ax.legend(loc="upper right", fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(Path(path), dpi=130); plt.close(fig)


def save_comparison_grid(instance: ScenarioInstance, expls: list[CostmapExplanation], path: Path | str) -> None:
    """Save all methods' explanations side by side for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = instance.costmap_info
    n = len(expls)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, expl in zip(axes[0], expls):
        ax.imshow(expl.image, origin="lower", interpolation="nearest")
        gc = [world_to_pixel(x, y, info)[0] for x, y in instance.global_plan]
        gr = [world_to_pixel(x, y, info)[1] for x, y in instance.global_plan]
        ax.plot(gc, gr, "-", color="blue", linewidth=1.5)
        ax.set_title(expl.method.upper()); ax.axis("off")
    fig.suptitle(f"Explanation methods — {instance.name}")
    fig.tight_layout(); fig.savefig(Path(path), dpi=120); plt.close(fig)
