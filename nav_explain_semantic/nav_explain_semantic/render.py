"""Matplotlib rendering of the semantic explanation maps (kept out of the core).

The figures are built with ``matplotlib.figure.Figure`` and written to files, so
no display and no ``pyplot`` backend are needed.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nav_explain_core import world_to_pixel, world_to_pixels
from nav_explain_core.semantic import VTInstance, VTResult


def _overlay(ax: Axes, instance: VTInstance, result: VTResult) -> None:
    """Draw the global plan, the local plan, the robot and the object labels."""
    info = instance.map_info.to_costmap_info()
    gc, gr = world_to_pixels(instance.global_plan, info)
    ax.plot(gc, gr, "-", color="blue", linewidth=2)
    if result.original_local_plan.size:
        lc, lr = world_to_pixels(result.original_local_plan, info)
        ax.plot(lc, lr, "o", color="yellow", markersize=2)
    rc, rr = world_to_pixel(instance.robot_pose.x, instance.robot_pose.y, info)
    ax.plot([rc], [rr], "o", color="white", markersize=8, markeredgecolor="k")
    # object labels
    for obj in instance.ontology:
        if not obj.affordances:
            continue
        c, r = world_to_pixel(obj.cx, obj.cy, info)
        ax.text(c, r, obj.label, color="k", fontsize=7, ha="center", va="center")
    ax.set_xlim(0, info.width)
    ax.set_ylim(0, info.height)
    ax.axis("off")


def save_explanation_maps(instance: VTInstance, result: VTResult, out_dir: Path | str) -> list[Path]:
    """Save one coloured semantic map per (object, affordance) feature."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for (obj_id, aff), img in result.explanation_maps.items():
        label = instance.ontology.by_id(obj_id).label
        coef = result.coefficients[(obj_id, aff)]
        fig = Figure(figsize=(6, 5))
        ax = fig.subplots()
        ax.imshow(img, origin="lower", interpolation="nearest")
        _overlay(ax, instance, result)
        ax.set_title(f"{label} -- {aff}  (coef={coef:+.3f})")
        fig.tight_layout()
        p = out / f"explanation_{label}_{aff}.png"
        fig.savefig(p, dpi=120)
        paths.append(p)
    return paths


def save_explanation_grid(instance: VTInstance, result: VTResult, path: Path | str) -> None:
    """Save all feature maps in one grid."""
    items = list(result.explanation_maps.items())
    n = len(items)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = Figure(figsize=(5 * cols, 4 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, ((obj_id, aff), img) in enumerate(items):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(img, origin="lower", interpolation="nearest")
        _overlay(ax, instance, result)
        label = instance.ontology.by_id(obj_id).label
        coef = result.coefficients[(obj_id, aff)]
        ax.set_title(f"{label}-{aff} ({coef:+.2f})", fontsize=9)
    fig.suptitle(f"Visuo-textual explanation maps -- {instance.name}")
    fig.tight_layout()
    fig.savefig(Path(path), dpi=120)
