"""Matplotlib rendering of the semantic explanation maps (kept out of the core)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nav_explain_core import world_to_pixel
from nav_explain_core.semantic import VTInstance, VTResult


def _overlay(ax, instance: VTInstance, result: VTResult):
    info = instance.map_info.to_costmap_info()

    def to_px(points):
        cols, rows = [], []
        for x, y in np.asarray(points, dtype=float).reshape(-1, 2):
            c, r = world_to_pixel(x, y, info)
            cols.append(c)
            rows.append(r)
        return cols, rows

    gc, gr = to_px(instance.global_plan)
    ax.plot(gc, gr, "-", color="blue", linewidth=2)
    if result.original_local_plan.size:
        lc, lr = to_px(result.original_local_plan)
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for (obj_id, aff), img in result.explanation_maps.items():
        label = instance.ontology.by_id(obj_id).label
        coef = result.coefficients[(obj_id, aff)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(img, origin="lower", interpolation="nearest")
        _overlay(ax, instance, result)
        ax.set_title(f"{label} -- {aff}  (coef={coef:+.3f})")
        fig.tight_layout()
        p = out / f"explanation_{label}_{aff}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def save_explanation_grid(instance: VTInstance, result: VTResult, path: Path | str) -> None:
    """Save all feature maps in one grid (mirrors the paper's Fig. r3--r6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(result.explanation_maps.items())
    n = len(items)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
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
    plt.close(fig)
