"""Figures: an explanation, the overview of a run, the evaluation, the study answers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - the backend is chosen before the import
import numpy as np  # noqa: E402

from nav_explain_core.hixron import Explanation, ReplayResult, ReplaySpec  # noqa: E402
from nav_explain_core.semantic.semantic_map import build_semantic_map  # noqa: E402

from .studies import SCALE, Group, answer_shares  # noqa: E402

#: colours of the answer scale, from disagreement to agreement
_SCALE_COLOURS = ("#b2182b", "#ef8a62", "#cfcfcf", "#67a9cf", "#2166ac")


def _extent(spec: ReplaySpec) -> tuple[float, float, float, float]:
    info = spec.map_info
    return (info.origin_x, info.origin_x + info.width * info.resolution,
            info.origin_y, info.origin_y + info.height * info.resolution)


def save_explanation(path: Path, spec: ReplaySpec, explanation: Explanation,
                     plan: np.ndarray | None = None) -> None:
    """The visual explanation with the robot and the plan, and the text as the title."""
    fig, ax = plt.subplots(figsize=(6, 6.6))
    ax.imshow(explanation.layer.rgb, origin="lower", extent=_extent(spec), interpolation="nearest")
    if plan is not None:
        ax.plot(plan[:, 0], plan[:, 1], color="tab:blue", linewidth=1.5)
    robot = explanation.robot
    ax.plot(robot.x, robot.y, marker="o", color="white", markeredgecolor="black", markersize=8)
    ax.set_title(explanation.text or "(visual only)", fontsize=9, wrap=True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_overview(path: Path, spec: ReplaySpec, result: ReplayResult) -> None:
    """The map with every global plan of the run, the driven path and the trigger points."""
    grey = np.where(build_semantic_map(spec.ontology, spec.map_info) > 0, 0.55, 0.9)
    fig, ax = plt.subplots(figsize=(6, 6.6))
    ax.imshow(grey, origin="lower", extent=_extent(spec), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    for i, plan in enumerate(result.plans):
        ax.plot(plan[:, 0], plan[:, 1], linestyle="--", linewidth=1, label=f"plan {i + 1}")
    ax.plot(result.trace[:, 0], result.trace[:, 1], color="black", linewidth=1.5, label="driven")
    for event in spec.events:
        ax.plot(*event.new_centre, marker="s", color="red", markersize=6)
    ax.plot(*spec.goal, marker="*", color="green", markersize=12)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{spec.name}, extroversion {result.extroversion:.1f}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _lines(path: Path, rows: list[dict[str, float]], series: list[tuple[str, str, str, str]],
           ylabel: str) -> None:
    levels = [r["extroversion"] for r in rows]
    fig, ax = plt.subplots()
    for column, label, colour, marker in series:
        ax.plot(levels, [r[column] for r in rows], color=colour, marker=marker, markersize=10,
                label=label)
    ax.legend()
    ax.set_xlabel("extroversion level (percentage)")
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_evaluation(out: Path, rows: list[dict[str, float]], runtimes: bool = True) -> list[Path]:
    """The three figures of the evaluation, from aggregated rows. Returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "eval_N_exps.png", out / "eval_N_artefacts.png"]
    _lines(files[0], rows, [("visual_N", "visual explanations", "red", "o"),
                            ("textual_N", "textual explanations", "blue", ".")],
           "number of explanations generated")
    _lines(files[1], rows,
           [("N_objects", "average number of objects in visual explanations", "red", "o"),
            ("N_words", "average number of words in textual explanations", "blue", "."),
            ("N_deviations", "number of deviations explained", "green", "x")], "")
    if runtimes:
        files.append(out / "eval_runtimes.png")
        _lines(files[2], rows, [("visual_time", "visual explanations", "red", "o"),
                                ("textual_time", "textual explanations", "blue", ".")],
               "average explanation runtime (milliseconds)")
    return files


def save_answers(path: Path, groups: list[tuple[str, Group]], statements: tuple[str, ...]) -> None:
    """Stacked bars of the answers per statement, one panel per group."""
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4), sharey=True)
    for ax, (title, group) in zip(np.atleast_1d(axes), groups):
        left = np.zeros(len(statements))
        for value, (label, colour) in enumerate(zip(SCALE, _SCALE_COLOURS)):
            share = np.array([s[value] for s in answer_shares(group)])
            ax.barh(range(len(statements)), share, left=left, color=colour, label=label)
            left += share
        ax.set_yticks(range(len(statements)), [f"Q{i + 1}" for i in range(len(statements))])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("participants (percent)")
        ax.set_title(title)
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(SCALE), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
