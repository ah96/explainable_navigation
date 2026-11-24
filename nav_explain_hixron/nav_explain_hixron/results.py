"""The recorded results of the ICSR 2023 experiments, and their aggregation.

A robot with a fixed extroversion crossed the simulated library, and the node
wrote one row per pass of its explanation loop. ``data/icsr2023/results`` holds
these rows: one folder per machine, one file per extroversion level
(``0.csv`` is extroversion 0, ``10.csv`` is extroversion 1). The folder
``laptop`` is the run the publication reports.

The aggregation is the one of the original evaluation script. It needs only the
standard library, so that it runs anywhere.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .scenarios import data_root

#: the machines whose runs are recorded
MACHINES = ("laptop", "laptop2", "labpc")
#: the machine whose run the publication reports
REPORTED_MACHINE = "laptop"
#: extroversion levels in percent, one recorded file each
LEVELS = tuple(range(0, 101, 10))
#: columns of an aggregated row
COLUMNS = ("extroversion", "visual_time", "visual_N", "textual_time", "textual_N",
           "N_objects", "N_words", "N_deviations")


def results_folder(machine: str) -> Path:
    if machine not in MACHINES:
        raise KeyError(f"unknown machine '{machine}'; available: {list(MACHINES)}")
    return data_root() / "icsr2023" / "results" / machine


def aggregate_level(path: Path, extroversion: int) -> dict[str, float]:
    """One aggregated row from the recorded rows of one extroversion level.

    Runtimes are means over the rows, in milliseconds. The counters of the
    explanations and of the explained deviations are running totals, so their
    last value counts. Objects and words are sums over the rows, divided by the
    number of visual and of textual explanations.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    visual_n = float(rows[-1]["visual_N"])
    textual_n = float(rows[-1]["textual_N"])
    return {
        "extroversion": extroversion,
        "visual_time": 1000 * statistics.fmean(float(r["visual_time"]) for r in rows),
        "visual_N": visual_n,
        "textual_time": 1000 * statistics.fmean(float(r["textual_time"]) for r in rows),
        "textual_N": textual_n,
        "N_objects": sum(float(r["N_objects"]) for r in rows) / visual_n,
        "N_words": sum(float(r["N_words"]) for r in rows) / textual_n,
        "N_deviations": float(rows[-1]["N_deviations"]),
    }


def aggregate(machine: str = REPORTED_MACHINE) -> list[dict[str, float]]:
    """The aggregated rows of one machine, one per extroversion level."""
    folder = results_folder(machine)
    return [aggregate_level(folder / f"{level // 10}.csv", level) for level in LEVELS]


def mean_runtimes(rows: list[dict[str, float]]) -> tuple[float, float]:
    """Mean runtime of a visual and of a textual explanation over the levels, in milliseconds."""
    return (statistics.fmean(r["visual_time"] for r in rows),
            statistics.fmean(r["textual_time"] for r in rows))


def write_rows(path: Path, rows: list[dict[str, float]], columns: tuple[str, ...] = COLUMNS) -> None:
    """Write aggregated rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: (f"{row[c]:.4f}" if isinstance(row[c], float) else row[c]) for c in columns})
