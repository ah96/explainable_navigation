"""Persist and reload perturbation datasets.

Querying the planner for the full 2**n perturbation set is the expensive part of
an explanation. This module caches a :class:`PerturbationDataset` to a compressed
``.npz`` so the same scenario can be re-explained (with any method) without
re-running the planner. This is the modern replacement for the original ROS1
``DataLoader``/``DatasetCreator`` CSV tooling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .perturbation_dataset import PerturbationDataset


def save_dataset(dataset: PerturbationDataset, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        segments=dataset.segments,
        labels=dataset.labels,
        data=dataset.data,
        deviations=dataset.deviations,
        segmentation_s=dataset.timings.get("segmentation_s", 0.0),
        planner_s=dataset.timings.get("planner_s", 0.0),
    )
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_dataset(path: Path | str) -> PerturbationDataset:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    z = np.load(path)
    return PerturbationDataset(
        segments=z["segments"],
        labels=z["labels"],
        data=z["data"],
        deviations=z["deviations"],
        timings={
            "segmentation_s": float(z["segmentation_s"]),
            "planner_s": float(z["planner_s"]),
        },
    )
