"""Persist and reload perturbation datasets.

Querying the planner for the full 2**n perturbation set is the expensive part of
an explanation. This module caches a :class:`PerturbationDataset` to a compressed
``.npz`` so the same scenario can be re-explained (with any method) without
re-running the planner. It replaces the CSV tooling (``DataLoader``,
``DatasetCreator``) of the original ROS 1 code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .perturbation_dataset import PerturbationDataset


def _npz_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def save_dataset(dataset: PerturbationDataset, path: Path | str) -> Path:
    """Write ``dataset`` to ``path`` (the suffix ``.npz`` is enforced)."""
    path = _npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        segments=dataset.segments,
        labels=dataset.labels,
        data=dataset.data,
        deviations=dataset.deviations,
        local_plan=dataset.local_plan,
        segmentation_s=dataset.timings.get("segmentation_s", 0.0),
        planner_s=dataset.timings.get("planner_s", 0.0),
    )
    return path


def load_dataset(path: Path | str) -> PerturbationDataset:
    """Read a dataset written by :func:`save_dataset`."""
    with np.load(_npz_path(path)) as z:
        return PerturbationDataset(
            segments=z["segments"],
            labels=z["labels"],
            data=z["data"],
            deviations=z["deviations"],
            # files written before the local plan was stored do not have it
            local_plan=z["local_plan"] if "local_plan" in z else np.empty((0, 2)),
            timings={
                "segmentation_s": float(z["segmentation_s"]),
                "planner_s": float(z["planner_s"]),
            },
        )
