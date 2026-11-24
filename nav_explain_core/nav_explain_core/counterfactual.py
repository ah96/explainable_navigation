"""Counterfactual explanations over the perturbation set.

A counterfactual answers "what would have to change for the robot *not* to
deviate?" Concretely: the smallest set of obstacles whose removal brings the
deviation below a target (a fraction of the instance deviation). Because the full
2**n perturbation set is enumerated, we can find the exact minimal-cardinality
counterfactual (ties broken by lowest resulting deviation) deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perturbation_dataset import PerturbationDataset


@dataclass
class CounterfactualExplanation:
    removed_segments: list[int]   # segment labels to remove
    resulting_deviation: float
    instance_deviation: float
    target: float
    found: bool = True


def counterfactual(dataset: PerturbationDataset, target_ratio: float = 0.25) -> CounterfactualExplanation:
    """Smallest set of obstacles whose removal drops deviation below the target.

    ``target_ratio`` is relative to the instance (all-obstacles) deviation.
    """
    cols = dataset.obstacle_columns
    labels = dataset.labels
    d0 = float(dataset.deviations[0])  # data[0] is the all-present instance
    target = target_ratio * d0

    best = None  # (n_removed, resulting_dev, removed_cols)
    for row, dev in zip(dataset.data, dataset.deviations):
        removed = [c for c in cols if row[c] == 0]  # obstacles turned off
        if dev <= target:
            key = (len(removed), dev)
            if best is None or key < (best[0], best[1]):
                best = (len(removed), dev, removed)

    if best is None:
        # nothing achieves the target; report the perturbation with min deviation
        idx = int(np.argmin(dataset.deviations))
        removed = [c for c in cols if dataset.data[idx][c] == 0]
        return CounterfactualExplanation(
            removed_segments=[int(labels[c]) for c in removed],
            resulting_deviation=float(dataset.deviations[idx]),
            instance_deviation=d0,
            target=target,
            found=False,
        )

    _, dev, removed = best
    return CounterfactualExplanation(
        removed_segments=[int(labels[c]) for c in removed],
        resulting_deviation=dev,
        instance_deviation=d0,
        target=target,
        found=True,
    )
