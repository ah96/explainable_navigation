"""Exact Shapley-value attributions over the perturbation set.

SHAP explains a prediction by distributing the difference between the model
output and a baseline across the features according to their (game-theoretic)
Shapley values. Because the number of obstacle segments is small (<= 8) and we
enumerate the full 2**n perturbation set, the Shapley values can be computed
*exactly* (no sampling), which makes this a deterministic, principled companion
to LIME's approximate additive coefficients.

For each obstacle feature i, the Shapley value is the average over all subsets
S not containing i of the marginal contribution ``v(S + i) - v(S)``, where
``v(S)`` is the planner deviation when exactly the obstacles in S are present.
The values sum to ``v(all obstacles) - v(no obstacles)`` (efficiency).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from .perturbation_dataset import PerturbationDataset


@dataclass
class ShapExplanation:
    values: dict[int, float]      # segment label -> Shapley value
    order: list[int]              # segment labels by descending |value|
    base_value: float             # v(no obstacles present)
    instance_value: float         # v(all obstacles present)


def _value_lookup(dataset: PerturbationDataset) -> dict[frozenset, float]:
    """Map each present-obstacle subset (by column index) to its deviation."""
    cols = dataset.obstacle_columns
    lut: dict[frozenset, float] = {}
    for row, dev in zip(dataset.data, dataset.deviations):
        present = frozenset(c for c in cols if row[c] == 1)
        lut[present] = float(dev)
    return lut


def shapley_values(dataset: PerturbationDataset) -> ShapExplanation:
    cols = dataset.obstacle_columns
    n = len(cols)
    lut = _value_lookup(dataset)
    full = frozenset(cols)

    values = {c: 0.0 for c in cols}
    for i in cols:
        others = [c for c in cols if c != i]
        acc = 0.0
        for k in range(len(others) + 1):
            weight = 1.0 / (n * comb(n - 1, k))
            for subset in combinations(others, k):
                S = frozenset(subset)
                v_S = lut.get(S)
                v_Si = lut.get(S | {i})
                if v_S is None or v_Si is None:
                    continue
                acc += weight * (v_Si - v_S)
        values[i] = acc

    labels = dataset.labels
    by_label = {int(labels[c]): v for c, v in values.items()}
    order = sorted(by_label, key=lambda label: abs(by_label[label]), reverse=True)
    return ShapExplanation(
        values=by_label,
        order=order,
        base_value=lut.get(frozenset(), 0.0),
        instance_value=lut.get(full, 0.0),
    )
