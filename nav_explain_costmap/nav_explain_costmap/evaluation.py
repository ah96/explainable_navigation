"""LIME explanation quality / robustness / runtime evaluation.

* quality    = var(|coef|) - mean(|coef|) (segment-importance separation);
* robustness = % of the coefficient ordering preserved vs the full set;
* runtime    = segmentation and planner time of the full perturbation set.

Quality and robustness are computed for growing subsets of the full perturbation
set. The subsets are taken from one shared dataset, so the planner runs once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from nav_explain_core import (
    DEFAULT_SEGMENTS,
    PerturbationDataset,
    ScenarioInstance,
    build_perturbation_dataset,
)
from nav_explain_core.lime.ridge_explainer import RidgeExplainer
from nav_explain_core.planner.base import PlannerBackend

#: numbers of perturbations at which quality and robustness are evaluated
PERTURBATION_COUNTS = (8, 16, 32, 48, 64, 96, 128, 192, 256)


def _subset(data: np.ndarray, deviations: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``k`` evenly spaced perturbations, always including the instance (row 0)."""
    k = int(min(max(k, 2), data.shape[0]))
    idx = np.unique(np.concatenate([[0], np.linspace(0, data.shape[0] - 1, k).round()])).astype(int)
    return data[idx], deviations[idx]


def _coefs(dataset: PerturbationDataset, data: np.ndarray, deviations: np.ndarray,
           explainer: RidgeExplainer) -> dict[int, float]:
    """LIME coefficients of the obstacle segments, fitted on the given rows."""
    expl = explainer.explain(data, deviations, segment_labels=dataset.labels,
                             num_features=dataset.n_features)
    return {lab: w for lab, w in expl.coef.items() if lab != 0}


def _quality(coefs: dict[int, float]) -> float:
    a = np.abs(np.array(list(coefs.values())))
    return float(np.var(a) - np.mean(a)) if a.size else 0.0


def _order_preserved(ref: dict[int, float], cand: dict[int, float]) -> float:
    """Fraction of segment pairs whose order of importance is the same in both."""
    labels = sorted(set(ref) & set(cand))
    if len(labels) < 2:
        return 1.0
    agree = total = 0
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            total += 1
            if np.sign(abs(ref[a]) - abs(ref[b])) == np.sign(abs(cand[a]) - abs(cand[b])):
                agree += 1
    return agree / total


def reproduce_figures(instance: ScenarioInstance, planner: PlannerBackend, out_dir: Path | str,
                      n_target_segments: int = DEFAULT_SEGMENTS) -> dict:
    """Evaluate LIME on ``instance`` and write ``evaluation.png`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = build_perturbation_dataset(instance, planner, n_target_segments)
    explainer = RidgeExplainer()
    n_max = ds.data.shape[0]
    counts = sorted({c for c in PERTURBATION_COUNTS if c <= n_max} | {n_max})

    ref = _coefs(ds, ds.data, ds.deviations, explainer)
    quality, robustness = [], []
    for k in counts:
        d, y = _subset(ds.data, ds.deviations, k)
        c = _coefs(ds, d, y, explainer)
        quality.append(_quality(c))
        robustness.append(100.0 * _order_preserved(ref, c))

    fig = Figure(figsize=(15, 4))
    ax = fig.subplots(1, 3)
    ax[0].plot(counts, quality, "o-")
    ax[0].set_title("(a) Quality")
    ax[0].set_ylabel("var(|coef|)-mean(|coef|)")
    ax[1].plot(counts, robustness, "o-", color="tab:green")
    ax[1].set_title("(b) Robustness")
    ax[1].set_ylabel("% order preserved")
    ax[2].bar(["segmentation", "planner"], [ds.timings["segmentation_s"], ds.timings["planner_s"]])
    ax[2].set_title("(c) Runtime")
    ax[2].set_ylabel("s")
    for a in ax[:2]:
        a.set_xlabel("number of perturbations")
        a.grid(alpha=0.3)
    fig.suptitle(f"LIME evaluation: {instance.name}")
    fig.tight_layout()
    fig.savefig(out / "evaluation.png", dpi=130)
    return {"counts": counts, "quality": quality, "robustness": robustness, "timings": ds.timings}
