"""LIME explanation quality / robustness / runtime evaluation.

* quality    = var(|coef|) - mean(|coef|) (segment-importance separation);
* robustness = % of the coefficient ordering preserved vs the full set;
* runtime    = segmentation / planner / total vs the number of perturbations.

Quality rises and robustness degrades as perturbations are removed, mirroring the
expected LIME behaviour. Reuses the shared core perturbation dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nav_explain_core import ScenarioInstance, build_perturbation_dataset
from nav_explain_core.lime.ridge_explainer import RidgeExplainer
from nav_explain_core.planner.base import PlannerBackend


def _subset(data, deviations, k):
    k = int(min(max(k, 2), data.shape[0]))
    idx = np.unique(np.concatenate([[0], np.linspace(0, data.shape[0] - 1, k).round()])).astype(int)
    return data[idx], deviations[idx]


def _coefs(dataset, data, deviations, explainer):
    expl = explainer.explain(data, deviations, segment_labels=dataset.labels, num_features=dataset.n_features)
    return {lab: w for lab, w in expl.coef.items() if lab != 0}


def _quality(coefs):
    a = np.abs(np.array(list(coefs.values())))
    return float(np.var(a) - np.mean(a)) if a.size else 0.0


def _order_preserved(ref, cand):
    labels = sorted(set(ref) & set(cand))
    if len(labels) < 2:
        return 1.0
    agree = total = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            total += 1
            if np.sign(abs(ref[a]) - abs(ref[b])) == np.sign(abs(cand[a]) - abs(cand[b])):
                agree += 1
    return agree / total if total else 1.0


def reproduce_figures(instance: ScenarioInstance, planner: PlannerBackend, out_dir: Path | str,
                      n_target_segments: int = 8) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = build_perturbation_dataset(instance, planner, n_target_segments)
    explainer = RidgeExplainer()
    n_max = ds.data.shape[0]
    counts = sorted({c for c in [8, 16, 32, 48, 64, 96, 128, 192, 256] if c <= n_max} | {n_max})

    ref = _coefs(ds, ds.data, ds.deviations, explainer)
    quality, robustness = [], []
    for k in counts:
        d, y = _subset(ds.data, ds.deviations, k)
        c = _coefs(ds, d, y, explainer)
        quality.append(_quality(c))
        robustness.append(100.0 * _order_preserved(ref, c))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(counts, quality, "o-"); ax[0].set_title("(a) Quality"); ax[0].set_ylabel("var(|coef|)-mean(|coef|)")
    ax[1].plot(counts, robustness, "o-", color="tab:green"); ax[1].set_title("(b) Robustness"); ax[1].set_ylabel("% order preserved")
    ax[2].bar(["segmentation", "planner"], [ds.timings["segmentation_s"], ds.timings["planner_s"]])
    ax[2].set_title("(c) Runtime"); ax[2].set_ylabel("s")
    for a in ax[:2]:
        a.set_xlabel("number of perturbations"); a.grid(alpha=0.3)
    fig.suptitle(f"LIME evaluation — {instance.name}")
    fig.tight_layout(); fig.savefig(out / "evaluation.png", dpi=130); plt.close(fig)
    return {"counts": counts, "quality": quality, "robustness": robustness, "timings": ds.timings}
