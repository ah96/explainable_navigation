"""``nec_explain`` — explain a costmap scenario with a chosen method (or all).

Examples::

    nec_explain --scenario doorway_blocked --method lime --out outputs/
    nec_explain --scenario clutter --method all --out outputs/     # compare all four

Writes a coloured explanation image per method, a comparison grid (for ``all``),
and a ``summary.txt`` with the per-method textual explanations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nav_explain_core import build_perturbation_dataset
from nav_explain_core.planner import ElasticBandConfig, ElasticBandPlanner

from ..explain import METHODS, explain_costmap
from ..render import save_comparison_grid, save_explanation_figure
from ..scenarios import SCENARIOS, load_scenario


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Explain a local costmap navigation decision.")
    p.add_argument("--scenario", default="doorway_blocked", choices=sorted(SCENARIOS))
    p.add_argument("--method", default="lime", choices=[*METHODS, "all"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--segments", type=int, default=8)
    p.add_argument("--iterations", type=int, default=60)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance = load_scenario(args.scenario)
    planner = ElasticBandPlanner(ElasticBandConfig(iterations=args.iterations))
    out = Path(args.out) / instance.name
    out.mkdir(parents=True, exist_ok=True)

    methods = list(METHODS) if args.method == "all" else [args.method]
    # share one dataset across shap/anchors/counterfactual
    dataset = build_perturbation_dataset(instance, planner, args.segments) if (
        args.method == "all" or args.method in ("shap", "counterfactual")
    ) else None

    results = []
    summary_lines = []
    for m in methods:
        expl = explain_costmap(instance, m, planner, args.segments, dataset=dataset)
        results.append(expl)
        save_explanation_figure(instance, expl, out / f"{m}.png")
        summary_lines.append(f"[{m}] {expl.text}")
        print(f"[{instance.name}/{m}] {expl.text}")

    if len(results) > 1:
        save_comparison_grid(instance, results, out / "comparison.png")
    (out / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {r.method: {k: (v if _js(v) else str(v)) for k, v in r.summary.items()} for r in results},
        indent=2, default=str))
    print(f"  wrote artefacts to {out}/")
    return 0


def _js(v):
    return isinstance(v, (int, float, str, list, dict, bool))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
