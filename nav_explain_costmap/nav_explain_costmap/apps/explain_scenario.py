"""``nec_explain``: explain a costmap scenario with a chosen method (or all).

Examples::

    nec_explain --scenario doorway_blocked --method lime --out outputs/
    nec_explain --scenario clutter --method all --out outputs/     # compare all four

Writes a coloured explanation image per method, a comparison grid (for ``all``),
``summary.txt`` with the textual explanation of each method and ``summary.json``
with its numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nav_explain_core import DEFAULT_SEGMENTS, build_perturbation_dataset
from nav_explain_core.planner import ElasticBandConfig, ElasticBandPlanner

from ..explain import METHODS, explain_costmap
from ..render import save_comparison_grid, save_explanation_figure
from ..scenarios import SCENARIOS, load_scenario


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Explain a local costmap navigation decision.")
    p.add_argument("--scenario", default="doorway_blocked", choices=sorted(SCENARIOS))
    p.add_argument("--method", default="lime", choices=[*METHODS, "all"])
    p.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    p.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS,
                   help="number of obstacle segments, i.e. interpretable features")
    p.add_argument("--iterations", type=int, default=ElasticBandConfig().iterations,
                   help="relaxation sweeps of the elastic-band planner")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance = load_scenario(args.scenario)
    planner = ElasticBandPlanner(ElasticBandConfig(iterations=args.iterations))
    out = Path(args.out) / instance.name
    out.mkdir(parents=True, exist_ok=True)

    methods = list(METHODS) if args.method == "all" else [args.method]
    # SHAP, Anchors and the counterfactual read the same full dataset: one planner pass.
    dataset = None
    if args.method != "lime":
        dataset = build_perturbation_dataset(instance, planner, args.segments)

    results = []
    for m in methods:
        expl = explain_costmap(instance, m, planner, args.segments, dataset=dataset)
        results.append(expl)
        save_explanation_figure(instance, expl, out / f"{m}.png")
        print(f"[{instance.name}/{m}] {expl.text}")

    if len(results) > 1:
        save_comparison_grid(instance, results, out / "comparison.png")
    (out / "summary.txt").write_text("".join(f"[{r.method}] {r.text}\n" for r in results))
    # JSON object keys must be strings, so the segment labels of LIME and SHAP become strings.
    (out / "summary.json").write_text(json.dumps({r.method: r.summary for r in results}, indent=2))
    print(f"  wrote artefacts to {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
