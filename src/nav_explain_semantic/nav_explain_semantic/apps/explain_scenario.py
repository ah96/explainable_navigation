"""``nes_explain`` — visuo-textual explanation of a semantic scenario.

Example::

    nes_explain --scenario cabinet_room --out outputs/

Writes one ``explanation_<object>_<affordance>.png`` per feature, a combined
``explanation_grid.png``, ``coefficients.csv`` and ``explanations.txt``
(descriptive + suggestive sentences).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from nav_explain_core.planner import ElasticBandConfig, ElasticBandPlanner
from nav_explain_core.semantic import VisuoTextualExplainer

from ..render import save_explanation_grid, save_explanation_maps
from ..scenarios import available_scenarios, load_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Visuo-textual explanation of a navigational choice.")
    p.add_argument("--scenario", default="cabinet_room")
    p.add_argument("--mode", default="one_off", choices=["one_off", "full_2n"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--iterations", type=int, default=60)
    args = p.parse_args(argv)

    if args.scenario not in available_scenarios():
        raise SystemExit(f"unknown scenario '{args.scenario}'. Available: {available_scenarios()}")
    instance = load_scenario(args.scenario)
    planner = ElasticBandPlanner(ElasticBandConfig(iterations=args.iterations))
    result = VisuoTextualExplainer(planner, mode=args.mode).explain(instance)

    out = Path(args.out) / instance.name
    out.mkdir(parents=True, exist_ok=True)
    save_explanation_maps(instance, result, out)
    save_explanation_grid(instance, result, out / "explanation_grid.png")

    with open(out / "coefficients.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "affordance", "coefficient"])
        for (oid, aff), coef in sorted(result.coefficients.items(), key=lambda kv: -abs(kv[1])):
            w.writerow([instance.ontology.by_id(oid).label, aff, coef])

    lines = ["# Descriptive"] + result.descriptive_text + ["", "# Suggestive"] + result.suggestive_text
    (out / "explanations.txt").write_text("\n".join(lines) + "\n")
    (out / "timing.json").write_text(json.dumps(
        {"scenario": instance.name, "features": len(result.pairs),
         "deviation": result.original_deviation, "runtime_s": result.runtime_s}, indent=2))

    print(f"[{instance.name}] {len(result.pairs)} features, deviation={result.original_deviation:.3f}, "
          f"runtime={result.runtime_s:.3f}s")
    if result.descriptive_text:
        print("  →", result.descriptive_text[0])
        print("  →", result.suggestive_text[0])
    print(f"  wrote artefacts to {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
