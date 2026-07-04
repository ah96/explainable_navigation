"""``neb_demo`` — showcase every explanation method on the built-in scenarios.

Runs the four costmap methods (LIME/SHAP/Anchors/counterfactual) on a costmap
scenario and the visuo-textual method on the semantic scenario, writing all
artefacts under one output directory. A single headless entry point that
exercises the whole toolkit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nav_explain_core import build_perturbation_dataset
from nav_explain_core.planner import ElasticBandPlanner

from nav_explain_costmap.explain import METHODS, explain_costmap
from nav_explain_costmap.render import save_comparison_grid, save_explanation_figure
from nav_explain_costmap.scenarios import load_scenario as load_costmap
from nav_explain_semantic.apps.explain_scenario import main as semantic_main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Showcase all explanation methods.")
    p.add_argument("--costmap-scenario", default="doorway_blocked")
    p.add_argument("--semantic-scenario", default="cabinet_room")
    p.add_argument("--out", default="outputs/demo")
    args = p.parse_args(argv)
    out = Path(args.out)

    # costmap: all four methods, shared dataset
    inst = load_costmap(args.costmap_scenario)
    planner = ElasticBandPlanner()
    ds = build_perturbation_dataset(inst, planner, 8)
    results = []
    cout = out / "costmap" / inst.name
    cout.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        e = explain_costmap(inst, m, planner, 8, dataset=ds)
        results.append(e)
        save_explanation_figure(inst, e, cout / f"{m}.png")
        print(f"[costmap/{m}] {e.text}")
    save_comparison_grid(inst, results, cout / "comparison.png")

    # semantic: visuo-textual
    print("[semantic] cabinet_room:")
    semantic_main(["--scenario", args.semantic_scenario, "--out", str(out / "semantic")])

    print(f"\nDemo artefacts written under {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
