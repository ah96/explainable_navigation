"""``neb_demo``: run every explanation method on the built-in scenarios.

Example::

    neb_demo --out outputs/demo

Runs the four costmap methods (LIME, SHAP, Anchors, counterfactual) on a costmap
scenario, the visuo-textual method on the semantic scenario and the HiXRoN
explanations on the scripted run of a scenario, once for an extroverted and
once for an introverted robot. It then recomputes the recorded results of the
publications from their files. All artefacts go under one output folder. It is
the one entry point that exercises the whole toolkit, without a robot or a
simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nav_explain_costmap.apps.explain_scenario import main as costmap_main
from nav_explain_costmap.scenarios import SCENARIOS
from nav_explain_hixron.apps.replay import main as replay_main
from nav_explain_hixron.apps.results import main as results_main
from nav_explain_hixron.scenarios import DEFAULT_SCENARIO as DEFAULT_HIXRON_SCENARIO
from nav_explain_hixron.scenarios import available_scenarios as available_hixron_scenarios
from nav_explain_semantic.apps.explain_scenario import main as semantic_main
from nav_explain_semantic.scenarios import DEFAULT_SCENARIO, available_scenarios


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run every explanation method on the built-in scenarios.")
    p.add_argument("--costmap-scenario", default="doorway_blocked", choices=sorted(SCENARIOS))
    p.add_argument("--semantic-scenario", default=DEFAULT_SCENARIO, choices=available_scenarios())
    p.add_argument("--hixron-scenario", default=DEFAULT_HIXRON_SCENARIO,
                   choices=available_hixron_scenarios())
    p.add_argument("--out", default="outputs/demo", help="output folder (default: outputs/demo)")
    args = p.parse_args(argv)
    out = Path(args.out)

    status = costmap_main(["--scenario", args.costmap_scenario, "--method", "all",
                           "--out", str(out / "costmap")])
    status = status or semantic_main(["--scenario", args.semantic_scenario,
                                      "--out", str(out / "semantic")])
    for extroversion in ("1.0", "0.0"):
        status = status or replay_main(["--scenario", args.hixron_scenario, "--extroversion",
                                        extroversion, "--out", str(out / f"hixron_{extroversion}")])
    status = status or results_main(["--out", str(out / "results")])
    if status == 0:
        print(f"\nDemo artefacts written under {out}/")
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
