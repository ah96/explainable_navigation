"""``nec_figures`` — LIME quality/robustness/runtime figures for a scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nav_explain_core.planner import ElasticBandPlanner

from ..evaluation import reproduce_figures
from ..scenarios import SCENARIOS, load_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reproduce the LIME evaluation figures.")
    p.add_argument("--scenario", default="doorway_blocked", choices=sorted(SCENARIOS))
    p.add_argument("--out", default="outputs/figures")
    p.add_argument("--segments", type=int, default=8)
    args = p.parse_args(argv)

    instance = load_scenario(args.scenario)
    out = Path(args.out) / instance.name
    series = reproduce_figures(instance, ElasticBandPlanner(), out, args.segments)
    print(f"[{instance.name}] quality {series['quality'][0]:.3f}->{series['quality'][-1]:.3f}, "
          f"robustness {series['robustness'][0]:.0f}%->{series['robustness'][-1]:.0f}% ; wrote {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
