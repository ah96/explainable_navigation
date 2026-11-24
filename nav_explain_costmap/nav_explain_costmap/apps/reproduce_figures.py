"""``nec_figures``: LIME quality, robustness and runtime figure for a scenario.

Example::

    nec_figures --scenario doorway_blocked --out outputs/figures

Writes ``evaluation.png`` to ``<out>/<scenario>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nav_explain_core import DEFAULT_SEGMENTS
from nav_explain_core.planner import ElasticBandPlanner

from ..evaluation import reproduce_figures
from ..scenarios import SCENARIOS, load_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate LIME on a costmap scenario.")
    p.add_argument("--scenario", default="doorway_blocked", choices=sorted(SCENARIOS))
    p.add_argument("--out", default="outputs/figures", help="output folder")
    p.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS,
                   help="number of obstacle segments, i.e. interpretable features")
    args = p.parse_args(argv)

    instance = load_scenario(args.scenario)
    out = Path(args.out) / instance.name
    series = reproduce_figures(instance, ElasticBandPlanner(), out, args.segments)
    print(f"[{instance.name}] quality {series['quality'][0]:.3f}->{series['quality'][-1]:.3f}, "
          f"robustness {series['robustness'][0]:.0f}%->{series['robustness'][-1]:.0f}% ; wrote {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
