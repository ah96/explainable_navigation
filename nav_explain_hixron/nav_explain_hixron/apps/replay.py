"""``neh_replay``: run a scenario of the publications without a simulator.

Examples::

    neh_replay --scenario library --extroversion 1.0
    neh_replay --scenario library --sweep

The robot follows its global route through the scenario, objects move into its
way, and the explanation loop explains what is around the robot and why it
deviates. One run uses one extroversion. ``--sweep`` runs the eleven levels
from 0 to 1, which is the evaluation of the ICSR 2023 publication, and answers:
how do the number of explanations, their detail and the explained deviations
change with the robot's extroversion?

Everything but the measured runtimes is deterministic. The runtimes go to their
own file for that reason.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nav_explain_core.hixron import ReplayResult, run_replay

from ..results import write_rows
from ..scenarios import DEFAULT_SCENARIO, available_scenarios, load_scenario

#: the extroversion levels of a sweep
SWEEP_LEVELS = tuple(i / 10 for i in range(11))
#: the deterministic columns of a sweep
SWEEP_COLUMNS = ("extroversion", "visual_N", "textual_N", "N_objects", "N_words", "N_deviations")


def _print_row(result: ReplayResult) -> None:
    s = result.summary()
    print(f"[{result.spec_name}] extroversion {result.extroversion:.1f}: "
          f"{s['visual_N']} visual, {s['textual_N']} textual, "
          f"{s['N_objects']:.2f} objects, {s['N_words']:.2f} words, "
          f"{result.n_causes_explained} of {result.n_causes} deviations and failures explained")


def _write_explanations(path: Path, result: ReplayResult) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["tick", "representation", "n_objects", "n_words", "cause", "cause_id", "text"])
        for e in result.explanations:
            writer.writerow([e.tick, e.representation, e.n_objects, e.n_words, e.cause_kind,
                             "" if e.cause_id is None else e.cause_id, e.text])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=available_scenarios())
    p.add_argument("--extroversion", type=float, default=1.0, help="0 (introverted) to 1 (extroverted)")
    p.add_argument("--sweep", action="store_true", help="run the eleven levels from 0 to 1")
    p.add_argument("--out", default="outputs/hixron", help="output folder (default: outputs/hixron)")
    p.add_argument("--no-figures", action="store_true", help="write the tables only")
    args = p.parse_args(argv)
    out = Path(args.out) / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    spec = load_scenario(args.scenario)

    if args.sweep:
        results = [run_replay(spec, level) for level in SWEEP_LEVELS]
        rows = [r.summary() for r in results]
        for r in results:
            _print_row(r)
        write_rows(out / "sweep.csv", rows, SWEEP_COLUMNS)
        write_rows(out / "sweep_runtimes.csv", rows, ("extroversion", "visual_time", "textual_time"))
        if not args.no_figures:
            from ..render import save_evaluation

            save_evaluation(out, rows)
        print(f"Sweep written under {out}/")
        return 0

    result = run_replay(spec, args.extroversion)
    _print_row(result)
    for e in result.explanations:
        if e.first_for_cause:
            print(f"  tick {e.tick} ({e.cause_kind}): {e.text or '(visual only)'}")
    _write_explanations(out / "explanations.csv", result)
    if not args.no_figures:
        from ..render import save_explanation, save_overview

        save_overview(out / "overview.png", spec, result)
        for e in result.explanations:
            if e.first_for_cause:
                save_explanation(out / f"explanation_{e.cause_kind}_tick{e.tick}.png", spec, e)
    print(f"Run written under {out}/")
    return 0 if result.reached_goal else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
