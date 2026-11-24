"""``neh_results``: the recorded results of the publications, recomputed from their files.

Examples::

    neh_results
    neh_results --machine labpc --out outputs/results

It answers two questions. For the ICSR 2023 experiments: how many explanations,
with how much detail and in what time, did the robot produce at each
extroversion level? For the ICRA 2024 user studies: how satisfied were the
groups, and do they differ?

Nothing is simulated here. The numbers come from the recorded files in the
data folder of this package.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ..results import COLUMNS, MACHINES, REPORTED_MACHINE, aggregate, mean_runtimes, write_rows
from ..studies import STATEMENTS, all_comparisons, describe, load_group, participants, studies_folder


def _experiments(machine: str, out: Path, figures: bool) -> None:
    rows = aggregate(machine)
    print(f"ICSR 2023 experiments, machine '{machine}':")
    print("  " + "  ".join(f"{c:>12}" for c in COLUMNS))
    for row in rows:
        print("  " + "  ".join(f"{row[c]:12.2f}" for c in COLUMNS))
    visual, textual = mean_runtimes(rows)
    print(f"  mean runtime: visual {visual:.2f} ms, textual {textual:.2f} ms, "
          f"together {visual + textual:.2f} ms")
    write_rows(out / f"icsr2023_{machine}.csv", rows)
    if figures:
        from ..render import save_evaluation

        save_evaluation(out / f"icsr2023_{machine}", rows)


def _studies(out: Path, figures: bool) -> None:
    comparisons = all_comparisons()
    print("ICRA 2024 user studies, all answers to the seven statements pooled per group:")
    with open(out / "icra2024_studies.csv", "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["comparison", "group_1", "n_1", "mean_1", "sd_1", "group_2", "n_2", "mean_2",
                         "sd_2", "t", "p"])
        for c in comparisons:
            a, b = describe(c.first.pooled), describe(c.second.pooled)
            print(f"  {c.name}: {c.first.name} M={a['mean']:.2f} SD={a['sd']:.2f}, "
                  f"{c.second.name} M={b['mean']:.2f} SD={b['sd']:.2f}, t={c.t:.2f}, p={c.p:.3g}")
            writer.writerow([c.name, c.first.name, len(c.first.ages), f"{a['mean']:.4f}", f"{a['sd']:.4f}",
                             c.second.name, len(c.second.ages), f"{b['mean']:.4f}", f"{b['sd']:.4f}",
                             f"{c.t:.4f}", f"{c.p:.4g}"])
    for name, folder, groups in (("representation (v, vt)", "study1/try2", ("v", "vt")),
                                 ("extroversion (i, e)", "study2", ("i", "e"))):
        who = participants([load_group(studies_folder() / folder / f"{g}.csv") for g in groups])
        sexes = ", ".join(f"{k[8:]} {v:.2f} %" for k, v in who.items() if k.startswith("percent_"))
        print(f"  participants, {name}: n={who['n']}, age {who['age_min']:.0f} to {who['age_max']:.0f}, "
              f"M={who['age_mean']:.2f} SD={who['age_sd']:.2f}, {sexes}")
    if figures:
        from ..render import save_answers

        first = comparisons[0]
        save_answers(out / "icra2024_study1_answers.png",
                     [("Group 1 (visual)", first.first), ("Group 2 (visual-textual)", first.second)],
                     STATEMENTS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default=REPORTED_MACHINE, choices=MACHINES,
                   help="whose recorded experiment runs to aggregate (default: the reported one)")
    p.add_argument("--out", default="outputs/results", help="output folder (default: outputs/results)")
    p.add_argument("--no-figures", action="store_true", help="write the tables only")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _experiments(args.machine, out, not args.no_figures)
    _studies(out, not args.no_figures)
    print(f"Tables written under {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
