"""The user studies of the ICRA 2024 publication, and their statistics.

``data/icra2024/statistics`` holds the answers of the participants, one file per
group and one row per participant:

* ``study1/try2``: explanation representation. Groups ``v`` (visual), ``vt``
  (visual and textual) and ``t`` (textual), 42 participants each. The
  publication reports ``v`` against ``vt``.
* ``study1/try1``: an earlier collection of the same study, with fewer
  participants and two more questions.
* ``study2``: extroversion. Groups ``i`` (introverted robot) and ``e``
  (extroverted robot), 10 participants each.

Satisfaction is measured with seven statements, answered from 1 (strongly
disagree) to 5 (strongly agree). All answers of a group are pooled, and two
groups are compared with a two-sample t-test, as in the original scripts.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import ttest_ind

from .scenarios import data_root

#: a part of the header of each of the seven statements, in the order of the questionnaire
STATEMENTS = (
    "understand the actions", "satisfied with", "sufficient details", "accurately describes",
    "reliable information", "efficiently", "completely",
)
#: the answer scale, from 1 to 5
SCALE = ("Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree")
#: the comparisons that are computed: name, folder, first group, second group
COMPARISONS = (
    ("representation: visual vs visual-textual", "study1/try2", "v", "vt"),
    ("representation: textual vs visual-textual", "study1/try2", "t", "vt"),
    ("representation: textual vs visual", "study1/try2", "t", "v"),
    ("extroversion: introverted vs extroverted", "study2", "i", "e"),
    ("representation, earlier collection: visual vs visual-textual", "study1/try1", "v", "vt"),
)


@dataclass(frozen=True)
class Group:
    """The answers of one group of participants."""

    name: str
    ages: tuple[float, ...]
    sexes: tuple[str, ...]
    #: one tuple of answers per statement, in the order of ``STATEMENTS``
    answers: tuple[tuple[float, ...], ...]

    @property
    def pooled(self) -> list[float]:
        """All answers to all statements."""
        return [a for statement in self.answers for a in statement]


@dataclass(frozen=True)
class Comparison:
    """Two groups and the test between them."""

    name: str
    first: Group
    second: Group
    t: float
    p: float


def _clean(header: str) -> str:
    """A header without the non-breaking and zero-width spaces of the survey export."""
    return " ".join(header.replace(" ", " ").replace("​", "").split()).lower()


def studies_folder() -> Path:
    return data_root() / "icra2024" / "statistics"


def load_group(path: Path) -> Group:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head = [_clean(h) for h in rows[0]]
    body = rows[1:]
    columns = [next(i for i, h in enumerate(head) if key in h) for key in STATEMENTS]
    return Group(
        name=path.stem,
        ages=tuple(float(r[head.index("age")]) for r in body),
        sexes=tuple(r[head.index("sex")] for r in body),
        answers=tuple(tuple(float(r[c]) for r in body) for c in columns),
    )


def describe(values: list[float]) -> dict[str, float]:
    """Number, mean, standard deviation of a sample, and median."""
    return {"n": len(values), "mean": statistics.fmean(values), "sd": statistics.stdev(values),
            "median": statistics.median(values)}


def compare(name: str, folder: str, first: str, second: str) -> Comparison:
    a = load_group(studies_folder() / folder / f"{first}.csv")
    b = load_group(studies_folder() / folder / f"{second}.csv")
    test = ttest_ind(a.pooled, b.pooled)
    return Comparison(name=name, first=a, second=b, t=float(test.statistic), p=float(test.pvalue))


def all_comparisons() -> list[Comparison]:
    return [compare(*c) for c in COMPARISONS]


def participants(groups: list[Group]) -> dict[str, float]:
    """Age and sex of the participants of several groups together."""
    ages = [a for g in groups for a in g.ages]
    sexes = [s for g in groups for s in g.sexes]
    out = {"n": len(ages), "age_mean": statistics.fmean(ages), "age_sd": statistics.stdev(ages),
           "age_min": min(ages), "age_max": max(ages)}
    for sex in sorted(set(sexes)):
        out[f"percent_{sex.lower()}"] = 100 * sexes.count(sex) / len(sexes)
    return out


def answer_shares(group: Group) -> list[list[float]]:
    """For each statement, the share of the participants per answer of the scale, in percent."""
    return [[100 * statement.count(float(value)) / len(statement) for value in range(1, len(SCALE) + 1)]
            for statement in group.answers]
