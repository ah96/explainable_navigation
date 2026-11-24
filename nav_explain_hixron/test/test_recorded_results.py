"""The numbers the publications report, recomputed from the recorded files."""

import csv
import re

import pytest

from nav_explain_hixron.results import LEVELS, MACHINES, aggregate, mean_runtimes, results_folder
from nav_explain_hixron.scenarios import data_root
from nav_explain_hixron.studies import (
    all_comparisons, compare, describe, load_group, participants, studies_folder,
)


def test_every_machine_has_a_file_per_extroversion_level():
    for machine in MACHINES:
        assert len(aggregate(machine)) == len(LEVELS) == 11


def test_the_reported_runtimes_are_the_means_of_the_laptop_run():
    visual, textual = mean_runtimes(aggregate("laptop"))
    assert round(visual, 2) == 348.90
    assert round(visual + textual, 2) == 349.24
    assert textual == pytest.approx(0.34, abs=0.01)


def test_cells_of_the_aggregation_follow_from_the_recorded_rows():
    with open(results_folder("laptop") / "10.csv", newline="") as fh:
        recorded = list(csv.DictReader(fh))
    row = aggregate("laptop")[-1]
    assert row["visual_N"] == float(recorded[-1]["visual_N"]) == 177
    assert row["textual_N"] == 141
    assert row["N_objects"] == pytest.approx(sum(float(r["N_objects"]) for r in recorded) / 177)


def test_more_extroverted_robots_explained_more_often_in_the_recorded_run():
    visual = [r["visual_N"] for r in aggregate("laptop")]
    assert visual == sorted(visual)


def test_only_the_totally_introverted_robot_missed_a_deviation_in_the_recorded_run():
    assert [r["N_deviations"] for r in aggregate("laptop")] == [1] + [2] * 10


def test_study_on_representation_reproduces_the_reported_statistics():
    c = compare("", "study1/try2", "v", "vt")
    visual, both = describe(c.first.pooled), describe(c.second.pooled)
    assert len(c.first.ages) == len(c.second.ages) == 42
    assert round(visual["mean"], 2) == 4.24
    assert visual["sd"] == pytest.approx(0.99, abs=0.01)  # 0.9989, printed as 0.99 in the publication
    assert (round(both["mean"], 2), round(both["sd"], 2)) == (4.62, 0.58)
    assert round(c.t, 1) == -5.6
    assert c.p == pytest.approx(3.28e-08, rel=0.01)
    who = participants([c.first, c.second])
    assert (who["n"], who["age_min"], who["age_max"]) == (84, 19, 40)
    assert (round(who["age_mean"], 2), round(who["age_sd"], 2)) == (22.75, 4.23)


def test_study_on_extroversion_reproduces_the_reported_means():
    c = compare("", "study2", "i", "e")
    introvert, extrovert = describe(c.first.pooled), describe(c.second.pooled)
    assert (round(introvert["mean"], 2), round(introvert["sd"], 2)) == (3.47, 1.21)
    assert round(extrovert["mean"], 2) == 3.77
    assert c.p > 0.05  # no significant difference, as reported
    who = participants([c.first, c.second])
    assert (who["n"], who["age_min"], who["age_max"]) == (20, 21, 40)
    assert (round(who["age_mean"], 2), round(who["age_sd"], 2)) == (27.45, 5.47)
    assert who["percent_male"] == 60.0


def test_every_comparison_can_be_computed():
    assert len(all_comparisons()) == 5


def test_the_study_files_hold_no_free_text_about_a_participant():
    for path in sorted(studies_folder().rglob("*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            head = " ".join(next(csv.reader(fh))).lower()
        for column in ("nationality", "country", "profession", "job title", "education"):
            assert column not in head, (path.name, column)
        assert len(load_group(path).answers) == 7


def test_every_action_of_a_recorded_plan_is_an_action_of_the_planning_domain():
    planning = data_root() / "icra2024" / "planning"
    domain = (planning / "domain_icra.pddl").read_text()
    actions = set(re.findall(r"\(:durative-action\s+(\S+)", domain))
    plans = sorted(planning.glob("*.dot"))
    assert len(plans) == 4 and len(actions) >= 12
    for plan in plans:
        used = set(re.findall(r'label="(\w+)_(?:start|end)\n', plan.read_text()))
        assert used and used <= actions, (plan.name, used - actions)
        kind, personality = plan.stem.split("_")[1], plan.stem.split("_")[0]
        assert f"explain_{kind}_{personality}" in used
