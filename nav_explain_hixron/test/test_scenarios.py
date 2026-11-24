"""The scripted runs of the library and the office scenario."""

from pathlib import Path

import numpy as np
import pytest

from nav_explain_core.hixron import run_replay

from nav_explain_hixron.apps.replay import SWEEP_LEVELS
from nav_explain_hixron.scenarios import DEFAULT_SCENARIO, available_scenarios, load_scenario


@pytest.fixture(scope="module")
def library_sweep():
    spec = load_scenario("library")
    return [run_replay(spec, level) for level in SWEEP_LEVELS]


def test_both_scenarios_are_available():
    assert available_scenarios() == ["library", "office"]


def test_the_default_scenario_of_the_node_and_the_launch_file_exists():
    launch = Path(__file__).resolve().parents[1] / "launch" / "hixron_explainer.launch.py"
    assert DEFAULT_SCENARIO in available_scenarios()
    assert f'DeclareLaunchArgument("scenario", default_value="{DEFAULT_SCENARIO}")' in launch.read_text()


def test_in_the_library_two_moved_chairs_make_the_robot_deviate_twice(library_sweep):
    for result in library_sweep:
        assert result.reached_goal
        assert len(result.plans) == 3
        assert result.n_causes == 2
    extrovert = library_sweep[-1]
    causes = [e for e in extrovert.explanations if e.first_for_cause]
    assert [e.cause_id for e in causes] == [8, 9]
    assert all(e.cause_kind == "deviation" for e in causes)


def test_a_more_extroverted_robot_explains_more_often(library_sweep):
    visual = [len(r.explanations) for r in library_sweep]
    assert visual == sorted(visual)
    assert visual[0] < 5 and visual[-1] == library_sweep[-1].n_ticks


def test_a_more_introverted_robot_shows_more_objects(library_sweep):
    objects = [r.summary()["N_objects"] for r in library_sweep]
    assert objects[0] > 2 * objects[-1]


def test_a_rich_text_has_more_words_than_a_poor_one(library_sweep):
    words = {round(10 * r.extroversion): r.summary()["N_words"] for r in library_sweep}
    assert words[3] > 2 * words[5]


def test_the_introverted_robot_stays_silent_while_humans_are_inside_its_threshold(library_sweep):
    assert all(e.representation == "visual" and e.text == "" for e in library_sweep[0].explanations)


def test_a_run_is_deterministic():
    spec = load_scenario("library")
    first, second = run_replay(spec, 0.7), run_replay(spec, 0.7)
    assert first.explanations == second.explanations
    assert all(np.array_equal(a.layer.rgb, b.layer.rgb)
               for a, b in zip(first.explanations, second.explanations))
    assert np.array_equal(first.trace, second.trace)


def test_in_the_office_a_chair_makes_the_robot_deviate_and_a_human_makes_it_stop():
    result = run_replay(load_scenario("office"), 1.0)
    assert result.reached_goal
    causes = [e for e in result.explanations if e.first_for_cause]
    assert [(e.cause_id, e.cause_kind) for e in causes] == [(2, "deviation"), (35, "failure")]
    assert causes[0].text == "The chair_2 was moved."
    assert causes[1].text == "Moving obstruction: 'human_4'. 'Human_4, please move!'"
    # the robot stands still during the failure and drives on afterwards
    stops = sum(1 for a, b in zip(result.trace, result.trace[1:]) if tuple(a) == tuple(b))
    assert stops == 23
