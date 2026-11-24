"""The explanation loop that depends on the robot's personality."""

import math

import numpy as np
import pytest

from nav_explain_core import Pose2D
from nav_explain_core.hixron import (
    DEVIATION_THRESHOLD_M, ExplanationPolicy, Replay, deviation_text, explanation_layer,
    neighbourhood_text, plan_deviation, plan_route,
)
from nav_explain_core.hixron.layer import FREE_GREY, HIGHLIGHT_COLOUR, OBJECT_GREY, RING_COLOURS
from nav_explain_core.hixron.text import side_of_robot
from nav_explain_core.semantic.ontology import ObjectEntry, Ontology
from nav_explain_core.semantic.semantic_map import SemanticMapInfo, build_semantic_map

INFO = SemanticMapInfo(resolution=0.05, origin_x=0.0, origin_y=0.0, width=200, height=200)


def _obj(oid: int, label: str, cx: float, cy: float, dx: float = 0.4, dy: float = 0.4) -> ObjectEntry:
    return ObjectEntry(id=oid, gazebo_label=f"{label}_{oid}", label=label, cx=cx, cy=cy, dx=dx, dy=dy,
                       movability=True, openability=False)


def test_the_threshold_spans_the_social_zone():
    assert ExplanationPolicy(0.0).representation_threshold == pytest.approx(3.6)
    assert ExplanationPolicy(1.0).representation_threshold == pytest.approx(1.2)


def test_timing_and_duration_fall_linearly_from_twenty_ticks_to_none():
    assert [ExplanationPolicy(p / 10).timing for p in range(11)] == [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    assert ExplanationPolicy(0.3).duration == ExplanationPolicy(0.3).timing


def test_the_text_is_rich_below_one_half_and_poor_from_there_on():
    assert ExplanationPolicy(0.4).detail_level == "rich"
    assert ExplanationPolicy(0.5).detail_level == "poor"


def test_a_human_inside_the_threshold_makes_the_explanation_visual_only():
    policy = ExplanationPolicy(1.0)
    assert policy.representation(1.0) == "visual"
    assert policy.representation(2.0) == "visual+textual"
    assert policy.representation(None) == "visual+textual"


def test_an_extroversion_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        ExplanationPolicy(1.5)


def test_the_layer_colours_near_objects_yellow_and_keeps_walls_and_far_objects_grey():
    ontology = Ontology([_obj(1, "chair", 5.0, 5.0), _obj(2, "wall", 5.0, 6.0, 4.0, 0.2),
                         _obj(3, "chair", 9.0, 9.0)])
    labels = build_semantic_map(ontology, INFO)
    layer = explanation_layer(labels, INFO, (5.0, 5.0), 2.0, wall_ids=frozenset({2}))
    assert layer.object_ids == (1, 2)
    assert tuple(layer.rgb[100, 100]) == RING_COLOURS[0]
    assert tuple(layer.rgb[120, 100]) == (OBJECT_GREY,) * 3      # the wall, inside the window
    assert tuple(layer.rgb[180, 180]) == (OBJECT_GREY,) * 3      # a chair outside the window
    assert tuple(layer.rgb[60, 60]) == (FREE_GREY,) * 3


def test_the_cause_of_a_deviation_is_red_also_outside_the_window():
    labels = build_semantic_map(Ontology([_obj(3, "chair", 9.0, 9.0)]), INFO)
    layer = explanation_layer(labels, INFO, (5.0, 5.0), 2.0, highlight_id=3)
    assert tuple(layer.rgb[180, 180]) == HIGHLIGHT_COLOUR


def test_objects_are_placed_relative_to_the_heading_of_the_robot():
    robot = Pose2D(0.0, 0.0, math.pi / 2)  # facing +y
    assert side_of_robot(robot, (0.0, 1.0)) == "front"
    assert side_of_robot(robot, (-1.0, 0.0)) == "left"
    assert side_of_robot(robot, (1.0, 0.0)) == "right"
    assert side_of_robot(robot, (0.0, -1.0)) == "back"


def test_a_poor_text_lists_and_a_rich_text_places_the_objects():
    objects = [("table", "left"), ("bookshelf", "right")]
    assert neighbourhood_text(objects, "poor") == "Around me are table and bookshelf."
    assert neighbourhood_text(objects, "rich") == \
        "I am passing by table to my left and bookshelf to my right."
    assert neighbourhood_text([("chair", "front")], "rich") == "In front of me is chair."
    assert neighbourhood_text([], "rich") == "There are no objects close to me."


def test_the_deviation_text_names_the_moved_object():
    assert deviation_text("chair", "left", "poor") == "The chair was moved."
    assert deviation_text("chair", "left", "rich") == \
        "I am deviating because the chair to my left, was moved."
    assert "in front of me" in deviation_text("chair", "front", "rich")


def test_two_plans_deviate_when_their_middle_points_lie_apart():
    straight = np.column_stack([np.linspace(0, 10, 11), np.zeros(11)])
    detour = straight + (0.0, 2.0)
    assert plan_deviation(straight, straight) == 0.0
    assert plan_deviation(straight, detour) == pytest.approx(2.0)
    assert plan_deviation(straight, detour) > DEVIATION_THRESHOLD_M


def test_a_gap_as_wide_as_the_robot_is_closed_and_the_route_goes_around():
    # two boxes leave a gap of 0.55 m at x = 5, twice the inflation radius
    narrow = [_obj(1, "shelf", 3.1125, 5.0, 3.225, 0.4), _obj(2, "shelf", 6.8875, 5.0, 3.225, 0.4)]
    route = plan_route(build_semantic_map(Ontology(narrow), INFO), INFO, (5.0, 2.0), (5.0, 8.0))
    crossing = route[np.argmin(np.abs(route[:, 1] - 5.0))]
    assert abs(crossing[0] - 5.0) > 3.0  # around an end of a box, not through the gap
    # a gap of 0.95 m is open
    wide = [_obj(1, "shelf", 3.0125, 5.0, 3.025, 0.4), _obj(2, "shelf", 6.9875, 5.0, 3.025, 0.4)]
    route = plan_route(build_semantic_map(Ontology(wide), INFO), INFO, (5.0, 2.0), (5.0, 8.0))
    assert np.allclose(route[:, 0], 5.0, atol=0.3)


def test_a_route_starts_also_from_a_pose_right_next_to_an_object():
    labels = build_semantic_map(Ontology([_obj(1, "table", 5.0, 5.0, 1.0, 1.0)]), INFO)
    assert plan_route(labels, INFO, (5.0, 4.3), (5.0, 2.0)) is not None
    assert plan_route(labels, INFO, (5.0, 5.0), (5.0, 2.0)) is None  # inside the table


def test_an_extroverted_robot_explains_on_every_tick_and_an_introverted_one_waits():
    ontology = Ontology([_obj(1, "chair", 5.0, 5.0)])
    robot = Pose2D(5.0, 4.0, math.pi / 2)
    extrovert = Replay(ontology, INFO, ExplanationPolicy(1.0))
    assert all(extrovert.tick(robot) is not None for _ in range(5))
    introvert = Replay(ontology, INFO, ExplanationPolicy(0.0))
    given = [tick for tick in range(100) if introvert.tick(robot) is not None]
    assert given == [20, 60]  # formed at 0 and 40, each given 20 ticks later


def test_a_deviation_after_a_moved_object_is_explained_once_as_new():
    ontology = Ontology([_obj(1, "chair", 5.0, 5.0)])
    robot = Pose2D(5.0, 4.0, math.pi / 2)
    replay = Replay(ontology, INFO, ExplanationPolicy(1.0))
    straight = np.column_stack([np.full(11, 5.0), np.linspace(4, 9, 11)])
    replay.set_plan(straight)
    replay.move_object(1, (5.0, 6.5))
    replay.set_plan(straight + (2.0, 0.0))
    given = [replay.tick(robot) for _ in range(15)]
    assert given[0].cause_id == 1 and given[0].cause_kind == "deviation"
    assert given[0].text == "The chair was moved."
    assert [e.first_for_cause for e in given].count(True) == 1
    assert given[11].cause_id == 1 and given[12].cause_id is None  # shown for twelve explanations
    assert replay.n_causes == replay.n_causes_explained == 1


def test_a_new_plan_without_a_moved_object_is_no_deviation():
    replay = Replay(Ontology([_obj(1, "chair", 5.0, 5.0)]), INFO, ExplanationPolicy(1.0))
    straight = np.column_stack([np.full(11, 5.0), np.linspace(4, 9, 11)])
    replay.set_plan(straight)
    replay.set_plan(straight + (2.0, 0.0))
    assert replay.n_causes == 0
