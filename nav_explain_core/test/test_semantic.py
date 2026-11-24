"""Semantic layer: QSR (single-cross + ternary), ontology, semantic map, affordances."""

import math

import numpy as np
import pytest

from nav_explain_core.geometry import Pose2D
from nav_explain_core.semantic.affordance import build_affordance_perturbations, object_affordance_pairs
from nav_explain_core.semantic.ontology import ObjectEntry, Ontology
from nav_explain_core.semantic.qsr import single_cross_quadrant, ternary_direction, ternary_qsr
from nav_explain_core.semantic.semantic_map import SemanticMapInfo, build_semantic_map, inflate
from nav_explain_core.semantic.templater import descriptive, suggestive


def _ontology():
    return Ontology([
        ObjectEntry(1, "cabinet", "cabinet", 2.0, 1.0, 1.0, 0.6, True, True),
        ObjectEntry(2, "door", "door", 3.0, 0.0, 0.15, 1.0, False, True),
        ObjectEntry(3, "wall", "wall", 0.0, 0.0, 0.15, 4.0, False, False),
    ])


def _info():
    return SemanticMapInfo(resolution=0.05, origin_x=-1.0, origin_y=-2.0, width=120, height=120)


@pytest.mark.parametrize("angle,expected", [
    (0.1, "left/front"), (math.pi / 2 + 0.1, "left/back"),
    (-0.1, "right/front"), (-math.pi / 2 - 0.1, "right/back"),
])
def test_single_cross(angle, expected):
    assert single_cross_quadrant(angle) == expected


@pytest.mark.parametrize("angle,expected", [
    (0.0, "front"), (math.pi / 2, "left"), (math.pi, "back"), (-math.pi / 2, "right"),
])
def test_ternary_direction(angle, expected):
    assert ternary_direction(angle) == expected


def test_ternary_qsr_distance():
    r = Pose2D(0, 0, 0)
    assert ternary_qsr(r, (1.0, 0.0), 2.0) == "close-front"
    assert ternary_qsr(r, (5.0, 0.0), 2.0) == "distant-front"


def test_ontology_affordances():
    ont = _ontology()
    assert ont.by_id(1).affordances == ["movability", "openability"]
    assert ont.by_id(2).affordances == ["openability"]
    assert ont.by_id(3).affordances == []


def test_object_affordance_pairs():
    pairs = object_affordance_pairs(_ontology())
    assert (1, "movability") in pairs and (1, "openability") in pairs and (2, "openability") in pairs
    assert len(pairs) == 3


def test_semantic_map_and_inflate():
    ont, info = _ontology(), _info()
    smap = build_semantic_map(ont, info)
    assert (smap == 1).any()
    assert (inflate(smap, info, 0.2) == 1).sum() > (smap == 1).sum()


def test_affordance_perturbations_shape():
    ont, info = _ontology(), _info()
    smap = inflate(build_semantic_map(ont, info), info)
    pairs = object_affordance_pairs(ont, {int(v) for v in np.unique(smap) if v})
    data, costmaps, labels = build_affordance_perturbations(smap, info, ont, pairs, "one_off")
    assert data.shape == (len(pairs) + 1, len(pairs) + 1)
    assert len(costmaps) == len(pairs) + 1


def test_affordances_change_the_costmap_as_described():
    ont, info = _ontology(), _info()
    smap = build_semantic_map(ont, info)
    pairs = [(1, "movability"), (1, "openability"), (2, "openability")]
    _, (base, moved, opened, door_open), _ = build_affordance_perturbations(smap, info, ont, pairs)
    assert (base[smap > 0] > 0).all()
    assert (moved[smap == 1] == 0).all()                       # the cabinet is moved away
    assert (opened > 0).sum() > (base > 0).sum()               # its open doors take more room
    assert (door_open[smap == 2] == 0).all()                   # the open door is passable
    assert (door_open[smap == 1] > 0).all()                    # and nothing else changes


def test_templater():
    assert suggestive("door", "right/front", "openability") == \
        "Dear human, please open the door right/front of me."
    assert "deviate because of the chair" in descriptive("chair", "left/front", increases=True)
