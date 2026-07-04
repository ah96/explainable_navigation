"""End-to-end costmap explanation across all four methods."""

import numpy as np
import pytest

from nav_explain_costmap.explain import METHODS, explain_costmap
from nav_explain_costmap.scenarios import SCENARIOS, load_scenario


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_build(name):
    inst = load_scenario(name)
    assert inst.costmap.shape == (160, 160)
    assert (inst.costmap > 0).any()


@pytest.mark.parametrize("method", METHODS)
def test_each_method_produces_image(method):
    inst = load_scenario("doorway_blocked")
    expl = explain_costmap(inst, method, n_target_segments=6)
    assert expl.method == method
    assert expl.image.shape == (160, 160, 3)
    assert expl.text  # a non-empty textual explanation


def test_methods_share_dataset_consistency():
    from nav_explain_core import build_perturbation_dataset
    from nav_explain_core.planner import ElasticBandPlanner

    inst = load_scenario("clutter")
    planner = ElasticBandPlanner()
    ds = build_perturbation_dataset(inst, planner, 6)
    shap = explain_costmap(inst, "shap", planner, 6, dataset=ds)
    cf = explain_costmap(inst, "counterfactual", planner, 6, dataset=ds)
    assert shap.segments.shape == cf.segments.shape
