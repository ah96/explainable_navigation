"""End-to-end costmap explanation across all four methods."""

import numpy as np
import pytest

from nav_explain_core import build_perturbation_dataset
from nav_explain_core.planner import ElasticBandPlanner

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


class _CountingPlanner(ElasticBandPlanner):
    """Counts the planner queries, the cost every method is measured in."""

    queries = 0

    def plan(self, request):
        self.queries += 1
        return super().plan(request)


def test_methods_on_a_shared_dataset_need_no_further_planner_query():
    inst = load_scenario("clutter")
    planner = _CountingPlanner()
    ds = build_perturbation_dataset(inst, planner, 6)
    assert planner.queries == len(ds.data)
    for method in ("shap", "anchors", "counterfactual"):
        shared = explain_costmap(inst, method, planner, 6, dataset=ds)
        assert planner.queries == len(ds.data), method
        alone = explain_costmap(inst, method, n_target_segments=6)
        assert shared.summary == alone.summary, method
        assert np.array_equal(shared.image, alone.image), method
        assert shared.local_plan.size  # every method reports the local plan it explains


@pytest.mark.parametrize("method", METHODS)
def test_a_costmap_without_obstacles_has_nothing_to_explain(method):
    """A live costmap is often empty. The node must then say so instead of failing."""
    inst = load_scenario("doorway")
    inst.costmap = np.zeros_like(inst.costmap)
    expl = explain_costmap(inst, method)
    assert expl.text == "no obstacle in the costmap, nothing to explain"
    assert expl.summary == {} and expl.local_plan.size


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        explain_costmap(load_scenario("doorway"), "saliency")
