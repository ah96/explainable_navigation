"""End-to-end visuo-textual explanation on the cabinet_room scenario."""

from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.semantic import VisuoTextualExplainer

from nav_explain_semantic.scenarios import available_scenarios, load_scenario


def test_scenario_available():
    assert "cabinet_room" in available_scenarios()


def test_pipeline_visual_and_textual():
    inst = load_scenario("cabinet_room")
    r = VisuoTextualExplainer(ElasticBandPlanner(), mode="one_off").explain(inst)
    assert len(r.explanation_maps) == len(r.pairs) == 5
    # the closed door blocking the route is the dominant deviation-increasing feature
    assert r.coefficients[(4, "openability")] > 0
    assert any("open the door" in t for t in r.suggestive_text)
