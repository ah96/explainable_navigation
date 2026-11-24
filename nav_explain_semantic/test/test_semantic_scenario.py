"""End-to-end visuo-textual explanation on the cabinet_room scenario."""

from pathlib import Path

from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.semantic import VisuoTextualExplainer

from nav_explain_semantic.scenarios import DEFAULT_SCENARIO, available_scenarios, load_scenario


def test_scenario_available():
    assert "cabinet_room" in available_scenarios()


def test_the_default_scenario_of_the_node_and_the_launch_file_exists():
    """The node once started with the name of a scenario that did not exist."""
    launch = Path(__file__).resolve().parents[1] / "launch" / "semantic_explainer.launch.py"
    assert DEFAULT_SCENARIO in available_scenarios()
    assert f'DeclareLaunchArgument("scenario", default_value="{DEFAULT_SCENARIO}")' in launch.read_text()


def test_pipeline_visual_and_textual():
    inst = load_scenario("cabinet_room")
    r = VisuoTextualExplainer(ElasticBandPlanner(), mode="one_off").explain(inst)
    assert len(r.explanation_maps) == len(r.pairs) == 5
    # the closed door blocking the route is the dominant deviation-increasing feature
    assert r.coefficients[(4, "openability")] > 0
    assert any("open the door" in t for t in r.suggestive_text)
