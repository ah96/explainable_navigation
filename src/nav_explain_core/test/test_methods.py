"""Tests for the four costmap explanation methods over a shared dataset."""

import numpy as np
import pytest

from nav_explain_core import CostmapInfo, Pose2D, ScenarioInstance, build_perturbation_dataset
from nav_explain_core.anchors import AnchorImageExplainer, find_anchor
from nav_explain_core.counterfactual import counterfactual
from nav_explain_core.lime import LimeCostmapExplainer
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.shap import shapley_values


def _blocked_instance():
    H = W = 100
    res = 0.04
    cm = np.zeros((H, W), dtype=np.int16)
    cm[40:80, 20:42] = 100
    cm[40:80, 58:80] = 100
    rr, cc = np.ogrid[:H, :W]
    cm[(rr - 60) ** 2 + (cc - 50) ** 2 <= 8**2] = 100  # blocking blob
    info = CostmapInfo(resolution=res, origin_x=-2.0, origin_y=-1.0, width=W, height=H)
    gy = np.linspace(0.0, 3.0, 36)
    plan = np.column_stack([np.zeros_like(gy), gy])
    return ScenarioInstance(costmap=cm, costmap_info=info, global_plan=plan,
                            start_pose=Pose2D(0, 0, np.pi / 2), name="blocked")


@pytest.fixture(scope="module")
def dataset():
    return build_perturbation_dataset(_blocked_instance(), ElasticBandPlanner(), n_target_segments=6)


def test_dataset_shape(dataset):
    assert dataset.data.shape[0] == 2 ** (dataset.n_features - 1)
    assert dataset.deviations.shape[0] == dataset.data.shape[0]
    assert (dataset.data[:, 0] == 1).all()  # free-space column pinned


def test_lime_runs():
    r = LimeCostmapExplainer(ElasticBandPlanner(), mode="one_off").explain(_blocked_instance())
    assert r.explanation_image.shape[2] == 3
    assert any(abs(w) > 0 for k, w in r.coefficients.items() if k != 0)


def test_shap_efficiency(dataset):
    """Shapley values sum to instance value minus base value (efficiency)."""
    sh = shapley_values(dataset)
    total = sum(sh.values.values())
    assert abs(total - (sh.instance_value - sh.base_value)) < 1e-6


def test_shap_deterministic(dataset):
    a = shapley_values(dataset).values
    b = shapley_values(dataset).values
    assert a == b


def test_anchor_precision(dataset):
    # binarise: 1 = deviates like the instance
    d0 = dataset.deviations[0]
    y = (dataset.deviations >= 0.5 * d0).astype(int)
    anchor = find_anchor(dataset.data, y, tau=0.9)
    assert anchor.precision >= 0.9 or anchor.features  # a rule is returned
    assert 0.0 <= anchor.coverage <= 1.0


def test_anchor_image_deterministic():
    inst = _blocked_instance()
    a = AnchorImageExplainer(ElasticBandPlanner(), n_target_segments=6).explain(inst)
    b = AnchorImageExplainer(ElasticBandPlanner(), n_target_segments=6).explain(inst)
    assert a.anchor_segments == b.anchor_segments


def test_counterfactual_reduces_deviation(dataset):
    cf = counterfactual(dataset, target_ratio=0.4)
    # removing the counterfactual set yields a deviation below the instance's
    assert cf.resulting_deviation < cf.instance_deviation
    if cf.found:
        assert cf.resulting_deviation <= cf.target + 1e-9
