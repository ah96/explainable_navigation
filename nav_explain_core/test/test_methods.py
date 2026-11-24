"""Tests for the four costmap explanation methods over a shared dataset."""

import numpy as np
import pytest

from nav_explain_core import CostmapInfo, Pose2D, ScenarioInstance, build_perturbation_dataset
from nav_explain_core.anchors import AnchorImageExplainer, find_anchor
from nav_explain_core.counterfactual import counterfactual
from nav_explain_core.lime import LimeCostmapExplainer
from nav_explain_core.planner import ElasticBandPlanner
from nav_explain_core.pointcloud import POINT_STEP, pack_xyzrgb
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
    assert dataset.local_plan.shape[1] == 2 and len(dataset.local_plan) > 2


def test_one_off_dataset_matches_the_rows_of_the_full_one(dataset):
    """The planner is deterministic, so both modes give the same deviation per perturbation."""
    one_off = build_perturbation_dataset(_blocked_instance(), ElasticBandPlanner(),
                                         n_target_segments=6, mode="one_off")
    full = {tuple(row): dev for row, dev in zip(dataset.data, dataset.deviations)}
    assert [full[tuple(row)] for row in one_off.data] == list(one_off.deviations)


def test_lime_runs():
    one_off = build_perturbation_dataset(_blocked_instance(), ElasticBandPlanner(),
                                         n_target_segments=6, mode="one_off")
    r = LimeCostmapExplainer().explain(one_off)
    assert r.explanation_image.shape[2] == 3
    assert r.n_perturbations == one_off.n_features
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


def test_anchor_of_a_scene_without_features_is_empty():
    anchor = find_anchor(np.ones((1, 1), dtype=np.int8), np.array([1]))
    assert anchor.features == [] and anchor.coverage == 1.0


def test_anchor_image_deterministic(dataset):
    a = AnchorImageExplainer().explain(dataset)
    b = AnchorImageExplainer().explain(dataset)
    assert a.anchor_segments == b.anchor_segments and a.anchor_segments
    assert np.array_equal(a.image, b.image)
    assert a.explanation.precision >= 0.9  # the default threshold of the search


def test_pointcloud_payload_has_one_point_per_cell():
    info = CostmapInfo(resolution=0.5, origin_x=1.0, origin_y=2.0, width=3, height=2)
    rgb = np.zeros((2, 3, 3))
    rgb[1, 2] = (1.0, 0.0, 0.0)  # one red cell, the others black
    payload, n = pack_xyzrgb(rgb, info)
    assert n == 6 and len(payload) == 6 * POINT_STEP
    payload, n = pack_xyzrgb(rgb, info, skip_black=True)
    assert n == 1
    point = np.frombuffer(payload, dtype=[("xyz", "<f4", 3), ("rgb", "<u4")])[0]
    assert np.allclose(point["xyz"], (2.25, 2.75, 0.0))  # centre of cell (col 2, row 1)
    assert point["rgb"] == 0xFF0000


def test_counterfactual_reduces_deviation(dataset):
    cf = counterfactual(dataset, target_ratio=0.4)
    # removing the counterfactual set yields a deviation below the instance's
    assert cf.resulting_deviation < cf.instance_deviation
    if cf.found:
        assert cf.resulting_deviation <= cf.target + 1e-9
