"""Trajectory-distance metrics."""

import numpy as np

from nav_explain_core import trajectory as tj


def test_deviation_identical_zero():
    plan = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    assert tj.deviation(plan, plan) == 0.0


def test_deviation_offset_known():
    glob = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    local = np.array([[0.5, 0.0], [0.5, 1.0]])
    assert abs(tj.deviation(local, glob) - 1.0) < 1e-9


def test_deviation_empty():
    glob = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert tj.deviation(np.empty((0, 2)), glob) == 0.0


def test_hausdorff_symmetric_and_nonneg():
    a = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    b = np.array([[1.0, 0.0], [1.0, 2.0]])
    h = tj.hausdorff(a, b)
    assert h == tj.hausdorff(b, a)          # symmetric
    assert h is not None and h >= 1.0        # bug-fixed: returns a value


def test_metrics_identical_are_trivial():
    t = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert tj.sspd(t, t) == 0.0
    assert tj.dtw(t, t) == 0.0
    assert tj.discrete_frechet(t, t) == 0.0
    assert tj.hausdorff(t, t) == 0.0


def test_dtw_and_frechet_positive_for_shifted():
    a = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    b = a + np.array([1.0, 0.0])
    assert tj.dtw(a, b) > 0
    assert abs(tj.discrete_frechet(a, b) - 1.0) < 1e-9
