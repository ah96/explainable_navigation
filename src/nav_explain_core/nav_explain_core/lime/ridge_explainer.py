"""The weighted Ridge surrogate at the heart of LIME.

Given a perturbation matrix (binary, one row per perturbation, ``data[0]`` the
unperturbed instance) and a target vector (here: the local-plan deviation for
each perturbation), fit a kernel-weighted linear model whose coefficients are
the per-segment importances.

This is a faithful, de-ROS'd port of the original
``lime_base.py::LimeBase.explain_instance_with_data`` and its feature selection.
``Ridge(alpha=1)`` weighted by the cosine/exponential kernel, with a fixed
``random_state`` for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from .kernel import DEFAULT_KERNEL_WIDTH, cosine_distances, exp_kernel


@dataclass
class Explanation:
    """Result of fitting the local surrogate.

    Attributes:
        coef: ``segment_label -> coefficient``. Positive => the segment increases
            the deviation (coloured green in the paper), negative => decreases it
            (red).
        order: segment labels sorted by descending ``|coef|`` (most important
            first).
        intercept: surrogate intercept.
        score: weighted R^2 of the surrogate on the perturbation set.
        local_pred: surrogate prediction on the original instance.
    """

    coef: dict[int, float]
    order: list[int]
    intercept: float
    score: float
    local_pred: float = 0.0
    used_features: list[int] = field(default_factory=list)


class RidgeExplainer:
    """Fits a kernel-weighted Ridge surrogate and extracts segment importances."""

    def __init__(
        self,
        kernel_width: float = DEFAULT_KERNEL_WIDTH,
        alpha: float = 1.0,
        feature_selection: str = "auto",
        random_state: int | None = 42,
    ):
        self.kernel_width = kernel_width
        self.alpha = alpha
        self.feature_selection = feature_selection
        self.random_state = check_random_state(random_state)

    # -- feature selection (ported from LimeBase) ------------------------------
    def _forward_selection(self, data, labels, weights, num_features) -> np.ndarray:
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used: list[int] = []
        for _ in range(min(num_features, data.shape[1])):
            best, best_score = 0, -np.inf
            for feature in range(data.shape[1]):
                if feature in used:
                    continue
                cols = used + [feature]
                clf.fit(data[:, cols], labels, sample_weight=weights)
                score = clf.score(data[:, cols], labels, sample_weight=weights)
                if score > best_score:
                    best, best_score = feature, score
            used.append(best)
        return np.array(used)

    def _highest_weights(self, data, labels, weights, num_features) -> np.ndarray:
        clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
        clf.fit(data, labels, sample_weight=weights)
        weighted = clf.coef_ * data[0]
        ranked = sorted(
            range(data.shape[1]), key=lambda i: abs(weighted[i]), reverse=True
        )
        return np.array(ranked[:num_features])

    def _select_features(self, data, labels, weights, num_features) -> np.ndarray:
        method = self.feature_selection
        if method == "none":
            return np.arange(data.shape[1])
        if method == "auto":
            method = "forward_selection" if num_features <= 6 else "highest_weights"
        if method == "forward_selection":
            return self._forward_selection(data, labels, weights, num_features)
        if method == "highest_weights":
            return self._highest_weights(data, labels, weights, num_features)
        raise ValueError(f"unknown feature_selection: {self.feature_selection}")

    # -- main entry point ------------------------------------------------------
    def explain(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        segment_labels: np.ndarray | None = None,
        num_features: int | None = None,
    ) -> Explanation:
        """Fit the surrogate.

        Args:
            data: ``(n_perturbations, n_features)`` binary matrix; ``data[0]`` is
                the unperturbed instance.
            targets: ``(n_perturbations,)`` regression target (deviation).
            segment_labels: maps each column to a segment label. Defaults to the
                column index (0 = free space, 1..K obstacles).
            num_features: max features in the explanation. Defaults to all.
        """
        data = np.asarray(data, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if segment_labels is None:
            segment_labels = np.arange(data.shape[1])
        if num_features is None:
            num_features = data.shape[1]

        weights = exp_kernel(cosine_distances(data), self.kernel_width)
        used = self._select_features(data, targets, weights, num_features)

        model = Ridge(alpha=self.alpha, fit_intercept=True, random_state=self.random_state)
        model.fit(data[:, used], targets, sample_weight=weights)
        score = model.score(data[:, used], targets, sample_weight=weights)
        local_pred = float(model.predict(data[0, used].reshape(1, -1))[0])

        pairs = sorted(
            zip(used, model.coef_), key=lambda t: abs(t[1]), reverse=True
        )
        coef = {int(segment_labels[i]): float(w) for i, w in pairs}
        order = [int(segment_labels[i]) for i, _ in pairs]
        return Explanation(
            coef=coef,
            order=order,
            intercept=float(model.intercept_),
            score=float(score),
            local_pred=local_pred,
            used_features=[int(i) for i in used],
        )
