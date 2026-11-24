"""Deterministic anchor search over a perturbation set.

Given a perturbation matrix (binary, columns = interpretable features, a 1 means
the feature/segment is present) and a binary label per perturbation (1 = the
model made the same decision as the instance), find a minimal set of features
whose presence keeps the decision with precision >= ``tau``.

Precision(A)  = P(label = 1 | every feature in A is present)
Coverage(A)   = P(every feature in A is present)

The original implementation used a stochastic LUCB multi-armed-bandit beam
search; because we evaluate against the full, enumerable perturbation set the
precision is computed exactly, so a deterministic greedy beam search suffices and
gives reproducible anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnchorExplanation:
    """The resulting anchor rule."""

    features: list[int]  # column indices that make up the anchor
    segment_labels: list[int] = field(default_factory=list)  # mapped segment ids
    precision: float = 0.0
    coverage: float = 0.0


def _precision_coverage(data: np.ndarray, labels: np.ndarray, feats: tuple[int, ...]):
    if not feats:
        cov = 1.0
        return float(labels.mean()), cov
    present = np.all(data[:, list(feats)] == 1, axis=1)
    n = int(present.sum())
    if n == 0:
        return 0.0, 0.0
    precision = float(labels[present].mean())
    coverage = float(n / data.shape[0])
    return precision, coverage


def find_anchor(
    data: np.ndarray,
    labels: np.ndarray,
    candidate_features: list[int] | None = None,
    tau: float = 0.95,
    beam_size: int = 5,
    max_len: int | None = None,
) -> AnchorExplanation:
    """Greedy beam search for a minimal, high-precision anchor.

    Args:
        data: ``(n_perturbations, n_features)`` binary matrix.
        labels: ``(n_perturbations,)`` binary labels (1 = decision preserved).
        candidate_features: columns eligible to be in the anchor (defaults to all
            except column 0, which is the always-on free-space/scene column).
        tau: required precision.
        beam_size: number of partial anchors kept between rounds.
        max_len: cap on anchor size (defaults to #candidates).
    """
    data = np.asarray(data)
    labels = np.asarray(labels).astype(float)
    n_features = data.shape[1]
    if candidate_features is None:
        candidate_features = [f for f in range(n_features) if f != 0]
    if max_len is None:
        max_len = len(candidate_features)

    best: AnchorExplanation | None = None
    beam: list[tuple[int, ...]] = [()]
    for _length in range(1, max_len + 1):
        scored = []
        for partial in beam:
            for f in candidate_features:
                if f in partial:
                    continue
                cand = tuple(sorted(partial + (f,)))
                prec, cov = _precision_coverage(data, labels, cand)
                scored.append((cand, prec, cov))
        if not scored:
            break
        # dedup
        seen = {}
        for cand, prec, cov in scored:
            seen[cand] = (prec, cov)
        # any anchor meeting tau? keep the one with best coverage, then smallest.
        qualifying = [(c, p, cov) for c, (p, cov) in seen.items() if p >= tau]
        if qualifying:
            cand, prec, cov = max(qualifying, key=lambda t: (t[2], -len(t[0])))
            if best is None or cov > best.coverage:
                best = AnchorExplanation(features=list(cand), precision=prec, coverage=cov)
            break  # minimal length reached
        # otherwise keep the most promising partials (highest precision) for the beam
        beam = [
            c for c, _ in sorted(seen.items(), key=lambda kv: kv[1][0], reverse=True)[:beam_size]
        ]

    if best is None:
        # no anchor reaches tau; fall back to the single most precise feature
        singles = [
            (f, *_precision_coverage(data, labels, (f,))) for f in candidate_features
        ]
        f, prec, cov = max(singles, key=lambda t: t[1])
        best = AnchorExplanation(features=[f], precision=prec, coverage=cov)
    return best
