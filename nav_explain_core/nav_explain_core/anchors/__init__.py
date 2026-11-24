"""Anchors explanation method (a rule-based alternative to LIME).

An *anchor* is a minimal set of interpretable features (here: obstacle segments)
whose presence "anchors" the model's decision: as long as those obstacles are
present, the robot deviates (roughly) as it currently does, regardless of the
other obstacles. This is a second, contrasting XAI method to LIME's additive
coefficients.
"""

from .base import AnchorExplanation, find_anchor
from .image import AnchorImageExplainer

__all__ = ["AnchorExplanation", "find_anchor", "AnchorImageExplainer"]
