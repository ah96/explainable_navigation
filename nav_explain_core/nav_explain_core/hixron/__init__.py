"""HiXRoN: explanations of navigation that depend on the robot's personality.

While the robot drives, it explains its surroundings and the changes of its
global plan, visually and in words. How often it explains, how much detail it
gives and whether it speaks at all follow from one number, its extroversion,
and from how near the closest human is.

* ``personality`` -- extroversion to timing, duration, detail and representation;
* ``layer``       -- the visual explanation: a heat map of the objects around the robot;
* ``text``        -- the textual explanation;
* ``deviation``   -- when a new global plan counts as a deviation;
* ``route``       -- a global route on the semantic map;
* ``replay``      -- a scripted run of a scenario, which stands in for a simulation.
"""

from .deviation import DEVIATION_THRESHOLD_M, plan_deviation
from .layer import LayerResult, explanation_layer
from .personality import ExplanationPolicy
from .replay import Explanation, MoveEvent, Replay, ReplayResult, ReplaySpec, run_replay
from .route import plan_route
from .text import deviation_text, neighbourhood_text

__all__ = [
    "DEVIATION_THRESHOLD_M", "plan_deviation", "LayerResult", "explanation_layer",
    "ExplanationPolicy", "Explanation", "MoveEvent", "Replay", "ReplayResult", "ReplaySpec",
    "run_replay", "plan_route", "deviation_text", "neighbourhood_text",
]
