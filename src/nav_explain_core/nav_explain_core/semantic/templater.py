"""Natural-language templates for the textual explanations.

Two kinds, both anchored by a QSR relation ("left/front of me"):

* **descriptive** -- why the robot deviates, e.g.
  "I deviate because of the chair left/front of me.";
* **suggestive** -- a human-directed mitigation tied to the affordance, e.g.
  "Dear human, please move the chair left/front of me."
  (movability -> "move", openability -> "open").

Mirrors the sentence construction in ``qsr_rt.py`` / the tahri textual node.
"""

from __future__ import annotations

_ACTION = {"movability": "move", "openability": "open"}


def action_for_affordance(affordance: str) -> str:
    return _ACTION.get(affordance, "change")


def descriptive(object_label: str, qsr: str, increases: bool = True) -> str:
    """Descriptive explanation.

    ``increases`` True  -> the object increases the deviation ("I deviate because
    of ...");  False -> it limits the deviation ("If ... were not there I would
    deviate more").
    """
    if increases:
        return f"I deviate because of the {object_label} {qsr} of me."
    return (
        f"If the {object_label} {qsr} of me were not there, "
        f"I would deviate more from my plan."
    )


def suggestive(object_label: str, qsr: str, affordance: str) -> str:
    """Human-directed suggestion keyed to the object's affordance."""
    action = action_for_affordance(affordance)
    return f"Dear human, please {action} the {object_label} {qsr} of me."
