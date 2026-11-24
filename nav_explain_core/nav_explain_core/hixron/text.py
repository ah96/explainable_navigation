"""The textual explanation.

Two sentences exist, each in a rich and a poor form. One names the objects
around the robot, the other names the object whose move made the robot
deviate. A rich text places every object relative to the robot (left, right,
in front, behind) and speaks in the first person. A poor text only lists.
"""

from __future__ import annotations

import math

from ..geometry import Pose2D
from ..semantic.qsr import relative_angle


def side_of_robot(robot: Pose2D, obj_xy: tuple[float, float]) -> str:
    """``"front"``, ``"left"``, ``"back"`` or ``"right"``: the quarter the object lies in."""
    angle = relative_angle(robot, obj_xy)
    if -math.pi / 4 <= angle < math.pi / 4:
        return "front"
    if math.pi / 4 <= angle < 3 * math.pi / 4:
        return "left"
    if -3 * math.pi / 4 <= angle < -math.pi / 4:
        return "right"
    return "back"


def _enumerate(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def deviation_text(label: str, side: str, detail_level: str) -> str:
    """The sentence for an object whose move made the robot deviate."""
    if detail_level == "poor":
        return f"The {label} was moved."
    if side in ("left", "right"):
        return f"I am deviating because the {label} to my {side}, was moved."
    where = "behind me" if side == "back" else "in front of me"
    return f"I am deviating because the {label}, which is {where}, was moved."


def neighbourhood_text(objects: list[tuple[str, str]], detail_level: str) -> str:
    """The sentence for the objects around the robot, given as ``(label, side)`` pairs."""
    if not objects:
        return "There are no objects close to me."
    labels = [label for label, _ in objects]
    if detail_level == "poor":
        verb = "is" if len(labels) == 1 else "are"
        return f"Around me {verb} {_enumerate(labels)}."
    by_side = {side: [label for label, s in objects if s == side]
               for side in ("left", "right", "front", "back")}
    passing = [f"{_enumerate(by_side[side])} to my {side}" for side in ("left", "right") if by_side[side]]
    sentences = []
    if passing:
        sentences.append("I am passing by " + " and ".join(passing) + ".")
    for side, where in (("front", "In front of me"), ("back", "Behind me")):
        if by_side[side]:
            verb = "is" if len(by_side[side]) == 1 else "are"
            sentences.append(f"{where} {verb} {_enumerate(by_side[side])}.")
    return " ".join(sentences)
