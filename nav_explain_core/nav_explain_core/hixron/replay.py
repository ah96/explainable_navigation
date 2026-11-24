"""The explanation loop, and a scripted run of a scenario that drives it.

:class:`Replay` is the loop of the original node without ROS. It holds the
world (the ontology and its semantic map), it is told about moved objects and
new global plans, and on every tick it may deliver one explanation. A live node
feeds it from topics. :func:`run_replay` feeds it from a script instead: the
robot follows a global route at a fixed step per tick, and objects move when
the robot comes near a trigger point, as in the simulated runs of the original
work. No simulator and no clock are involved, so a run is deterministic.

One cycle of the loop, for timing ``t`` and duration ``d`` (see ``personality``):
the robot forms an explanation, waits ``t`` ticks, gives it, and keeps it for
``d`` ticks. A totally extroverted robot forms and gives one on every tick.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose2D
from ..semantic.ontology import Ontology
from ..semantic.semantic_map import SemanticMapInfo, build_semantic_map
from .deviation import DEVIATION_THRESHOLD_M, plan_deviation
from .layer import LayerResult, explanation_layer
from .personality import ExplanationPolicy
from .route import plan_route, resample
from .text import deviation_text, neighbourhood_text, side_of_robot

#: distance the robot covers per tick in a scripted run, in metres. With it the
#: built-in library run has about as many ticks as the recorded run of the
#: totally extroverted robot has rows (177).
STEP_M = 0.08
#: number of delivered explanations that keep showing the cause of a deviation
HIGHLIGHT_EXPLANATIONS = 12


@dataclass(frozen=True)
class MoveEvent:
    """An object that moves when the robot comes near a point."""

    #: id of the object in the ontology
    object_id: int
    #: the event fires when the robot is nearer to this point than ``trigger_radius``
    trigger_xy: tuple[float, float]
    trigger_radius: float
    #: centre of the object after the move
    new_centre: tuple[float, float]
    #: above zero the robot stops for this many ticks and the object then moves
    #: back: a planning failure. Zero is a move the robot plans around.
    stop_ticks: int = 0


@dataclass(frozen=True)
class ReplaySpec:
    """A scenario for a scripted run."""

    name: str
    ontology: Ontology
    map_info: SemanticMapInfo
    start: Pose2D
    goal: tuple[float, float]
    events: tuple[MoveEvent, ...] = ()
    #: positions of humans that are not objects of the ontology
    humans: tuple[tuple[float, float], ...] = ()
    step: float = STEP_M


@dataclass(frozen=True)
class Explanation:
    """One delivered explanation."""

    #: tick of the run at which it was given
    tick: int
    #: ``"visual"`` or ``"visual+textual"``
    representation: str
    #: the text, empty when the representation is visual only
    text: str
    layer: LayerResult
    #: number of objects the visual explanation shows
    n_objects: int
    #: number of words of the text
    n_words: int
    #: id of the object named as the cause of a deviation or failure, or ``None``
    cause_id: int | None
    #: ``"deviation"``, ``"failure"`` or ``""``
    cause_kind: str
    #: true for the first explanation that names this cause
    first_for_cause: bool
    #: pose of the robot when the explanation was formed
    robot: Pose2D
    #: time it took to form the visual and the textual part, in seconds
    visual_seconds: float = field(compare=False, default=0.0)
    textual_seconds: float = field(compare=False, default=0.0)


@dataclass
class ReplayResult:
    """What a scripted run produced."""

    spec_name: str
    extroversion: float
    explanations: list[Explanation]
    n_ticks: int
    #: deviations and failures that happened, and how many of them were explained
    n_causes: int
    n_causes_explained: int
    reached_goal: bool
    #: every global plan of the run, in order
    plans: list[np.ndarray]
    #: position of the robot at every tick
    trace: np.ndarray

    def summary(self) -> dict[str, float]:
        """The quantities of one row of the evaluation, see ``nav_explain_hixron.results``."""
        visual = len(self.explanations)
        textual = [e for e in self.explanations if e.text]
        return {
            "extroversion": round(100 * self.extroversion),
            "visual_N": visual,
            "textual_N": len(textual),
            "N_objects": sum(e.n_objects for e in self.explanations) / visual if visual else 0.0,
            "N_words": sum(e.n_words for e in textual) / len(textual) if textual else 0.0,
            "N_deviations": self.n_causes_explained,
            "visual_time": 1000 * sum(e.visual_seconds for e in self.explanations) / visual if visual else 0.0,
            "textual_time": 1000 * sum(e.textual_seconds for e in self.explanations) / visual if visual else 0.0,
        }


def is_wall(label: str) -> bool:
    return label.startswith("wall")


def is_human(label: str) -> bool:
    return label.startswith("human")


class Replay:
    """The explanation loop over a world that changes."""

    def __init__(self, ontology: Ontology, map_info: SemanticMapInfo, policy: ExplanationPolicy,
                 humans: tuple[tuple[float, float], ...] = ()) -> None:
        self.ontology = ontology
        self.map_info = map_info
        self.policy = policy
        self._static_humans = tuple(humans)
        self.wall_ids = frozenset(o.id for o in ontology if is_wall(o.label))
        self.label_map = build_semantic_map(ontology, map_info)
        self.n_causes = 0
        self.n_causes_explained = 0
        self._previous_plan: np.ndarray | None = None
        self._last_moved: int | None = None
        self._pending: tuple[int, str] | None = None
        self._cause: tuple[int, str] | None = None
        self._cause_left = 0
        self._cause_is_new = False
        self._formed: tuple[LayerResult, str, Pose2D, float, float] | None = None
        self._k = 0
        self._tick = 0

    # -- the world ---------------------------------------------------------

    def move_object(self, object_id: int, centre: tuple[float, float]) -> None:
        """Place an object at a new centre and rebuild the semantic map."""
        objects = [dataclasses.replace(o, cx=centre[0], cy=centre[1]) if o.id == object_id else o
                   for o in self.ontology]
        self.ontology = Ontology(objects)
        self.label_map = build_semantic_map(self.ontology, self.map_info)
        self._last_moved = object_id

    def set_plan(self, plan: np.ndarray) -> float:
        """Take a new global plan. Returns its deviation from the previous plan in metres.

        A deviation above the threshold after an object has moved is remembered
        and named in the next explanation that is formed.
        """
        plan = np.asarray(plan, dtype=float).reshape(-1, 2)
        deviation = 0.0
        if self._previous_plan is not None:
            deviation = plan_deviation(self._previous_plan, plan)
            if deviation > DEVIATION_THRESHOLD_M and self._last_moved is not None:
                self._report(self._last_moved, "deviation")
        self._previous_plan = plan
        return deviation

    def report_failure(self, object_id: int) -> None:
        """The robot has stopped because this object is in its way."""
        self._report(object_id, "failure")

    def _report(self, object_id: int, kind: str) -> None:
        self._pending = (object_id, kind)
        self._last_moved = None
        self.n_causes += 1

    def human_positions(self) -> list[tuple[float, float]]:
        return list(self._static_humans) + [(o.cx, o.cy) for o in self.ontology if is_human(o.label)]

    # -- the loop ----------------------------------------------------------

    def tick(self, robot: Pose2D) -> Explanation | None:
        """Advance the loop by one tick. Returns the explanation given at this tick, if any."""
        delivered = None
        if self._k == 0:
            self._formed = self._form(robot)
        if self._k == self.policy.timing % self.policy.cycle:
            delivered = self._deliver(robot)
        self._k = (self._k + 1) % self.policy.cycle
        self._tick += 1
        return delivered

    def _form(self, robot: Pose2D) -> tuple[LayerResult, str, Pose2D, float, float]:
        if self._pending is not None:
            self._cause, self._pending = self._pending, None
            self._cause_left = HIGHLIGHT_EXPLANATIONS
            self._cause_is_new = True
        cause = self._cause if self._cause_left > 0 else None

        t0 = time.perf_counter()
        layer = explanation_layer(self.label_map, self.map_info, (robot.x, robot.y), self.policy.window,
                                  wall_ids=self.wall_ids, highlight_id=cause[0] if cause else None)
        t1 = time.perf_counter()
        if cause is not None:
            obj = self.ontology.by_id(cause[0])
            if cause[1] == "failure":
                text = f"Moving obstruction: '{obj.label}'. '{obj.label.capitalize()}, please move!'"
            else:
                text = deviation_text(obj.label, side_of_robot(robot, (obj.cx, obj.cy)),
                                      self.policy.detail_level)
        else:
            near = [self.ontology.by_id(i) for i in layer.object_ids]
            text = neighbourhood_text([(o.label, side_of_robot(robot, (o.cx, o.cy))) for o in near],
                                      self.policy.detail_level)
        t2 = time.perf_counter()
        return layer, text, robot, t1 - t0, t2 - t1

    def _deliver(self, robot: Pose2D) -> Explanation:
        layer, text, formed_at, visual_seconds, textual_seconds = self._formed
        humans = self.human_positions()
        nearest = min((math.hypot(x - robot.x, y - robot.y) for x, y in humans), default=None)
        representation = self.policy.representation(nearest)
        if representation == "visual":
            text = ""
        cause = self._cause if layer.highlight_id is not None else None
        first = cause is not None and self._cause_is_new
        if first:
            self.n_causes_explained += 1
            self._cause_is_new = False
        if self._cause_left > 0 and cause is not None:
            self._cause_left -= 1
        shown = set(layer.object_ids) | ({cause[0]} if cause else set())
        return Explanation(
            tick=self._tick, representation=representation, text=text, layer=layer,
            n_objects=len(shown), n_words=len(text.split()),
            cause_id=cause[0] if cause else None, cause_kind=cause[1] if cause else "",
            first_for_cause=first, robot=formed_at,
            visual_seconds=visual_seconds, textual_seconds=textual_seconds,
        )


def run_replay(spec: ReplaySpec, extroversion: float, max_ticks: int = 5000) -> ReplayResult:
    """Run the scripted scenario once with a robot of the given extroversion."""
    replay = Replay(spec.ontology, spec.map_info, ExplanationPolicy(extroversion), spec.humans)
    original = {o.id: (o.cx, o.cy) for o in spec.ontology}

    def route(from_xy: tuple[float, float]) -> np.ndarray:
        found = plan_route(replay.label_map, spec.map_info, from_xy, spec.goal)
        if found is None:
            raise RuntimeError(f"scenario '{spec.name}': no route from {from_xy} to {spec.goal}")
        return resample(found, spec.step)

    pose = spec.start
    plan = route((pose.x, pose.y))
    replay.set_plan(plan)
    plans, trace, explanations = [plan], [], []
    index, stopped, to_return = 0, 0, None
    fired: set[int] = set()
    reached = False
    tick = 0
    for tick in range(max_ticks):
        replanned = False
        for i, event in enumerate(spec.events):
            if i in fired or math.hypot(pose.x - event.trigger_xy[0],
                                        pose.y - event.trigger_xy[1]) >= event.trigger_radius:
                continue
            fired.add(i)
            replay.set_plan(plan[index:])
            replay.move_object(event.object_id, event.new_centre)
            if event.stop_ticks > 0:
                stopped, to_return = event.stop_ticks, event.object_id
                replay.report_failure(event.object_id)
            else:
                replanned = True
        if to_return is not None and stopped == 0:
            replay.set_plan(plan[index:])
            replay.move_object(to_return, original[to_return])
            to_return, replanned = None, True
        if replanned:
            plan, index = route((pose.x, pose.y)), 0
            replay.set_plan(plan)
            plans.append(plan)

        trace.append((pose.x, pose.y))
        delivered = replay.tick(pose)
        if delivered is not None:
            explanations.append(delivered)

        if stopped > 0:
            stopped -= 1
            continue
        if index >= len(plan) - 1:
            reached = True
            break
        index += 1
        dx, dy = plan[index] - plan[index - 1]
        pose = Pose2D(float(plan[index, 0]), float(plan[index, 1]), math.atan2(dy, dx))
    return ReplayResult(
        spec_name=spec.name, extroversion=extroversion, explanations=explanations, n_ticks=tick + 1,
        n_causes=replay.n_causes, n_causes_explained=replay.n_causes_explained, reached_goal=reached,
        plans=plans, trace=np.asarray(trace, dtype=float),
    )
