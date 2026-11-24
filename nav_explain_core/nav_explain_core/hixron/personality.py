"""From the robot's extroversion to the properties of its explanations.

The extroversion is a number between 0 (totally introverted) and 1 (totally
extroverted). It sets four things:

* the **representation threshold**, a distance in metres. A human nearer than
  this makes the robot explain visually only. It falls linearly from the outer
  edge of the social zone of Hall's social space (3.6 m) to its inner edge
  (1.2 m), since extroverts accept others closer to them.
* the **window**, the half side length of the visual explanation layer. It
  equals the threshold, so that an introverted robot shows more objects.
* the **timing** and the **duration**, both in ticks of the explanation loop:
  how long the robot waits before it gives an explanation it has formed, and
  how long that explanation stays. Both fall linearly from 20 to 0, since
  extroverts pause less and speak shorter.
* the **detail level** of the text: rich below 0.5, poor from 0.5 on.
"""

from __future__ import annotations

from dataclasses import dataclass

#: representation threshold of a totally introverted robot, in metres
THRESHOLD_INTROVERT_M = 3.6
#: how much the threshold falls from extroversion 0 to 1, in metres
THRESHOLD_RANGE_M = 2.4
#: timing and duration of a totally introverted robot, in ticks
MAX_DELAY_TICKS = 20
#: extroversion below which the text is rich in detail
RICH_BELOW = 0.5


@dataclass(frozen=True)
class ExplanationPolicy:
    """The explanation properties of a robot with a given extroversion."""

    #: 0 is totally introverted, 1 is totally extroverted
    extroversion: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.extroversion <= 1.0:
            raise ValueError(f"extroversion must lie in [0, 1], got {self.extroversion}")

    @property
    def representation_threshold(self) -> float:
        return THRESHOLD_INTROVERT_M - THRESHOLD_RANGE_M * self.extroversion

    @property
    def window(self) -> float:
        return self.representation_threshold

    @property
    def timing(self) -> int:
        # round, not int: 20 * (1 - 0.8) is 3.9999999999999996 in floating point
        return round(MAX_DELAY_TICKS * (1.0 - self.extroversion))

    @property
    def duration(self) -> int:
        return self.timing

    @property
    def cycle(self) -> int:
        """Ticks from one formed explanation to the next, at least one."""
        return max(1, self.timing + self.duration)

    @property
    def detail_level(self) -> str:
        return "rich" if self.extroversion < RICH_BELOW else "poor"

    def representation(self, distance_to_closest_human: float | None) -> str:
        """``"visual"`` when a human is nearer than the threshold, else ``"visual+textual"``."""
        if distance_to_closest_human is not None and distance_to_closest_human < self.representation_threshold:
            return "visual"
        return "visual+textual"
