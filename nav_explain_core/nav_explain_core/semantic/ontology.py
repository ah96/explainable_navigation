"""Ontology of environment objects with affordances.

The ontology is a CSV with one row per object and the exact columns used in the
original work (``hixron_subscriber.py`` / the scenario ``ontology.csv``)::

    ID, gazebo_label, label, c_map_x, c_map_y, d_map_x, d_map_y, movability, openability

``c_map_*`` is the object centroid in the map frame; ``d_map_*`` is its full
extent (width/height) in metres; ``movability`` / ``openability`` are 0/1 flags
encoding the object's affordances.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObjectEntry:
    """One object of the environment: a row of ``ontology.csv``."""

    id: int
    gazebo_label: str
    label: str
    cx: float
    cy: float
    dx: float  # full extent in x (metres)
    dy: float  # full extent in y (metres)
    movability: bool
    openability: bool

    @property
    def affordances(self) -> list[str]:
        out = []
        if self.movability:
            out.append("movability")
        if self.openability:
            out.append("openability")
        return out

    @property
    def half_extents(self) -> tuple[float, float]:
        return 0.5 * self.dx, 0.5 * self.dy


@dataclass
class Ontology:
    """The objects of one environment."""

    objects: list[ObjectEntry]

    def __iter__(self) -> Iterator[ObjectEntry]:
        return iter(self.objects)

    def __len__(self) -> int:
        return len(self.objects)

    def by_id(self, oid: int) -> ObjectEntry:
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(oid)

    def with_affordances(self) -> list[ObjectEntry]:
        """Objects that have at least one affordance (the LIME-relevant ones)."""
        return [o for o in self.objects if o.affordances]


def load_ontology(csv_path: Path | str) -> Ontology:
    """Read an ontology from its CSV file."""
    objects: list[ObjectEntry] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            objects.append(
                ObjectEntry(
                    id=int(row["ID"]),
                    gazebo_label=row["gazebo_label"],
                    label=row["label"],
                    cx=float(row["c_map_x"]),
                    cy=float(row["c_map_y"]),
                    dx=float(row["d_map_x"]),
                    dy=float(row["d_map_y"]),
                    movability=bool(int(row["movability"])),
                    openability=bool(int(row["openability"])),
                )
            )
    return Ontology(objects)
