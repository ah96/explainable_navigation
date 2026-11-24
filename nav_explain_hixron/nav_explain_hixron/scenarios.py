"""The scenarios of the publications, as scripts for a run without a simulator.

``library`` is the scenario of the ICSR 2023 experiments: the robot crosses a
library to fetch a book, and two chairs are moved into its way, one after the
other. ``office`` is the scenario of the ICRA 2024 and TAHRI 2024 user studies:
the robot fetches a coffee, a chair is moved into its way, and later a human
steps into its way and leaves again.

Ontology and map geometry are the files of the original runs, in ``data/``.
Start, goal, the moved objects with their trigger points and the stop time are
the values of the original nodes. Coordinates are those of the ontology (the
simulated world). The original nodes worked in a map frame shifted by
``MAP_FRAME_OFFSET``, and their values are converted here.

The data folder is resolved in this order: the environment variable
``NAV_EXPLAIN_HIXRON_DATA`` (an explicit answer takes precedence), the source
tree (a checkout that is run without installing), the installed share folder.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from nav_explain_core import Pose2D
from nav_explain_core.hixron import MoveEvent, ReplaySpec
from nav_explain_core.semantic import load_ontology
from nav_explain_core.semantic.semantic_map import SemanticMapInfo

#: the scenario used when none is named
DEFAULT_SCENARIO = "library"
#: the map frame of the original runs has its origin here, in ontology coordinates.
#: A point of the map frame is ``(x - 9, y + 9)`` in ontology coordinates.
MAP_FRAME_OFFSET = (-9.0, 9.0)
#: radius around a trigger point of the library in the original node
LIBRARY_TRIGGER_RADIUS_M = 0.5
#: radius around the trigger point of the first chair. The scripted robot follows
#: its route exactly and passes this point at 0.66 m, so the original radius
#: would never be met.
LIBRARY_FIRST_TRIGGER_RADIUS_M = 0.8
#: ticks the robot stands still at a planning failure: the 8 s of the original
#: node, at the 0.35 s one pass of the explanation loop took there
FAILURE_STOP_TICKS = 23

#: offset from the pose of a simulated model to the centre of its box, in units
#: of the box size, by the ``mass_centre`` column of the ontology
_CENTRE_OFFSET = {
    "tr": (-0.5, -0.5), "r": (-0.5, 0.0), "br": (-0.5, 0.5), "b": (0.0, 0.5),
    "bl": (0.5, 0.5), "l": (0.5, 0.0), "tl": (0.5, -0.5), "t": (0.0, -0.5), "c": (0.0, 0.0),
}

# scenario name -> folder of its data, start pose, goal, moved objects, extra humans.
# A moved object is (id, trigger point, trigger radius, new model pose, stop ticks).
_SPECS = {
    "library": {
        "folder": "icsr2023/scenario",
        "start": Pose2D(MAP_FRAME_OFFSET[0], MAP_FRAME_OFFSET[1], 0.0),
        "goal": (-7.65, 2.5),
        "moves": [
            # chair_8 is drawn out from under the table, into the passage to the bookshelf
            (8, (-7.73, 5.84 + 1.6), LIBRARY_FIRST_TRIGGER_RADIUS_M, (-7.73, 5.84), 0),
            # chair_9 is put behind the exit of the aisle between two bookshelves
            (9, (-6.16, 3.76 + 1.8), LIBRARY_TRIGGER_RADIUS_M, (-6.16, 3.76), 0),
        ],
        # the three standing humans of icsr_world_1.world, which are not in the ontology
        "humans": [(-8.79, 4.79), (-1.92, 9.54), (-7.19, 3.57)],
    },
    "office": {
        "folder": "icra2024/scenario",
        "start": Pose2D(MAP_FRAME_OFFSET[0], MAP_FRAME_OFFSET[1], 0.0),
        "goal": (0.603 + MAP_FRAME_OFFSET[0], -6.632 + MAP_FRAME_OFFSET[1]),
        "moves": [
            # chair_2 is moved into the robot's way
            (2, (-8.5, 7.2), 1.5, (-8.5, 7.2), 0),
            # human_4 steps into the robot's way and leaves again
            (35, (-6.22, 4.16), 1.5, (-6.22, 4.16), FAILURE_STOP_TICKS),
        ],
        "humans": [],
    },
}


def data_root() -> Path:
    """The folder that holds the data of the publications."""
    env = os.environ.get("NAV_EXPLAIN_HIXRON_DATA")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[1] / "data"
    if repo.exists():
        return repo
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("nav_explain_hixron")) / "data"
    except (ImportError, LookupError):  # no ROS environment, or the package is not installed
        return repo


def available_scenarios() -> list[str]:
    root = data_root()
    return sorted(n for n, s in _SPECS.items() if (root / s["folder"] / "ontology.csv").exists())


def _centre(model_pose: tuple[float, float], row: dict) -> tuple[float, float]:
    """Centre of an object's box from the pose of its simulated model."""
    fx, fy = _CENTRE_OFFSET[row["mass_centre"]]
    return (model_pose[0] + fx * float(row["d_map_x"]), model_pose[1] + fy * float(row["d_map_y"]))


def load_scenario(name: str) -> ReplaySpec:
    if name not in _SPECS:
        raise KeyError(f"unknown scenario '{name}'; available: {sorted(_SPECS)}")
    spec = _SPECS[name]
    folder = data_root() / spec["folder"]
    with open(folder / "ontology.csv", newline="") as fh:
        rows = {int(r["ID"]): r for r in csv.DictReader(fh)}
    with open(folder / "map_info.csv", newline="") as fh:
        info = SemanticMapInfo.from_csv_row(next(csv.DictReader(fh)))
    events = tuple(
        MoveEvent(object_id=oid, trigger_xy=trigger, trigger_radius=radius,
                  new_centre=_centre(pose, rows[oid]), stop_ticks=stop)
        for oid, trigger, radius, pose, stop in spec["moves"]
    )
    return ReplaySpec(name=name, ontology=load_ontology(folder / "ontology.csv"), map_info=info,
                      start=spec["start"], goal=spec["goal"], events=events,
                      humans=tuple(spec["humans"]))
