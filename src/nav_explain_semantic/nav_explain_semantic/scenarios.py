"""Conference-free semantic scenarios (ontology + map + authored plan).

``cabinet_room`` is a furnished room (cabinet / table / chair / door + walls). The
robot routes from the upper to the lower part of the room *through the closed
door*, passing the cabinet — exercising the door's openability and the cabinet's
movability/openability. Ontology and map geometry come from a real scenario; the
robot pose and global plan are authored here (no external files needed at run
time beyond the ontology/map CSVs shipped in ``data/``).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from nav_explain_core import Pose2D
from nav_explain_core.semantic import VTInstance, load_ontology
from nav_explain_core.semantic.semantic_map import SemanticMapInfo


def _data_root() -> Path:
    env = os.environ.get("NES_DATA_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[1] / "data" / "scenarios"
    if repo.exists():
        return repo
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("nav_explain_semantic")) / "data" / "scenarios"
    except Exception:
        return repo


# scenario name -> (robot pose, plan waypoints)
_SPECS = {
    "cabinet_room": {
        "robot_pose": Pose2D(2.0, 1.2, 0.0),
        "waypoints": [(2.0, 1.2), (6.6, 1.0), (7.47, 0.3), (7.47, -1.2)],
        "description": "Robot routes through the closed door past the cabinet.",
    }
}


def available_scenarios() -> list[str]:
    root = _data_root()
    return sorted(n for n in _SPECS if (root / n / "ontology.csv").exists())


def _interp(waypoints, n=60) -> np.ndarray:
    wp = np.asarray(waypoints, float)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, wp[:, 0]), np.interp(t, s, wp[:, 1])])


def load_scenario(name: str) -> VTInstance:
    if name not in _SPECS:
        raise KeyError(f"unknown scenario '{name}'; available: {sorted(_SPECS)}")
    d = _data_root() / name
    ontology = load_ontology(d / "ontology.csv")
    with open(d / "map_info.csv", newline="") as fh:
        info = SemanticMapInfo.from_csv_row(next(csv.DictReader(fh)))
    spec = _SPECS[name]
    return VTInstance(
        ontology=ontology,
        map_info=info,
        global_plan=_interp(spec["waypoints"]),
        robot_pose=spec["robot_pose"],
        name=name,
    )
