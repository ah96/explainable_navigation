"""Affordance-driven perturbations -- the semantic analogue of turning a
superpixel off.

In the visuo-textual setting the interpretable features are **(object,
affordance) pairs**. Perturbing a feature applies that affordance to the object
and asks the planner what changes:

* **movability**  -- imagine the object *moved away*: its cells become free space;
* **openability** -- imagine the object's *open* state:
    * a door opens, i.e. its cells become passable (free space);
    * any other object's open parts protrude, i.e. its footprint *grows* by a
      small swing margin (more obstacle).

This lets one object contribute several, distinct LIME coefficients -- e.g. a
cabinet that is both movable and openable yields a "cabinet-movability" and a
"cabinet-openability" explanation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage

from ..perturbation import perturbations
from .ontology import Ontology
from .semantic_map import OBSTACLE_VALUE, SemanticMapInfo, semantic_to_costmap

# How far open parts (e.g. cabinet doors) protrude, in metres.
SWING_MARGIN = 0.35


def object_affordance_pairs(
    ontology: Ontology, ids_present: Iterable[int] | None = None
) -> list[tuple[int, str]]:
    """List of ``(object_id, affordance)`` pairs (the LIME features).

    ``ids_present`` optionally restricts to object ids actually in the local map.
    """
    present = None if ids_present is None else set(ids_present)
    pairs: list[tuple[int, str]] = []
    for obj in ontology:
        if present is not None and obj.id not in present:
            continue
        for aff in obj.affordances:
            pairs.append((obj.id, aff))
    return pairs


def pair_effect(
    semantic_map: np.ndarray,
    info: SemanticMapInfo,
    ontology: Ontology,
    obj_id: int,
    affordance: str,
) -> tuple[np.ndarray, int]:
    """What applying ``affordance`` to ``obj_id`` does: ``(cell mask, new value)``.

    The effect does not depend on the costmap it is applied to, so it is
    computed once per feature and reused for every perturbation.
    """
    obj_cells = semantic_map == obj_id
    if affordance == "movability":
        return obj_cells, 0  # moved away -> free space
    if affordance == "openability":
        if "door" in ontology.by_id(obj_id).label.lower():
            return obj_cells, 0  # door opens -> passable
        pad = max(1, int(round(SWING_MARGIN / info.resolution)))
        grown = ndimage.binary_dilation(obj_cells, iterations=pad)
        return grown, OBSTACLE_VALUE  # open parts protrude -> more obstacle
    raise ValueError(f"unknown affordance: {affordance}")


def build_affordance_perturbations(
    semantic_map: np.ndarray,
    info: SemanticMapInfo,
    ontology: Ontology,
    pairs: list[tuple[int, str]],
    mode: str = "one_off",
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Build the perturbation matrix and the corresponding perturbed costmaps.

    Column 0 is a constant "scene present" reference (always 1, like the
    free-space column in the costmap explainer); columns 1..F correspond to the
    feature ``pairs``. ``data[0]`` is the unperturbed instance.

    Returns ``(data, costmaps, labels)`` where ``labels[k]`` is the column->label
    map (0 for the constant column, 1..F for the pairs).
    """
    n_features = len(pairs) + 1  # + constant column
    data = perturbations(n_features, mode)
    effects = [pair_effect(semantic_map, info, ontology, obj_id, aff) for obj_id, aff in pairs]

    base = semantic_to_costmap(semantic_map)
    costmaps = []
    for row in data:
        cm = base.copy()
        # a 0 applies the feature (the object in its alternative state), in pair order
        for k in np.flatnonzero(row[1:] == 0):
            mask, value = effects[k]
            cm[mask] = value
        costmaps.append(cm)

    labels = np.arange(n_features)
    return data, costmaps, labels
