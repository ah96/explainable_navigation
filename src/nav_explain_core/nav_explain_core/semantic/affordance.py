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
"cabinet-openability" explanation, exactly as in the paper's Fig. r3--r6.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .ontology import Ontology
from .semantic_map import OBSTACLE_VALUE, SemanticMapInfo, semantic_to_costmap

# How far open parts (e.g. cabinet doors) protrude, in metres.
SWING_MARGIN = 0.35


def object_affordance_pairs(ontology: Ontology, ids_present=None) -> list[tuple[int, str]]:
    """List of ``(object_id, affordance)`` pairs (the LIME features).

    ``ids_present`` optionally restricts to object ids actually in the local map.
    """
    pairs: list[tuple[int, str]] = []
    for obj in ontology:
        if ids_present is not None and obj.id not in ids_present:
            continue
        for aff in obj.affordances:
            pairs.append((obj.id, aff))
    return pairs


def perturb_for_pair(
    costmap: np.ndarray,
    semantic_map: np.ndarray,
    info: SemanticMapInfo,
    ontology: Ontology,
    obj_id: int,
    affordance: str,
) -> np.ndarray:
    """Return ``costmap`` modified by applying ``affordance`` to ``obj_id``."""
    out = costmap.copy()
    obj_cells = semantic_map == obj_id
    obj = ontology.by_id(obj_id)

    if affordance == "movability":
        out[obj_cells] = 0  # moved away -> free space
    elif affordance == "openability":
        if "door" in obj.label.lower():
            out[obj_cells] = 0  # door opens -> passable
        else:
            pad = max(1, int(round(SWING_MARGIN / info.resolution)))
            grown = ndimage.binary_dilation(obj_cells, iterations=pad)
            out[grown] = OBSTACLE_VALUE  # open parts protrude -> more obstacle
    else:
        raise ValueError(f"unknown affordance: {affordance}")
    return out


def build_affordance_perturbations(
    semantic_map: np.ndarray,
    info: SemanticMapInfo,
    ontology: Ontology,
    pairs: list[tuple[int, str]],
    mode: str = "one_off",
):
    """Build the perturbation matrix and the corresponding perturbed costmaps.

    Column 0 is a constant "scene present" reference (always 1, like the
    free-space column in the costmap explainer); columns 1..F correspond to the
    feature ``pairs``. ``data[0]`` is the unperturbed instance.

    Returns ``(data, costmaps, labels)`` where ``labels[k]`` is the column->label
    map (0 for the constant column, 1..F for the pairs).
    """
    from ..lime.perturbation import full_2n_perturbations, one_off_perturbations

    n_features = len(pairs) + 1  # + constant column
    if mode == "one_off":
        data = one_off_perturbations(n_features)
    elif mode == "full_2n":
        data = full_2n_perturbations(n_features)
    else:
        raise ValueError("mode must be 'one_off' or 'full_2n'")

    base = semantic_to_costmap(semantic_map)
    costmaps = []
    for row in data:
        cm = base.copy()
        for k in range(1, n_features):
            if row[k] == 0:  # feature applied (object in its alternative state)
                obj_id, aff = pairs[k - 1]
                cm = perturb_for_pair(cm, semantic_map, info, ontology, obj_id, aff)
        costmaps.append(cm)

    labels = np.arange(n_features)
    return data, costmaps, labels
