"""Turn surrogate coefficients into a colour-coded explanation image.

Following the paper:

* obstacle segment with a **positive** coefficient (it increases deviation) ->
  **green**;
* obstacle segment with a **negative** coefficient (it decreases deviation) ->
  **red**;
* colour intensity is proportional to ``|coef| / max|coef|``;
* free space (segment 0) is rendered grey.

(The original ``get_image_and_mask`` uses exactly this green=positive /
red=negative mapping with intensity normalised by the maximum absolute weight.)
"""

from __future__ import annotations

import numpy as np

from .ridge_explainer import Explanation

FREE_SPACE_GREY = 0.7


def colorize_segments(
    segments: np.ndarray,
    explanation: Explanation,
    free_space_shade: float = FREE_SPACE_GREY,
) -> np.ndarray:
    """Return an ``(H, W, 3)`` float RGB image in ``[0, 1]``.

    Args:
        segments: integer segment labels (0 = free space).
        explanation: fitted :class:`Explanation` with per-segment coefficients.
    """
    h, w = segments.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[segments == 0] = free_space_shade

    coef = explanation.coef
    if coef:
        max_abs = max(abs(v) for v in coef.values()) or 1.0
    else:
        max_abs = 1.0

    for label, weight in coef.items():
        if label == 0:
            continue  # free space stays grey
        mask = segments == label
        intensity = abs(weight) / max_abs
        if weight > 0:  # increases deviation -> green
            img[mask] = (0.0, intensity, 0.0)
        elif weight < 0:  # decreases deviation -> red
            img[mask] = (intensity, 0.0, 0.0)
        # weight == 0 -> leave black
    return img
