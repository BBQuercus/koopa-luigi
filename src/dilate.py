"""Pure-NumPy helpers for dilating labeled segmentation maps and planning
which (channel, dilation) combinations a Merge task needs to load.

Kept in its own module (no koopa or luigi imports) so the logic can be
unit-tested without pulling in the heavy ML stack.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.ndimage as ndi
import skimage.morphology
import skimage.segmentation


def plan_other_segmaps(
    sego_channels: Iterable, sego_dilations
) -> list[tuple[str, str, dict]]:
    """Return the segmap dependency plan for "other" segmentations.

    Each entry is ``(segmap_key, kind, params)`` where ``kind`` is either
    ``"segment"`` (use SegmentOther) or ``"dilate"`` (use DilateSegmentOther).
    The order is preserved.

    Rules:
      - Empty (or missing) ``sego_dilations[idx]`` → legacy: just ``other_{idx}``.
      - Otherwise, for each radius ``r`` in the list:
          - ``r == 0`` → ``other_{idx}`` from SegmentOther
          - ``r > 0``  → ``other_{idx}_d{r}`` from DilateSegmentOther
    """
    plan: list[tuple[str, str, dict]] = []
    dilations = list(sego_dilations) if sego_dilations else []

    for idx, _ in enumerate(sego_channels):
        radii = dilations[idx] if idx < len(dilations) else []
        if not radii:
            plan.append((f"other_{idx}", "segment", {"index_list": idx}))
            continue
        for r in radii:
            if r == 0:
                plan.append((f"other_{idx}", "segment", {"index_list": idx}))
            else:
                plan.append(
                    (
                        f"other_{idx}_d{r}",
                        "dilate",
                        {"index_list": idx, "dilation": r},
                    )
                )
    return plan


def dilate_labels(segmap: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a 2D or 3D labeled segmentation map by ``radius`` pixels.

    Uses a unit-radius structuring element iterated ``radius`` times so the
    result grows by exactly ``radius`` pixels along each axis. Original
    instance labels are preserved via watershed seeded by the original mask,
    so adjacent objects stay distinct after dilation.
    """
    if radius <= 0:
        return segmap

    if segmap.ndim == 2:
        structure = skimage.morphology.disk(1)
    elif segmap.ndim == 3:
        structure = skimage.morphology.ball(1)
    else:
        raise ValueError(
            f"Unsupported segmap dimensions for dilation: {segmap.ndim}"
        )

    binary = segmap > 0
    mask = ndi.binary_dilation(binary, structure, iterations=radius)
    return skimage.segmentation.watershed(
        mask.astype(np.uint8), markers=segmap, mask=mask
    )
