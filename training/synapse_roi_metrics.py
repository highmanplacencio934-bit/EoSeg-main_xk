"""Explicit opt-in ROI acceleration; the original metric implementation stays unchanged."""
from __future__ import annotations

import numpy as np

from training.synapse_physical_geometry import binary_metrics_mm, checked_spacing


def union_roi_slices(pred, gt, halo=1):
    """Keep every foreground voxel from both equal-shape binary 3D arrays."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("Expected equal-shape 3D masks")
    if isinstance(halo, bool) or not isinstance(halo, int) or halo < 1:
        raise ValueError("halo must be a positive integer")
    union = pred | gt
    if not union.any():
        return None
    result = []
    for axis in range(3):
        other_axes = tuple(i for i in range(3) if i != axis)
        occupied = np.flatnonzero(np.any(union, axis=other_axes))
        result.append(slice(max(0, int(occupied[0]) - halo),
                            min(union.shape[axis], int(occupied[-1]) + halo + 1)))
    return tuple(result)


def binary_metrics_mm_roi(pred, gt, spacing_zyx_mm):
    """Same Dice, HD95 mm/voxel and legacy empty policy, on a common union ROI.

    Geometry/label hashes must be checked on the original full case by callers.
    No connected components, resampling, spacing changes, or precision changes.
    """
    spacing = checked_spacing(spacing_zyx_mm)
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("Physical HD95 requires equal-shape 3D masks")
    if not pred.any() or not gt.any():
        return binary_metrics_mm(pred, gt, spacing)
    roi = union_roi_slices(pred, gt)
    return binary_metrics_mm(pred[roi], gt[roi], spacing)
