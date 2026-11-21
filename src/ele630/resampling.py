"""Prototype NCCT voxel resampling onto an explicitly planned physical grid.

This module applies no intensity window, normalization, or brain mask. It does
not load or alter patient files. Input and output affines use NIfTI RAS+ world
millimetres, regardless of their voxel-axis orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import GridPlan, Matrix4


@dataclass(frozen=True)
class ResampledVolume:
    data: np.ndarray
    affine: Matrix4


def resample_to_grid(
    data: np.ndarray,
    input_affine: Sequence[Sequence[float]],
    plan: GridPlan,
    *,
    outside_value: float,
) -> ResampledVolume:
    """Linearly sample source voxels at each output voxel's world position.

    The source array and affine must match those used to create ``plan``.
    Samples outside the source array receive ``outside_value`` with SciPy's
    constant-boundary mode. Output values are float64 to retain interpolation
    results even when the source array has an integer dtype.
    """
    if not isinstance(plan, GridPlan):
        raise TypeError("plan must be a GridPlan")
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("data must be a 3D NumPy array")
    if data.shape != plan.input_shape:
        raise ValueError("data shape does not match plan.input_shape")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("data must contain real numeric voxels")
    if not np.isfinite(data).all():
        raise ValueError("data must contain only finite voxels")

    try:
        source_affine = np.asarray(input_affine, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_affine must match plan.input_affine") from exc
    if source_affine.shape != (4, 4) or not np.array_equal(source_affine, plan.input_affine):
        raise ValueError("input_affine must match plan.input_affine")
    try:
        fill = float(outside_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("outside_value must be a finite real scalar") from exc
    if not math.isfinite(fill):
        raise ValueError("outside_value must be a finite real scalar")

    from scipy.ndimage import affine_transform

    # scipy.ndimage maps each output index back to a continuous input index.
    # This composition works for flips, translations, shear, and obliquity.
    output_to_input = np.linalg.inv(source_affine) @ np.asarray(plan.output_affine)
    resampled = affine_transform(
        data,
        matrix=output_to_input[:3, :3],
        offset=output_to_input[:3, 3],
        output_shape=plan.output_shape,
        output=np.float64,
        order=1,
        mode="constant",
        cval=fill,
        prefilter=False,
    )
    return ResampledVolume(resampled, plan.output_affine)
