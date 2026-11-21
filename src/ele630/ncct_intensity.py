"""Pure, configurable NCCT intensity clipping for one native-grid model input.

Call after ``prepare_native_ncct_input``. This module changes intensities only;
it has no affine, spatial operation, file access or patient-level state.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


class NCCTIntensityError(ValueError):
    """The model-input array or explicitly supplied bounds are invalid."""


def clip_scale_ncct(
    model_input: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Return an independent float32 array: (clip(x, lower, upper)-lower)/(upper-lower).

    Bounds are required at every call. Values below/above the bounds lose their
    original distinctions. All arithmetic and the result use float32; bounds
    must remain finite and distinct when represented in float32.
    """
    if not isinstance(model_input, np.ndarray) or model_input.ndim != 4:
        raise NCCTIntensityError("model_input must be a 4D NumPy array (1, D, H, W)")
    if model_input.shape[0] != 1:
        raise NCCTIntensityError("model_input must have exactly one NCCT channel")
    if any(size == 0 for size in model_input.shape[1:]):
        raise NCCTIntensityError("model_input must have nonempty D, H and W dimensions")
    if model_input.dtype != np.dtype(np.float32):
        raise NCCTIntensityError("model_input must have float32 dtype from the native adapter")
    if not np.isfinite(model_input).all():
        raise NCCTIntensityError("model_input must contain only finite voxels")

    if (
        isinstance(lower, bool) or isinstance(upper, bool)
        or not isinstance(lower, Real) or not isinstance(upper, Real)
    ):
        raise NCCTIntensityError("lower and upper must be finite real numeric bounds")
    lower_float, upper_float = float(lower), float(upper)
    if not math.isfinite(lower_float) or not math.isfinite(upper_float) or lower_float >= upper_float:
        raise NCCTIntensityError("lower and upper must be finite with lower < upper")
    with np.errstate(over="ignore", under="ignore"):
        lower32 = np.float32(lower_float)
        upper32 = np.float32(upper_float)
        span32 = np.float32(upper_float - lower_float)
    if not (np.isfinite(lower32) and np.isfinite(upper32) and np.isfinite(span32)) or span32 <= 0 or lower32 >= upper32:
        raise NCCTIntensityError("bounds must remain finite and distinct in float32")

    result = np.array(model_input, dtype=np.float32, order="C", copy=True)
    np.clip(result, lower32, upper32, out=result)
    np.subtract(result, lower32, out=result)
    np.divide(result, span32, out=result)
    np.clip(result, 0.0, 1.0, out=result)  # Remove only float32 roundoff outside [0, 1].
    return result
