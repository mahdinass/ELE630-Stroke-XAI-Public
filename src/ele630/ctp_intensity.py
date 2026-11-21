"""Fixed intensity scaling of Stage 4O registered CTP frame inputs."""

from __future__ import annotations

import numpy as np


CTP_INTENSITY_DIVISOR = 255.0


def scale_ctp_input(model_input: np.ndarray) -> np.ndarray:
    """Copy native-adapter float32 (1,z,y,x) values to x/255, without clipping."""
    if (not isinstance(model_input, np.ndarray) or model_input.ndim != 4
            or model_input.shape[0] != 1 or any(s <= 0 for s in model_input.shape[1:])):
        raise ValueError("CTP model input must have nonempty shape (1, D, H, W)")
    if model_input.dtype != np.dtype(np.float32):
        raise ValueError("CTP model input must be float32 from the frame adapter")
    if not np.isfinite(model_input).all():
        raise ValueError("CTP model input must be finite")
    if np.any((model_input < 0) | (model_input > 255)):
        raise ValueError("CTP model input must be within 0–255")
    result = np.array(model_input, dtype=np.float32, order="C", copy=True)
    result /= np.float32(CTP_INTENSITY_DIVISOR)
    return result
