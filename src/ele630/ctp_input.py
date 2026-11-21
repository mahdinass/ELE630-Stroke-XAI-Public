"""Pure in-memory adapter for one native-grid registered CTP frame."""

from __future__ import annotations

from numbers import Integral
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import GeometryError, _affine


class CTPInputError(ValueError):
    """A requested CTP frame cannot be used without changing its meaning."""


def validate_frame_index(frame_index: int) -> int:
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, Integral) or not 0 <= frame_index < 40:
        raise CTPInputError("frame_index must be an integer from 0 through 39")
    return int(frame_index)


@dataclass(frozen=True)
class CTPFrameInput:
    tensor: np.ndarray
    native_shape: tuple[int, int, int]
    native_affine: np.ndarray
    frame_index: int


def prepare_ctp_frame_input(
    frame: np.ndarray,
    native_affine: Sequence[Sequence[float]],
    frame_index: int,
) -> CTPFrameInput:
    """Copy (x,y,z) to contiguous float32 (1,z,y,x), without spatial processing."""
    frame_index = validate_frame_index(frame_index)
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or any(s <= 0 for s in frame.shape):
        raise CTPInputError("frame must be a nonempty 3D NumPy array in (x, y, z) order")
    if not np.issubdtype(frame.dtype, np.number) or np.iscomplexobj(frame):
        raise CTPInputError("frame must contain real numeric voxels")
    if not np.isfinite(frame).all():
        raise CTPInputError("frame must contain only finite voxels")
    if np.any((frame < 0) | (frame > 255)):
        raise CTPInputError("processed CTP frame values must be within 0–255")
    try:
        affine = np.array(_affine(native_affine), dtype=np.float64, copy=True)
    except GeometryError as exc:
        raise CTPInputError(f"native_affine is invalid: {exc}") from exc
    affine.setflags(write=False)
    tensor = np.array(frame.transpose(2, 1, 0)[np.newaxis], dtype=np.float32, order="C", copy=True)
    if not np.isfinite(tensor).all() or np.any((tensor < 0) | (tensor > 255)):
        raise CTPInputError("processed CTP values cannot be represented within 0–255 as float32")
    return CTPFrameInput(tensor, tuple(frame.shape), affine, frame_index)
