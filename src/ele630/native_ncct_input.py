"""Pure in-memory NCCT adapter for a single native NIfTI voxel array.

Native array axes are (x, y, z). Model-input axes are (C, D, H, W), with
C=0, D=z, H=y and W=x. The native affine maps (x, y, z) voxel indices to
RAS+ world coordinates; it does not directly map (D, H, W) tensor indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import GeometryError, _affine


class NativeNCCTInputError(ValueError):
    """The native voxel array or affine cannot be adapted as supplied."""


@dataclass(frozen=True)
class NativeNCCTInput:
    tensor: np.ndarray
    native_shape: tuple[int, int, int]
    native_affine: np.ndarray
    native_axis_order: tuple[str, str, str]
    tensor_axis_order: tuple[str, str, str, str]
    tensor_nbytes: int


def prepare_native_ncct_input(
    native_data: np.ndarray,
    native_affine: Sequence[Sequence[float]],
) -> NativeNCCTInput:
    """Copy one native 3D volume to contiguous float32 (1, z, y, x) order.

    Every voxel is retained. There is no resampling, spatial transform,
    intensity transform beyond float32 conversion, or background replacement.
    Values that cannot remain finite in float32 are rejected.
    """
    if not isinstance(native_data, np.ndarray) or native_data.ndim != 3:
        raise NativeNCCTInputError("native_data must be a 3D NumPy array in (x, y, z) order")
    if any(size == 0 for size in native_data.shape):
        raise NativeNCCTInputError("native_data must have nonempty x, y and z dimensions")
    if not np.issubdtype(native_data.dtype, np.number) or np.iscomplexobj(native_data):
        raise NativeNCCTInputError("native_data must contain real numeric voxels")
    if not np.isfinite(native_data).all():
        raise NativeNCCTInputError("native_data must contain only finite voxels")

    try:
        validated_affine = _affine(native_affine)
    except GeometryError as exc:
        raise NativeNCCTInputError(f"native_affine is invalid: {exc}") from exc
    preserved_affine = np.array(validated_affine, dtype=np.float64, copy=True)
    preserved_affine.setflags(write=False)

    with np.errstate(over="ignore", invalid="ignore"):
        tensor = np.array(
            native_data.transpose(2, 1, 0)[np.newaxis, ...],
            dtype=np.float32,
            order="C",
            copy=True,
        )
    if not np.isfinite(tensor).all():
        raise NativeNCCTInputError("native_data cannot be converted to finite float32 voxels")

    return NativeNCCTInput(
        tensor=tensor,
        native_shape=tuple(native_data.shape),
        native_affine=preserved_affine,
        native_axis_order=("x", "y", "z"),
        tensor_axis_order=("C", "z", "y", "x"),
        tensor_nbytes=int(tensor.nbytes),
    )
