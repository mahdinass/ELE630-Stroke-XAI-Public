"""Deterministic physical-grid planning; no voxel interpolation or cropping.

Input affines are NIfTI RAS+ voxel-to-world matrices. The planned output has
fixed LPS voxel-axis directions in that world frame. Target spacing and shape
must be supplied explicitly; this module does not choose preprocessing policy.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence


Matrix4 = tuple[tuple[float, float, float, float], ...]
Triplet = tuple[float, float, float]
Shape3 = tuple[int, int, int]


class GeometryError(ValueError):
    """A requested grid cannot be planned from the supplied geometry."""


def _shape(values: Sequence[int], name: str) -> Shape3:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise GeometryError(f"{name} must contain three positive integers") from exc
    if len(result) != 3 or any(type(value) is not int or value <= 0 for value in result):
        raise GeometryError(f"{name} must contain three positive integers")
    return result  # type: ignore[return-value]


def _spacing(values: Sequence[float]) -> Triplet:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise GeometryError("target_spacing_mm must contain three finite positive values") from exc
    if len(result) != 3 or any(not math.isfinite(value) or value <= 0 for value in result):
        raise GeometryError("target_spacing_mm must contain three finite positive values")
    return result  # type: ignore[return-value]


def _affine(values: Sequence[Sequence[float]]) -> Matrix4:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in values)
    except (TypeError, ValueError) as exc:
        raise GeometryError("input_affine must be a finite 4×4 voxel-to-world matrix") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise GeometryError("input_affine must be a finite 4×4 voxel-to-world matrix")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise GeometryError("input_affine must be finite")
    if any(abs(rows[3][axis] - expected) > 1e-8 for axis, expected in enumerate((0, 0, 0, 1))):
        raise GeometryError("input_affine must have homogeneous last row [0, 0, 0, 1]")
    a, b, c = rows[0][:3]
    d, e, f = rows[1][:3]
    g, h, k = rows[2][:3]
    determinant = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        raise GeometryError("input_affine spatial transform is singular")
    return rows  # type: ignore[return-value]


def _world(affine: Matrix4, index: Triplet) -> Triplet:
    return tuple(
        sum(affine[axis][column] * index[column] for column in range(3)) + affine[axis][3]
        for axis in range(3)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class GridPlan:
    input_shape: Shape3
    input_affine: Matrix4
    target_spacing_mm: Triplet
    output_shape: Shape3
    output_affine: Matrix4
    center_world_mm: Triplet

    def input_index_to_world(self, index: Triplet) -> Triplet:
        return _world(self.input_affine, index)

    def output_index_to_world(self, index: Triplet) -> Triplet:
        return _world(self.output_affine, index)

    def input_corner_output_bounds(self) -> tuple[tuple[float, float], ...]:
        """Continuous output-index bounds of the eight input voxel centers."""
        corners = itertools.product(*((0, size - 1) for size in self.input_shape))
        indices = []
        for corner in corners:
            world = self.input_index_to_world(corner)
            indices.append(tuple(
                (world[axis] - self.output_affine[axis][3]) / self.output_affine[axis][axis]
                for axis in range(3)
            ))
        return tuple(
            (min(index[axis] for index in indices), max(index[axis] for index in indices))
            for axis in range(3)
        )

    def contains_input_voxel_centers(self, tolerance: float = 1e-6) -> bool:
        """Geometric full-input check, not an anatomical brain-support check."""
        return all(
            low >= -tolerance and high <= size - 1 + tolerance
            for (low, high), size in zip(self.input_corner_output_bounds(), self.output_shape)
        )


def plan_output_grid(
    input_shape: Sequence[int],
    input_affine: Sequence[Sequence[float]],
    *,
    target_spacing_mm: Sequence[float],
    target_shape: Sequence[int],
) -> GridPlan:
    """Center a fixed-axis LPS output lattice on the input grid's world center.

    Even dimensions use the half-index center (N-1)/2. Coordinates are voxel
    centers. No source voxel values are read or transformed.
    """
    source_shape = _shape(input_shape, "input_shape")
    affine = _affine(input_affine)
    spacing = _spacing(target_spacing_mm)
    output_shape = _shape(target_shape, "target_shape")
    source_center = tuple((size - 1) / 2 for size in source_shape)
    center_world = _world(affine, source_center)
    directions = (-spacing[0], -spacing[1], spacing[2])
    translation = tuple(
        center_world[axis] - directions[axis] * (output_shape[axis] - 1) / 2
        for axis in range(3)
    )
    output_affine: Matrix4 = (
        (directions[0], 0.0, 0.0, translation[0]),
        (0.0, directions[1], 0.0, translation[1]),
        (0.0, 0.0, directions[2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    return GridPlan(source_shape, affine, spacing, output_shape, output_affine, center_world)
