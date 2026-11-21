"""Synthetic voxel and physical-coordinate checks for NCCT resampling."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.geometry import plan_output_grid
from ele630.resampling import resample_to_grid


LPS = (
    (-1.0, 0.0, 0.0, 10.0),
    (0.0, -1.0, 0.0, 20.0),
    (0.0, 0.0, 1.0, 30.0),
    (0.0, 0.0, 0.0, 1.0),
)


def ramp(shape: tuple[int, int, int]) -> np.ndarray:
    i, j, k = np.indices(shape)
    return (100 * i + 10 * j + k).astype(np.float64)


class ResamplingValidationTests(unittest.TestCase):
    def test_rejects_incompatible_shape_and_affine_before_sampling(self):
        plan = plan_output_grid(
            (3, 3, 3), LPS, target_spacing_mm=(1, 1, 1), target_shape=(3, 3, 3),
        )
        with self.assertRaisesRegex(ValueError, "3D NumPy array"):
            resample_to_grid(np.zeros((3, 3)), LPS, plan, outside_value=-1)
        with self.assertRaisesRegex(ValueError, "data shape"):
            resample_to_grid(np.zeros((3, 3, 2)), LPS, plan, outside_value=-1)
        shifted = np.array(LPS)
        shifted[0, 3] += 1
        with self.assertRaisesRegex(ValueError, "input_affine"):
            resample_to_grid(np.zeros((3, 3, 3)), shifted, plan, outside_value=-1)
        with self.assertRaisesRegex(ValueError, "input_affine"):
            resample_to_grid(np.zeros((3, 3, 3)), np.eye(3), plan, outside_value=-1)

    def test_rejects_nonfinite_voxels_and_fill(self):
        plan = plan_output_grid(
            (3, 3, 3), LPS, target_spacing_mm=(1, 1, 1), target_shape=(3, 3, 3),
        )
        data = np.zeros((3, 3, 3))
        data[1, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite voxels"):
            resample_to_grid(data, LPS, plan, outside_value=-1)
        with self.assertRaisesRegex(ValueError, "outside_value"):
            resample_to_grid(np.zeros((3, 3, 3)), LPS, plan, outside_value=np.inf)


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "SciPy is not installed locally")
class ResamplingVoxelTests(unittest.TestCase):
    def test_identity_resampling_preserves_every_voxel(self):
        data = ramp((3, 4, 2))
        plan = plan_output_grid(
            data.shape, LPS, target_spacing_mm=(1, 1, 1), target_shape=data.shape,
        )
        result = resample_to_grid(data, LPS, plan, outside_value=-999)
        np.testing.assert_array_equal(result.data, data)
        self.assertEqual(result.affine, plan.output_affine)
        self.assertEqual(plan.output_index_to_world((2, 1, 1)), (8.0, 19.0, 31.0))
        self.assertEqual(result.data[2, 1, 1], 211.0)

    def test_translation_and_continuous_physical_mapping(self):
        source = (
            (-2.0, 0.0, 0.0, 30.0),
            (0.0, -2.0, 0.0, -40.0),
            (0.0, 0.0, 2.0, 5.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        plan = plan_output_grid(
            (3, 3, 3), source, target_spacing_mm=(1, 1, 1), target_shape=(5, 5, 5),
        )
        result = resample_to_grid(ramp((3, 3, 3)).astype(np.int16), source, plan, outside_value=-999)
        self.assertEqual(plan.output_index_to_world((1, 2, 3)), (29.0, -42.0, 8.0))
        self.assertEqual(result.data.dtype, np.float64)
        self.assertAlmostEqual(result.data[1, 2, 3], 61.5)  # source index (0.5, 1, 1.5)
        self.assertEqual(result.data[2, 2, 2], 111.0)

    def test_ras_input_is_flipped_into_lps_output_axes(self):
        ras = (
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        data = ramp((3, 4, 2))
        plan = plan_output_grid(
            data.shape, ras, target_spacing_mm=(1, 1, 1), target_shape=data.shape,
        )
        result = resample_to_grid(data, ras, plan, outside_value=-999)
        np.testing.assert_array_equal(result.data, data[::-1, ::-1, :])
        self.assertEqual(plan.output_index_to_world((0, 0, 1)), (12.0, 23.0, 31.0))
        self.assertEqual(result.data[0, 0, 1], 231.0)

    def test_oblique_input_uses_inverse_affine(self):
        c = math.sqrt(0.5)
        oblique = (
            (c, -c, 0.0, 10.0),
            (c, c, 0.0, 20.0),
            (0.0, 0.0, 1.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        plan = plan_output_grid(
            (5, 5, 3), oblique, target_spacing_mm=(1, 1, 1), target_shape=(5, 5, 3),
        )
        result = resample_to_grid(ramp((5, 5, 3)), oblique, plan, outside_value=-999)
        center = plan.output_index_to_world((2, 2, 1))
        self.assertAlmostEqual(plan.output_index_to_world((3, 2, 1))[0], center[0] - 1)
        self.assertAlmostEqual(plan.output_index_to_world((3, 2, 1))[1], center[1])
        # The output step (-1, 0, 0) maps to source step (-c, +c, 0).
        self.assertAlmostEqual(result.data[3, 2, 1], 221.0 - 90.0 * c, places=10)
        self.assertAlmostEqual(result.data[2, 2, 1], 221.0)

    def test_output_shape_affine_and_even_grid_center(self):
        source = (
            (-2.0, 0.0, 0.0, 9.0),
            (0.0, -3.0, 0.0, -5.0),
            (0.0, 0.0, 4.0, 2.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        plan = plan_output_grid(
            (3, 3, 3), source, target_spacing_mm=(2, 3, 4), target_shape=(4, 4, 4),
        )
        result = resample_to_grid(ramp((3, 3, 3)), source, plan, outside_value=-999)
        self.assertEqual(result.data.shape, (4, 4, 4))
        self.assertEqual(result.affine, plan.output_affine)
        self.assertEqual(plan.output_index_to_world((1.5, 1.5, 1.5)), plan.center_world_mm)
        self.assertAlmostEqual(result.data[1, 1, 1], 55.5)  # source index (0.5, 0.5, 0.5)

    def test_explicit_constant_fill_outside_source(self):
        data = np.full((3, 3, 3), 7.0)
        plan = plan_output_grid(
            data.shape, LPS, target_spacing_mm=(1, 1, 1), target_shape=(5, 5, 5),
        )
        result = resample_to_grid(data, LPS, plan, outside_value=-123.5)
        self.assertEqual(plan.output_index_to_world((0, 0, 0)), (11.0, 21.0, 29.0))
        self.assertEqual(result.data[0, 0, 0], -123.5)
        self.assertEqual(result.data[1, 1, 1], 7.0)
        self.assertEqual(result.data[4, 4, 4], -123.5)


if __name__ == "__main__":
    unittest.main()
