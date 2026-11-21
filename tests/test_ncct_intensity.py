"""Synthetic, small-array tests for the explicit NCCT intensity candidate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.native_ncct_input import prepare_native_ncct_input
from ele630.ncct_intensity import NCCTIntensityError, clip_scale_ncct


PILOT = {"lower": -20.0, "upper": 120.0}


class NCCTIntensityTests(unittest.TestCase):
    def test_pilot_values_shape_dtype_contiguity_and_range(self):
        source = np.array([-1024, -20, 0, 43, 120, 3071], dtype=np.float32).reshape(1, 1, 2, 3)
        result = clip_scale_ncct(source, **PILOT)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        np.testing.assert_allclose(
            result.ravel(), [0.0, 0.0, 20 / 140, 63 / 140, 1.0, 1.0], rtol=1e-6, atol=0,
        )
        self.assertEqual(result[0, 0, 0, 0], result[0, 0, 0, 1])  # Both below/at lower.
        self.assertEqual(result[0, 0, 1, 1], result[0, 0, 1, 2])  # Both at/above upper.

    def test_source_is_not_mutated_or_aliased(self):
        source = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4).transpose(0, 1, 3, 2)
        original = source.copy()
        self.assertFalse(source.flags.c_contiguous)
        result = clip_scale_ncct(source, lower=0.0, upper=10.0)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, result))
        np.testing.assert_array_equal(source, original)
        result[0, 0, 0, 0] = 0.9
        np.testing.assert_array_equal(source, original)

    def test_two_small_variable_native_like_shapes(self):
        for shape in ((1, 2, 3, 4), (1, 4, 2, 3)):
            with self.subTest(shape=shape):
                source = np.linspace(-40, 140, num=np.prod(shape), dtype=np.float32).reshape(shape)
                result = clip_scale_ncct(source, **PILOT)
                self.assertEqual(result.shape, shape)
                expected = np.clip(source, -20, 120).astype(np.float32)
                expected = (expected + np.float32(20)) / np.float32(140)
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_identical_inputs_and_explicit_bounds_are_deterministic(self):
        source = np.array([[[[-20, 0, 43, 120]]]], dtype=np.float32)
        first = clip_scale_ncct(source, **PILOT)
        second = clip_scale_ncct(source.copy(), **PILOT)
        np.testing.assert_array_equal(first, second)
        different = clip_scale_ncct(source, lower=0.0, upper=100.0)
        self.assertNotEqual(float(first[0, 0, 0, 2]), float(different[0, 0, 0, 2]))

    def test_rejects_invalid_shapes_channels_and_dtype(self):
        valid = np.zeros((1, 2, 3, 4), dtype=np.float32)
        for data, message in (
            (valid[0], "4D NumPy array"),
            (valid[np.newaxis], "4D NumPy array"),
            (np.zeros((2, 2, 3, 4), dtype=np.float32), "one NCCT channel"),
            (np.zeros((1, 0, 3, 4), dtype=np.float32), "nonempty"),
            (valid.astype(np.float64), "float32 dtype"),
            (valid.astype(np.int16), "float32 dtype"),
            (valid.astype(np.complex64), "float32 dtype"),
        ):
            with self.subTest(message=message, shape=data.shape, dtype=data.dtype):
                with self.assertRaisesRegex(NCCTIntensityError, message):
                    clip_scale_ncct(data, **PILOT)

    def test_rejects_nonfinite_voxels_and_invalid_bounds(self):
        valid = np.zeros((1, 2, 3, 4), dtype=np.float32)
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(voxel=bad):
                source = valid.copy()
                source[0, 1, 2, 3] = bad
                with self.assertRaisesRegex(NCCTIntensityError, "finite voxels"):
                    clip_scale_ncct(source, **PILOT)
        for lower, upper in ((1, 1), (2, 1), (np.nan, 1), (-1, np.inf), ("-20", 120), (0, 1e-50)):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaisesRegex(NCCTIntensityError, "bounds|lower < upper"):
                    clip_scale_ncct(valid, lower=lower, upper=upper)

    def test_composes_with_native_adapter_without_changing_affine_or_shape(self):
        native = np.zeros((2, 3, 4), dtype=np.int16)
        native[0, 0, 0] = -1024
        native[1, 2, 3] = 43
        native[1, 0, 1] = 3071
        affine = np.array((
            (-2.0, 0.0, 0.0, 100.0),
            (0.0, -3.0, 0.0, 200.0),
            (0.0, 0.0, 4.0, -30.0),
            (0.0, 0.0, 0.0, 1.0),
        ))
        adapted = prepare_native_ncct_input(native, affine)
        original_adapter_tensor = adapted.tensor.copy()
        result = clip_scale_ncct(adapted.tensor, **PILOT)
        self.assertEqual(adapted.native_shape, (2, 3, 4))
        self.assertEqual(result.shape, (1, 4, 3, 2))
        self.assertAlmostEqual(float(result[0, 3, 2, 1]), 63 / 140, places=6)
        self.assertEqual(result[0, 0, 0, 0], 0.0)
        self.assertEqual(result[0, 1, 0, 1], 1.0)
        np.testing.assert_array_equal(adapted.native_affine, affine)
        np.testing.assert_array_equal(adapted.tensor, original_adapter_tensor)
        self.assertEqual(native[1, 2, 3], 43)


if __name__ == "__main__":
    unittest.main()
