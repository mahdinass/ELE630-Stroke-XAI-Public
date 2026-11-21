"""Small synthetic arrays only; no loader, patient files or optional imaging packages."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.native_ncct_input import NativeNCCTInputError, prepare_native_ncct_input


AFFINE = np.array((
    (-2.0, 0.0, 0.0, 100.0),
    (0.0, -3.0, 0.0, 200.0),
    (0.0, 0.0, 4.0, -30.0),
    (0.0, 0.0, 0.0, 1.0),
))


class NativeNCCTInputTests(unittest.TestCase):
    def test_exact_coordinate_mapping_and_reversible_axes(self):
        x, y, z = np.indices((2, 3, 4))
        native = (100 * x + 10 * y + z).astype(np.int16)
        result = prepare_native_ncct_input(native, AFFINE)

        self.assertEqual(result.native_shape, (2, 3, 4))
        self.assertEqual(result.tensor.shape, (1, 4, 3, 2))
        self.assertEqual(result.tensor.dtype, np.float32)
        self.assertTrue(result.tensor.flags.c_contiguous)
        self.assertEqual(result.tensor_nbytes, 2 * 3 * 4 * 4)
        self.assertEqual(result.native_axis_order, ("x", "y", "z"))
        self.assertEqual(result.tensor_axis_order, ("C", "z", "y", "x"))
        self.assertEqual(result.tensor[0, 3, 2, 1], 123.0)
        self.assertEqual(result.tensor[0, 1, 0, 1], 101.0)
        self.assertEqual(result.tensor[0, 2, 1, 0], 12.0)
        np.testing.assert_array_equal(result.tensor[0].transpose(2, 1, 0), native)
        self.assertFalse(np.shares_memory(result.tensor, native))

    def test_native_affine_is_separate_and_requires_restored_axis_order(self):
        affine = AFFINE.copy()
        result = prepare_native_ncct_input(np.zeros((2, 3, 4), dtype=np.float32), affine)
        np.testing.assert_array_equal(result.native_affine, AFFINE)
        self.assertFalse(np.shares_memory(result.native_affine, affine))
        self.assertFalse(result.native_affine.flags.writeable)

        tensor_index = (0, 3, 2, 1)
        native_index = (tensor_index[3], tensor_index[2], tensor_index[1])
        world = result.native_affine @ np.array((*native_index, 1.0))
        np.testing.assert_array_equal(world, (98.0, 194.0, -18.0, 1.0))
        self.assertFalse(np.array_equal(
            world, result.native_affine @ np.array((*tensor_index[1:], 1.0)),
        ))
        affine[0, 3] = 999.0
        self.assertEqual(result.native_affine[0, 3], 100.0)

    def test_variable_y_and_z_lengths_are_not_padded_or_cropped(self):
        for shape in ((2, 3, 4), (2, 5, 2), (3, 2, 5)):
            with self.subTest(shape=shape):
                source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                result = prepare_native_ncct_input(source, AFFINE)
                self.assertEqual(result.tensor.shape, (1, shape[2], shape[1], shape[0]))
                self.assertEqual(result.tensor.size, source.size)
                np.testing.assert_array_equal(result.tensor[0].transpose(2, 1, 0), source)

    def test_int16_and_float32_values_and_declared_float32_rounding(self):
        integers = np.array([[[-1024, 0], [37, 305]]], dtype=np.int16)
        converted_int = prepare_native_ncct_input(integers, AFFINE)
        np.testing.assert_array_equal(converted_int.tensor[0].transpose(2, 1, 0), integers)
        self.assertEqual(converted_int.tensor.dtype, np.float32)

        floats = np.array([[[0.125, -12.5], [80.25, 1.5]]], dtype=np.float32)
        converted_float = prepare_native_ncct_input(floats, AFFINE)
        np.testing.assert_array_equal(converted_float.tensor[0].transpose(2, 1, 0), floats)

        high_precision = np.array([[[16_777_217.0]]], dtype=np.float64)
        rounded = prepare_native_ncct_input(high_precision, AFFINE)
        self.assertEqual(rounded.tensor[0, 0, 0, 0], np.float32(16_777_216.0))
        self.assertEqual(high_precision[0, 0, 0], 16_777_217.0)

    def test_rejects_invalid_dimensions_and_voxel_types(self):
        invalid_cases = (
            (np.zeros((2, 3)), "3D NumPy array"),
            (np.zeros((2, 3, 4, 1)), "3D NumPy array"),
            (np.zeros((2, 0, 3)), "nonempty"),
            (np.zeros((2, 2, 2), dtype=bool), "real numeric"),
            (np.zeros((2, 2, 2), dtype=np.complex64), "real numeric"),
            (np.full((2, 2, 2), "x"), "real numeric"),
        )
        for data, message in invalid_cases:
            with self.subTest(shape=data.shape, dtype=data.dtype):
                with self.assertRaisesRegex(NativeNCCTInputError, message):
                    prepare_native_ncct_input(data, AFFINE)

    def test_rejects_nonfinite_voxels_and_float32_overflow(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                data = np.zeros((2, 2, 2), dtype=np.float64)
                data[1, 0, 1] = invalid
                with self.assertRaisesRegex(NativeNCCTInputError, "finite voxels"):
                    prepare_native_ncct_input(data, AFFINE)
        with self.assertRaisesRegex(NativeNCCTInputError, "finite float32"):
            prepare_native_ncct_input(np.array([[[1e100]]]), AFFINE)

    def test_rejects_invalid_affines(self):
        bad_last_row = AFFINE.copy()
        bad_last_row[3, 0] = 0.01
        singular = AFFINE.copy()
        singular[0, :3] = 0.0
        nonfinite = AFFINE.copy()
        nonfinite[0, 3] = np.nan
        for affine, message in (
            (np.eye(3), "4×4"),
            (bad_last_row, "homogeneous last row"),
            (singular, "singular"),
            (nonfinite, "finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(NativeNCCTInputError, message):
                    prepare_native_ncct_input(np.zeros((2, 2, 2)), affine)

    def test_import_and_conversion_do_not_need_loader_resampler_or_qc(self):
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        script = (
            "import sys, numpy as np\n"
            f"sys.path.insert(0, {source_root!r})\n"
            "for name in ('ele630.ncct', 'ele630.resampling', "
            "'ele630.resampling_qc', 'nibabel', 'scipy'):\n"
            "    sys.modules[name] = None\n"
            "from ele630.native_ncct_input import prepare_native_ncct_input\n"
            "result = prepare_native_ncct_input(np.array([[[7]]], dtype=np.int16), np.eye(4))\n"
            "assert result.tensor[0, 0, 0, 0] == 7.0\n"
        )
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
