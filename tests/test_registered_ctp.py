"""Registered CTP tests use only fictional patients and synthetic arrays/NIfTIs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.ctp_input import CTPInputError, prepare_ctp_frame_input
from ele630.ctp_paths import CTPError, CTPResolver
from ele630.data_contract import DatasetContract, OutcomePatient

try:
    import nibabel as nib
except ImportError:
    nib = None


AFFINE = np.array((
    (-2.0, 0.0, 0.0, 100.0),
    (0.0, -3.0, 0.0, 200.0),
    (0.0, 0.0, 4.0, -30.0),
    (0.0, 0.0, 0.0, 1.0),
))


class CTPInputTests(unittest.TestCase):
    def test_exact_axis_mapping_variable_and_odd_shapes(self):
        for shape in ((2, 3, 4), (3, 5, 3)):
            with self.subTest(shape=shape):
                x, y, z = np.indices(shape)
                source = (40 * x + 7 * y + z).astype(np.float32)
                original = source.copy()
                affine = AFFINE.copy()
                adapted = prepare_ctp_frame_input(source, affine, 19)
                self.assertEqual(adapted.tensor.shape, (1, shape[2], shape[1], shape[0]))
                self.assertEqual(adapted.tensor.dtype, np.float32)
                self.assertTrue(adapted.tensor.flags.c_contiguous)
                self.assertEqual(adapted.tensor[0, 2, 1, 1], source[1, 1, 2])
                np.testing.assert_array_equal(adapted.tensor[0].transpose(2, 1, 0), source)
                self.assertFalse(np.shares_memory(adapted.tensor, source))
                np.testing.assert_array_equal(source, original)
                adapted.tensor[:] = -1
                np.testing.assert_array_equal(source, original)
                self.assertFalse(np.shares_memory(adapted.native_affine, affine))
                self.assertFalse(adapted.native_affine.flags.writeable)
                affine[0, 3] = 999
                self.assertEqual(adapted.native_affine[0, 3], 100)

    def test_boundary_values_and_no_clipping(self):
        source = np.array([[[0.0, 255.0]]], dtype=np.float32)
        result = prepare_ctp_frame_input(source, AFFINE, 39)
        np.testing.assert_array_equal(result.tensor.flatten(), (0, 255))
        self.assertEqual(prepare_ctp_frame_input(source, AFFINE, np.int64(1)).frame_index, 1)
        for bad in (-0.01, 255.01, np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(CTPInputError):
                    prepare_ctp_frame_input(np.array([[[bad]]]), AFFINE, 0)

    def test_invalid_index_shape_type_and_affine(self):
        source = np.zeros((2, 3, 4), dtype=np.float32)
        for index in (-1, 40, True, False, np.bool_(True), 1.0, "1", None):
            with self.subTest(index=index), self.assertRaisesRegex(CTPInputError, "frame_index"):
                prepare_ctp_frame_input(source, AFFINE, index)
        for bad in (np.zeros((2, 3)), np.zeros((2, 3, 4, 1)),
                    np.zeros((0, 2, 3)), np.zeros((1, 2, 3), dtype=bool),
                    np.zeros((1, 2, 3), dtype=np.complex64)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(CTPInputError):
                prepare_ctp_frame_input(bad, AFFINE, 0)
        singular = AFFINE.copy()
        singular[0, :3] = 0
        with self.assertRaisesRegex(CTPInputError, "singular"):
            prepare_ctp_frame_input(source, singular, 0)


class CTPResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patient = OutcomePatient("fictional-case-001", "01", "good", False, 0)
        self.contract = DatasetContract((self.patient,), (), 1, 0, 0, "synthetic", 630)
        self.resolver = CTPResolver(self.contract, self.root)
        self.patient_dir = self.root / "derivatives" / self.patient.patient_id

    def touch_ctp(self, name="fictional_ses-01_space-ncct_ctp.nii.gz"):
        target = self.patient_dir / "ses-01" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    def test_audited_nested_pattern_and_ambiguity(self):
        target = self.touch_ctp()
        self.assertEqual(self.resolver.resolve(self.patient), target)
        self.touch_ctp("another_ses-01_space-ncct_ctp.nii.gz")
        with self.assertRaisesRegex(CTPError, "multiple"):
            self.resolver.resolve(self.patient)

    def test_wrong_modality_and_missing_file(self):
        self.touch_ctp("fictional_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")
        with self.assertRaisesRegex(CTPError, "missing"):
            self.resolver.resolve(self.patient)

    def test_contract_rejection_precedes_filesystem_access(self):
        outsider = OutcomePatient("fictional-outsider", "01", "good", False, 0)
        altered = OutcomePatient(self.patient.patient_id, "01", "poor", False, 0)
        for patient in (outsider, altered, "fictional-case-001"):
            with self.subTest(patient=type(patient)), patch.object(Path, "is_dir", side_effect=AssertionError("file access")):
                with self.assertRaisesRegex(CTPError, "validated outcome contract"):
                    self.resolver.resolve(patient)

    def test_rejects_unsafe_patient_directory_name(self):
        for pid in ("", "..", "fictional/other", "fictional\\other"):
            patient = OutcomePatient(pid, "01", "good", False, 0)
            contract = DatasetContract((patient,), (), 1, 0, 0, "synthetic", 630)
            with self.subTest(pid=pid), patch.object(Path, "is_dir", side_effect=AssertionError("file access")):
                with self.assertRaisesRegex(CTPError, "safe directory name"):
                    CTPResolver(contract, self.root).resolve(patient)

    def test_rejects_symlinked_patient_or_candidate(self):
        target = self.touch_ctp()
        link = target.with_name("alternate_ses-01_space-ncct_ctp.nii.gz")
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("creating symlinks is unavailable on this Windows account")
        with self.assertRaisesRegex(CTPError, "multiple"):
            self.resolver.resolve(self.patient)
        target.unlink()
        with self.assertRaisesRegex(CTPError, "symlink"):
            self.resolver.resolve(self.patient)
        link.unlink()
        other = self.root / "derivatives" / "fictional-other"
        other.mkdir(parents=True)
        # The nested session directory is now empty; remove it before linking.
        (self.patient_dir / "ses-01").rmdir()
        self.patient_dir.rmdir()
        try:
            self.patient_dir.symlink_to(other, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("creating directory symlinks is unavailable on this Windows account")
        with self.assertRaisesRegex(CTPError, "symlink"):
            self.resolver.resolve(self.patient)


@unittest.skipUnless(nib is not None, "nibabel is unavailable locally")
class RegisteredCTPNiftiTests(unittest.TestCase):
    def setUp(self):
        from ele630.ctp import RegisteredCTPRepository

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patient = OutcomePatient("fictional-case-001", "01", "good", False, 0)
        self.contract = DatasetContract((self.patient,), (), 1, 0, 0, "synthetic", 630)
        self.ctp_path = (self.root / "derivatives" / self.patient.patient_id /
                         "ses-01" / "fictional_ses-01_space-ncct_ctp.nii.gz")
        self.ncct_path = (self.root / "raw_data_nifti" / self.patient.patient_id /
                          "ses-01" / "fictional_ses-01_ncct_skull_stripped.nii.gz")
        self.ctp_path.parent.mkdir(parents=True)
        self.ncct_path.parent.mkdir(parents=True)
        self.repo = RegisteredCTPRepository(self.contract, self.root)
        self.save_ncct((3, 5, 3))

    def save_ncct(self, shape, affine=AFFINE):
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), affine), self.ncct_path)

    def save_ctp(self, data=None, affine=AFFINE, *, path=None):
        if data is None:
            x, y, z, t = np.indices((3, 5, 3, 40))
            data = (50 * x + 10 * y + 2 * z + t).astype(np.float32)
            data[2, 4, 2, 39] = 255
        image = nib.Nifti1Image(data, affine)
        if data.ndim == 4:
            image.header.set_zooms((2.0, 3.0, 4.0, 1.5))
            image.header.set_xyzt_units(xyz="mm", t="sec")
            image.header["toffset"] = 2.5
        nib.save(image, path or self.ctp_path)
        return data

    def test_first_intermediate_last_and_temporal_metadata(self):
        data = self.save_ctp()
        for index in (0, 19, 39):
            with self.subTest(index=index):
                result = self.repo.load_frame(self.patient, index)
                self.assertEqual(result.input.frame_index, index)
                self.assertEqual(result.input.tensor.shape, (1, 3, 5, 3))
                self.assertEqual(result.input.tensor.dtype, np.float32)
                self.assertTrue(result.input.tensor.flags.c_contiguous)
                np.testing.assert_array_equal(result.input.tensor[0].transpose(2, 1, 0), data[..., index])
                self.assertEqual(result.input.tensor[0, 2, 1, 1], data[1, 1, 2, index])
                self.assertEqual(result.metadata.frame_count, 40)
                self.assertEqual(result.metadata.temporal_spacing, 1.5)
                self.assertEqual(result.metadata.temporal_unit, "sec")
                self.assertEqual(result.metadata.temporal_offset, 2.5)
                np.testing.assert_array_equal(result.input.native_affine, AFFINE)
                np.testing.assert_array_equal(result.metadata.affine, AFFINE)
        self.assertEqual(self.repo.load_frame(self.patient, 39).input.tensor.max(), 255)
        self.assertEqual(self.repo.load_frame(self.patient, 0).input.tensor.min(), 0)

    def test_open_series_validates_once_for_multiple_frames(self):
        source = self.save_ctp()
        with patch.object(self.repo, "_open", wraps=self.repo._open) as ctp_open, \
             patch.object(self.repo._ncct, "read_header", wraps=self.repo._ncct.read_header) as ncct_header, \
             patch.object(nib, "load", wraps=nib.load) as nifti_open:
            series = self.repo.open_series(self.patient)
            results = [series.load_frame(index) for index in (0, 19, 39)]
        self.assertEqual(ctp_open.call_count, 1)
        self.assertEqual(ncct_header.call_count, 1)
        self.assertEqual(nifti_open.call_count, 2)  # One CTP and one NCCT image.
        self.assertEqual(series.metadata.frame_count, 40)
        self.assertEqual(series.metadata.temporal_spacing, 1.5)
        self.assertEqual(series.metadata.temporal_unit, "sec")
        self.assertEqual(series.metadata.temporal_offset, 2.5)
        np.testing.assert_array_equal(series.metadata.affine, AFFINE)
        for index, result in zip((0, 19, 39), results):
            with self.subTest(index=index):
                self.assertIs(result.metadata, series.metadata)
                self.assertEqual(result.input.frame_index, index)
                self.assertEqual(result.input.tensor.shape, (1, 3, 5, 3))
                self.assertEqual(result.input.tensor.dtype, np.float32)
                self.assertTrue(result.input.tensor.flags.c_contiguous)
                np.testing.assert_array_equal(result.input.tensor[0].transpose(2, 1, 0), source[..., index])

    def test_series_frames_are_independent_and_read_only_on_disk(self):
        source = self.save_ctp()
        original_file = self.ctp_path.read_bytes()
        files_before = {p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()}
        series = self.repo.open_series(self.patient)
        first = series.load_frame(19).input.tensor
        second = series.load_frame(19).input.tensor
        self.assertFalse(np.shares_memory(first, source))
        self.assertFalse(np.shares_memory(first, second))
        first[:] = -1
        np.testing.assert_array_equal(second[0].transpose(2, 1, 0), source[..., 19])
        np.testing.assert_array_equal(series.load_frame(19).input.tensor, second)
        self.assertEqual(self.ctp_path.read_bytes(), original_file)
        self.assertEqual({p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()}, files_before)

    def test_series_invalid_indices_rejected_before_proxy_decode(self):
        self.save_ctp()
        series = self.repo.open_series(self.patient)
        with patch.object(nib.arrayproxy.ArrayProxy, "__getitem__", side_effect=AssertionError("frame decoded")):
            for index in (-1, 40, True, False, 1.0, "1"):
                with self.subTest(index=index), self.assertRaisesRegex(CTPError, "frame_index"):
                    series.load_frame(index)

    def test_open_series_rejects_invalid_metadata_and_unlisted_patient(self):
        outsider = OutcomePatient("fictional-outsider", "01", "good", False, 0)
        with patch.object(nib, "load", side_effect=AssertionError("NIfTI accessed")):
            with self.assertRaisesRegex(CTPError, "validated outcome contract"):
                self.repo.open_series(outsider)
        self.save_ctp(np.zeros((3, 5, 3, 39), dtype=np.float32))
        with self.assertRaisesRegex(CTPError, "40 frames"):
            self.repo.open_series(self.patient)
        shifted = AFFINE.copy()
        shifted[0, 3] += 0.01
        self.save_ctp(affine=shifted)
        with self.assertRaisesRegex(CTPError, "spatial affines"):
            self.repo.open_series(self.patient)

    def test_different_patient_needs_separate_validated_series(self):
        self.save_ctp()
        other = OutcomePatient("fictional-case-002", "01", "poor", False, 1)
        contract = DatasetContract((self.patient, other), (), 1, 1, 0, "synthetic", 630)
        other_ncct = (self.root / "raw_data_nifti" / other.patient_id /
                      "ses-01" / "fictional_ses-01_ncct_skull_stripped.nii.gz")
        other_ctp = (self.root / "derivatives" / other.patient_id /
                     "ses-01" / "fictional_ses-01_space-ncct_ctp.nii.gz")
        other_ncct.parent.mkdir(parents=True)
        other_ctp.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((3, 5, 3), dtype=np.int16), AFFINE), other_ncct)
        self.save_ctp(np.full((3, 5, 3, 40), 42, dtype=np.float32), path=other_ctp)
        from ele630.ctp import RegisteredCTPRepository

        repo = RegisteredCTPRepository(contract, self.root)
        with patch.object(repo, "resolve", wraps=repo.resolve) as lookup, \
             patch.object(repo._ncct, "read_header", wraps=repo._ncct.read_header) as ncct_header:
            first_series = repo.open_series(self.patient)
            other_series = repo.open_series(other)
            first_frame = first_series.load_frame(0)
            other_frame = other_series.load_frame(0)
        self.assertEqual(lookup.call_count, 2)
        self.assertEqual(ncct_header.call_count, 2)
        self.assertEqual(first_frame.input.tensor[0, 0, 0, 1], 50)
        self.assertEqual(other_frame.input.tensor[0, 0, 0, 1], 42)
        with self.assertRaises(TypeError):
            first_series.load_frame(other, 0)

    def test_series_uses_frame_slices_without_full_four_d_materialization(self):
        self.save_ctp()
        with patch.object(nib.arrayproxy.ArrayProxy, "__array__", side_effect=AssertionError("full proxy materialized")), \
             patch.object(nib.nifti1.Nifti1Image, "get_fdata", side_effect=AssertionError("full image materialized")):
            series = self.repo.open_series(self.patient)
            frames = [series.load_frame(index) for index in (0, 19, 39)]
        self.assertEqual([frame.input.frame_index for frame in frames], [0, 19, 39])

    def test_variable_native_shape_and_no_mutation_or_output(self):
        shape = (2, 3, 5)
        self.save_ncct(shape)
        source = np.arange(np.prod((*shape, 40)), dtype=np.float32).reshape((*shape, 40)) % 255
        self.save_ctp(source)
        before = self.ctp_path.read_bytes()
        files_before = {p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()}
        result = self.repo.load_frame(self.patient, 20)
        self.assertEqual(result.input.tensor.shape, (1, 5, 3, 2))
        np.testing.assert_array_equal(result.input.tensor[0].transpose(2, 1, 0), source[..., 20])
        result.input.tensor[:] = -5
        self.assertEqual(self.ctp_path.read_bytes(), before)
        self.assertEqual({p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()}, files_before)

    def test_header_and_single_frame_use_proxy_not_full_four_d_materialization(self):
        self.save_ctp()
        with patch.object(nib.arrayproxy.ArrayProxy, "__array__", side_effect=AssertionError("full proxy materialized")):
            header = self.repo.read_header(self.patient)
            result = self.repo.load_frame(self.patient, 19)
        self.assertEqual(header.spatial_shape, (3, 5, 3))
        self.assertEqual(result.input.tensor[0, 2, 1, 1], 83)

    def test_invalid_frame_indices_before_file_open(self):
        for index in (-1, 40, True, False, 1.0, "1"):
            with self.subTest(index=index), patch.object(self.repo, "_open", side_effect=AssertionError("file access")):
                with self.assertRaisesRegex(CTPError, "frame_index"):
                    self.repo.load_frame(self.patient, index)

    def test_contract_rejection_before_nifti_open(self):
        outsider = OutcomePatient("fictional-outsider", "01", "good", False, 0)
        with patch.object(nib, "load", side_effect=AssertionError("NIfTI accessed")):
            with self.assertRaisesRegex(CTPError, "validated outcome contract"):
                self.repo.load_frame(outsider, 0)

    def test_rejects_wrong_dimension_count_dtype_and_geometry(self):
        for data, message in (
            (np.zeros((3, 5, 3), dtype=np.float32), "four nonempty dimensions"),
            (np.zeros((3, 5, 3, 39), dtype=np.float32), "40 frames"),
            (np.zeros((3, 5, 3, 40, 1), dtype=np.float32), "four nonempty dimensions"),
            (np.zeros((3, 5, 3, 40), dtype=np.int16), "float32"),
            (np.zeros((4, 5, 3, 40), dtype=np.float32), "spatial shapes"),
        ):
            with self.subTest(shape=data.shape, dtype=data.dtype), self.assertRaisesRegex(CTPError, message):
                self.save_ctp(data)
                self.repo.load_frame(self.patient, 0)
        shifted = AFFINE.copy()
        shifted[0, 3] += 0.01
        self.save_ctp(affine=shifted)
        with self.assertRaisesRegex(CTPError, "spatial affines"):
            self.repo.load_frame(self.patient, 0)
        scaled = AFFINE.copy()
        scaled[0, 0] = -2.01
        self.save_ctp(affine=scaled)
        with self.assertRaisesRegex(CTPError, "spatial affines"):
            self.repo.load_frame(self.patient, 0)

    def test_rejects_header_spacing_mismatch_with_matching_affine(self):
        self.save_ctp()
        image = nib.load(str(self.ctp_path))
        image.header.set_zooms((2.01, 3.0, 4.0, 1.5))
        np.testing.assert_array_equal(image.affine, AFFINE)
        with patch.object(self.repo, "_open", return_value=image):
            with self.assertRaisesRegex(CTPError, "spatial spacings"):
                self.repo.read_header(self.patient)

    def test_rejects_invalid_spatial_affine_and_temporal_metadata(self):
        self.save_ctp()
        image = nib.load(str(self.ctp_path))
        singular = AFFINE.copy()
        singular[0, :3] = 0
        image._affine = singular
        with patch.object(self.repo, "_open", return_value=image):
            with self.assertRaisesRegex(CTPError, "spatial affine is invalid"):
                self.repo.read_header(self.patient)
        image = nib.load(str(self.ctp_path))
        image.header["toffset"] = np.nan
        with patch.object(self.repo, "_open", return_value=image):
            with self.assertRaisesRegex(CTPError, "temporal offset is invalid"):
                self.repo.read_header(self.patient)

    def test_rejects_nonfinite_and_out_of_range_loaded_frame(self):
        for value, message in ((np.nan, "finite"), (np.inf, "finite"), (-1, "0–255"), (256, "0–255")):
            with self.subTest(value=value):
                data = np.zeros((3, 5, 3, 40), dtype=np.float32)
                data[1, 2, 1, 19] = value
                self.save_ctp(data)
                self.assertEqual(self.repo.load_frame(self.patient, 0).input.tensor.max(), 0)
                with self.assertRaisesRegex(CTPError, message):
                    self.repo.load_frame(self.patient, 19)


if __name__ == "__main__":
    unittest.main()
