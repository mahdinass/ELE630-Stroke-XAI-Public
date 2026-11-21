"""Generated NIfTI images and fabricated patient keys only."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "preflight"))

from ele630.data_contract import DatasetContract, OutcomePatient
from ele630.ncct import NCCTError, NCCTRepository, scan_ncct
from ele630.ncct_headers import scan_ncct_headers
from inventory_ncct import main as inventory_main


class NCCTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.dataset = self.root / "dataset"
        self.runtime = self.root / "runtime"
        self.patient = OutcomePatient("fictional-case-001", "01", "good", False, 0)
        self.contract = DatasetContract((self.patient,), (), 1, 0, 0, "synthetic", 630)
        self.patient_dir = self.dataset / "raw_data_nifti" / self.patient.patient_id
        self.patient_dir.mkdir(parents=True)
        (self.project / "outputs" / "tables").mkdir(parents=True)
        self.runtime.mkdir()
        self.manifest = self.project / "outputs" / "tables" / "patient_manifest.csv"
        self.folds = self.project / "outputs" / "tables" / "outcome_cv_folds_private.csv"
        self.manifest.touch()
        self.folds.touch()
        self.repo = NCCTRepository(self.contract, self.dataset)

    def image_file(self, name="fictional_ses-01_ncct_skull_stripped.nii.gz"):
        return self.patient_dir / "ses-01" / name

    def save_image(self, data=None, affine=None, name=None):
        path = self.image_file(name) if name else self.image_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.arange(24, dtype=np.int16).reshape((2, 3, 4)) if data is None else data
        affine = np.diag([-2.0, -3.0, 4.0, 1.0]) if affine is None else affine
        nib.save(nib.Nifti1Image(data, affine), path)
        return path

    def cli_args(self, *extra):
        return [
            "--project-root", str(self.project),
            "--manifest", str(self.manifest),
            "--folds", str(self.folds),
            "--dataset-root", str(self.dataset),
            "--runtime-root", str(self.runtime),
            *extra,
        ]

    def test_correct_nested_path_resolution(self):
        path = self.save_image()
        self.assertEqual(self.repo.resolve(self.patient), path)

    def test_missing_file(self):
        with self.assertRaisesRegex(NCCTError, "missing"):
            self.repo.resolve(self.patient)

    def test_ambiguous_multiple_candidates(self):
        self.save_image()
        self.save_image(name="another_ses-01_ncct_skull_stripped.nii.gz")
        with self.assertRaisesRegex(NCCTError, "multiple"):
            self.repo.resolve(self.patient)

    def test_followup_is_not_an_ncct_candidate(self):
        self.save_image(name="fictional_ses-02_ncct_skull_stripped.nii.gz")
        with self.assertRaisesRegex(NCCTError, "missing"):
            self.repo.resolve(self.patient)

    def test_reject_patient_outside_contract(self):
        outsider = OutcomePatient("fictional-outsider", "01", "good", False, 0)
        with self.assertRaisesRegex(NCCTError, "not in the validated"):
            self.repo.resolve(outsider)

    def test_readable_volume_and_metadata_without_gt(self):
        self.save_image()
        volume = self.repo.load(self.patient)
        self.assertEqual(volume.shape, (2, 3, 4))
        self.assertEqual(volume.spacing_mm, (2.0, 3.0, 4.0))
        self.assertEqual(volume.orientation, "LPS")
        self.assertEqual(volume.stored_dtype, "int16")
        self.assertEqual(volume.data.shape, (2, 3, 4))
        np.testing.assert_array_equal(volume.affine, np.diag([-2.0, -3.0, 4.0, 1.0]))
        self.assertIsNotNone(volume.header)

    def test_declared_nifti_scaling_is_exposed(self):
        path = self.image_file()
        path.parent.mkdir(parents=True)
        image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4))
        image.header.set_slope_inter(2.0, 10.0)
        nib.save(image, path)
        volume = self.repo.load(self.patient)
        self.assertTrue(np.all(volume.data == 12.0))
        self.assertEqual((volume.slope, volume.intercept), (2.0, 10.0))

    def test_invalid_dimensionality(self):
        self.save_image(data=np.zeros((2, 2, 2, 2), dtype=np.int16))
        with self.assertRaisesRegex(NCCTError, "three spatial dimensions"):
            self.repo.load(self.patient)

    def test_nonfinite_voxels(self):
        data = np.zeros((2, 2, 2), dtype=np.float32)
        data[0, 0, 0] = np.nan
        self.save_image(data=data)
        with self.assertRaisesRegex(NCCTError, "NaN or infinite"):
            self.repo.load(self.patient)

    def test_unreadable_nifti(self):
        path = self.image_file()
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a NIfTI")
        with self.assertRaisesRegex(NCCTError, "could not open"):
            self.repo.load(self.patient)

    def test_header_only_does_not_decode_voxels(self):
        self.save_image()
        with patch.object(nib.arrayproxy.ArrayProxy, "__array__", side_effect=AssertionError("decoded voxels")):
            metadata = self.repo.read_header(self.patient)
        self.assertEqual(metadata.orientation, "LPS")

    def test_stdlib_header_inventory_matches_nibabel_metadata(self):
        self.save_image()
        header = scan_ncct_headers(self.contract, self.dataset)
        full = scan_ncct(self.contract, self.dataset)
        self.assertTrue(header.passed)
        self.assertEqual(header.shapes, full.shapes)
        self.assertEqual(header.orientations, full.orientations)
        self.assertEqual(header.spacings, full.spacings)
        self.assertEqual(header.minima, [])

    def test_full_inventory_aggregates(self):
        self.save_image()
        inventory = scan_ncct(self.contract, self.dataset)
        self.assertTrue(inventory.passed)
        self.assertEqual(inventory.counts["ok"], 1)
        self.assertEqual((inventory.minima, inventory.maxima), ([0.0], [23.0]))
        self.assertEqual(inventory.extents, [(2.0, 6.0, 12.0)])

    def test_cli_success_hides_patient_id(self):
        self.save_image()
        output = io.StringIO()
        with patch("inventory_ncct.load_and_validate", return_value=self.contract), redirect_stdout(output):
            code = inventory_main(self.cli_args())
        self.assertEqual(code, 0)
        self.assertIn("Status: PASS", output.getvalue())
        self.assertIn("Finite / non-finite volumes: 1 / 0", output.getvalue())
        self.assertNotIn(self.patient.patient_id, output.getvalue())

    def test_cli_failure_hides_patient_id(self):
        output = io.StringIO()
        with patch("inventory_ncct.load_and_validate", return_value=self.contract), redirect_stdout(output):
            code = inventory_main(self.cli_args())
        self.assertEqual(code, 1)
        self.assertIn("Status: FAIL", output.getvalue())
        self.assertNotIn(self.patient.patient_id, output.getvalue())

    def test_cli_header_only_success(self):
        self.save_image()
        self.runtime.rmdir()  # A header-only preflight does not require the runtime directory.
        output = io.StringIO()
        with patch("inventory_ncct.load_and_validate", return_value=self.contract), \
             patch.dict(sys.modules, {"nibabel": None}), redirect_stdout(output):
            code = inventory_main(self.cli_args("--headers-only"))
        self.assertEqual(code, 0)
        self.assertIn("not checked in header-only mode", output.getvalue())

    def test_details_path_outside_private_area_rejected(self):
        output = io.StringIO()
        with redirect_stderr(output):
            code = inventory_main(self.cli_args("--details-csv", str(self.project / "tracked.csv")))
        self.assertEqual(code, 1)
        self.assertIn("must be a CSV under", output.getvalue())

    def test_optional_case_table_stays_in_ignored_output_area(self):
        self.save_image()
        private_table = self.project / "outputs" / "tables" / "ncct_inventory_private.csv"
        output = io.StringIO()
        with patch("inventory_ncct.load_and_validate", return_value=self.contract), redirect_stdout(output):
            code = inventory_main(self.cli_args("--details-csv", str(private_table)))
        self.assertEqual(code, 0)
        self.assertTrue(private_table.is_file())
        self.assertIn(self.patient.patient_id, private_table.read_text(encoding="utf-8"))
        self.assertNotIn(self.patient.patient_id, output.getvalue())


if __name__ == "__main__":
    unittest.main()
