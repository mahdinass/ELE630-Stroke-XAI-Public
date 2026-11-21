"""Stage 5C2D: fictional membership, synthetic arrays/NIfTI, no patient files."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "preflight"))
from ele630 import ncct_anatomy_qc as qc, ncct_private_output as private, ncct_splits
from ele630.data_contract import DatasetContract, FoldSummary, OutcomePatient, _fingerprint
import run_ncct_training_qc as cli

NIBABEL_AVAILABLE = importlib.util.find_spec("nibabel") is not None


def fictional_contract():
    patients = tuple(OutcomePatient(f"fictional-{fold}-{index:02d}", "A",
                                   "poor" if index < 9 else "good", not (fold == 0 and index == 0), fold)
                     for fold, size in enumerate((28, 29, 29, 29, 28)) for index in range(size))
    summaries = tuple(FoldSummary(fold, size, size - 9, 9, (("A", size),))
                      for fold, size in enumerate((28, 29, 29, 29, 28)))
    return DatasetContract(patients, summaries, 98, 45, 142,
                           _fingerprint({p.patient_id: p.fold for p in patients}), 630)


def fake_volume(data=None, affine=None, spacing=None, units="mm"):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4) if data is None else data
    affine = np.diag([2., 3., 4., 1.]) if affine is None else affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0) if spacing is None else spacing
    header = SimpleNamespace(get_xyzt_units=lambda: (units, "unknown"),
                             get_sform=lambda: affine, get_qform=lambda: affine)
    return SimpleNamespace(data=data, affine=affine, shape=data.shape,
                           spacing_mm=tuple(spacing), header=header, orientation="RAS",
                           slope=1., intercept=0., sform_code=1, qform_code=1,
                           stored_dtype=str(data.dtype))


class TrainingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.contract = fictional_contract()
        profile = replace(qc.ACTIVE_PROFILE, fingerprint=self.contract.fingerprint)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(qc, "ACTIVE_PROFILE", profile))
        self.stack.enter_context(patch.object(ncct_splits, "ACTIVE_PROFILE", profile))
        self.repo = Mock()
        self.repo.load.return_value = fake_volume()
        self.guard = qc.TrainingOnlyNCCTQC(self.contract, self.repo)

    def test_contract_and_canonical_split_enforced_before_repository_io(self):
        for changed in (replace(self.contract, fingerprint="forged"),
                        replace(self.contract, seed=631), replace(self.contract, good=97),
                        replace(self.contract, patients=self.contract.patients[:-1])):
            with self.assertRaises(qc.AnatomyQCError):
                qc.TrainingOnlyNCCTQC(changed, self.repo)
        for changed in (replace(self.guard.plan, seed=631),
                        replace(self.guard.plan, validation_fraction=.21),
                        replace(self.guard.plan, inner_train=self.guard.plan.inner_validation,
                                inner_validation=self.guard.plan.inner_train)):
            with self.assertRaises(qc.AnatomyQCError):
                qc.TrainingOnlyNCCTQC(self.contract, self.repo, plan=changed)
        self.repo.load.assert_not_called()
        self.repo.resolve.assert_not_called()

    def test_all_validation_outer_and_forged_objects_rejected_before_io(self):
        train = self.guard.plan.inner_train[0]
        forged = (replace(train), replace(train, outcome="poor" if train.outcome == "good" else "good"),
                  SimpleNamespace(**vars(train)), OutcomePatient("fictional-outsider", "A", "good", False, 1))
        for patient in (*self.guard.plan.inner_validation, *self.guard.plan.outer_test, *forged):
            with self.assertRaises(qc.AnatomyQCError):
                self.guard.load(patient)
        self.repo.load.assert_not_called()
        self.repo.resolve.assert_not_called()
        self.guard.load(train)
        self.repo.load.assert_called_once_with(train)

    def test_scan_reads_exactly_92_canonical_training_objects_no_gt_or_labels(self):
        report = qc.scan_training_qc(self.guard)
        self.assertEqual([c.args[0] for c in self.repo.load.call_args_list], list(self.guard.plan.inner_train))
        self.assertEqual(self.repo.method_calls, [unittest.mock.call.load(p) for p in self.guard.plan.inner_train])
        self.assertTrue(report["summary"]["integrity_complete"])
        self.assertFalse(report["summary"]["anatomy_certified"])
        self.assertEqual(len(report["visual_selection"]), 1)
        serialized = json.dumps(report)
        for key in ("outcome", "has_gt", "labels", "lesion"):
            self.assertNotIn('"' + key + '"', serialized)

    def test_failures_are_private_fixed_categories_and_no_partial_selection(self):
        self.repo.load.side_effect = [OSError("fictional-private/path")] + [fake_volume()] * 91
        report = qc.scan_training_qc(self.guard)
        self.assertFalse(report["summary"]["integrity_complete"])
        self.assertEqual(report["summary"]["status_counts"], {"integrity": 1, "ok": 91})
        self.assertEqual(report["visual_selection"], [])
        self.assertNotIn("fictional-private/path", json.dumps(report))

    def test_nonfinite_volume_pauses_recipe_without_exclusion_or_repair(self):
        data = np.zeros((2, 3, 4), dtype=np.float32)
        data[0, 0, 0] = np.nan
        self.repo.load.return_value = fake_volume(data)
        report = qc.scan_training_qc(self.guard)
        self.assertEqual(report["summary"]["status_counts"], {"integrity": 92})
        self.assertEqual(report["visual_selection"], [])
        self.assertTrue(np.isnan(data[0, 0, 0]))

    def test_loader_warnings_and_prints_cannot_export_private_text(self):
        def load(patient):
            print("fictional-private-path")
            print("fictional-private-path", file=sys.stderr)
            warnings.warn("fictional-private-path")
            return fake_volume()
        self.repo.load.side_effect = load
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            report = qc.scan_training_qc(self.guard)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertEqual(report["summary"]["flag_counts"], {"loader_or_measurement_warning": 92})
        self.assertNotIn("fictional-private-path", json.dumps(report))


class MeasurementTests(unittest.TestCase):
    def test_native_shapes_cell_and_center_extent_obliquity(self):
        record = qc.measure_volume(fake_volume())
        self.assertEqual(record["native_shape_xyz"], [2, 3, 4])
        self.assertEqual(record["model_shape_czyx"], [1, 4, 3, 2])
        self.assertEqual(record["affine_spacing_mm"], [2, 3, 4])
        self.assertEqual(record["axis_voxel_cell_extent_mm"], [4, 9, 16])
        self.assertEqual(record["axis_center_span_mm"], [2, 6, 12])
        self.assertEqual(record["world_voxel_cell_bounds_mm"], [[-1, -1.5, -2], [3, 7.5, 14]])
        self.assertEqual(record["maximum_axis_obliquity_degrees"], 0)

    def test_rotation_reflection_shear_and_header_consistency(self):
        theta = np.deg2rad(30)
        affine = np.eye(4)
        affine[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) @ np.diag([-2, 3, 4])
        record = qc.measure_volume(fake_volume(affine=affine, spacing=(2, 3, 4)))
        self.assertAlmostEqual(record["maximum_axis_obliquity_degrees"], 30)
        self.assertLess(record["affine_determinant_native_units"], 0)
        self.assertEqual(record["flags"], [])
        affine[0, 2] = 1
        record = qc.measure_volume(fake_volume(affine=affine, spacing=(1, 1, 1)))
        self.assertIn("header_affine_spacing_disagreement", record["flags"])
        self.assertIn("nonorthogonal_affine_axes", record["flags"])

    def test_window_saturation_zero_is_not_background_or_brain_mask(self):
        data = np.array([-100, -20, 0, 50, 120, 200], dtype=np.float32).reshape(1, 2, 3)
        before = data.copy()
        record = qc.measure_volume(fake_volume(data))
        self.assertAlmostEqual(record["transformed_zero_fraction"], 2/6)
        self.assertAlmostEqual(record["transformed_one_fraction"], 2/6)
        self.assertAlmostEqual(record["below_window_fraction_float32"], 1/6)
        self.assertAlmostEqual(record["above_window_fraction_float32"], 1/6)
        self.assertAlmostEqual(record["source_zero_fraction"], 1/6)
        self.assertAlmostEqual(record["scaled_intensity_mean"], float(np.mean(data, dtype=np.float64)))
        self.assertEqual(record["scaled_intensity_percentiles_0_1_25_50_75_99_100"][3], 25)
        np.testing.assert_array_equal(data, before)
        self.assertFalse(any("mask" in key or "brain" in key for key in record))

    def test_constant_array_flag_and_zero_background_transform(self):
        record = qc.measure_volume(fake_volume(np.zeros((2, 3, 4), dtype=np.float32)))
        self.assertEqual(record["transformed_zero_fraction"], 0)
        self.assertAlmostEqual(record["transformed_mean"], 1/7, places=6)
        self.assertEqual(record["flags"], ["constant_source_intensity", "constant_transformed_intensity"])

    def test_measurement_units_conversion_never_changes_input_affine(self):
        volume = fake_volume(affine=np.diag([.002, .003, .004, 1]), units="meter")
        before = volume.affine.copy()
        np.testing.assert_allclose(qc.measure_volume(volume)["affine_spacing_mm"], [2, 3, 4])
        np.testing.assert_array_equal(volume.affine, before)
        with self.assertRaises(qc.AnatomyQCError):
            qc.measure_volume(fake_volume(units="unknown"))

    def test_malformed_affine_scaling_dimensions_nonfinite_and_float32_overflow(self):
        for data in (np.full((2, 3, 4), np.inf), np.full((2, 3, 4), 1e100), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                qc.measure_volume(fake_volume(data))
        for affine in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([2, 3, 4, 2])):
            with self.assertRaises(ValueError):
                qc.measure_volume(fake_volume(affine=affine))
        for field, value in (("slope", np.nan), ("intercept", np.inf), ("slope", 0),
                             ("spacing_mm", (0, 1, 1)), ("shape", (1, 1, 1))):
            volume = fake_volume()
            setattr(volume, field, value)
            with self.assertRaises(ValueError):
                qc.measure_volume(volume)


class SelectionTests(unittest.TestCase):
    def cases(self):
        return [{"canonical_index": i, "patient_id": f"fictional-selector-{i}", "status": "ok",
                 "measurements": {key: 1. for _, key, _ in qc.SELECTORS}} for i in range(10)]

    def test_exact_order_eight_extrema_deduplicate_no_replacement(self):
        cases = self.cases()
        for i, (_, key, maximum) in enumerate(qc.SELECTORS):
            cases[i]["measurements"][key] = 2 if maximum else 0
        selected = qc.select_visual_cases(list(reversed(cases)))
        self.assertEqual([case["canonical_index"] for case in selected], list(range(8)))
        self.assertEqual([case["selectors"] for case in selected], [[reason] for reason, _, _ in qc.SELECTORS])
        for case in cases:
            case["measurements"] = cases[0]["measurements"].copy()
        selected = qc.select_visual_cases(list(reversed(cases)))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["canonical_index"], 0)
        self.assertEqual(selected[0]["selectors"], [reason for reason, _, _ in qc.SELECTORS])

    def test_empty_or_incomplete_qc_cannot_select_a_valid_subcohort(self):
        self.assertEqual(qc.select_visual_cases([]), [])
        cases = self.cases()
        cases[-1]["status"] = "nonfinite"
        self.assertEqual(qc.select_visual_cases(cases), [])


class PrivateAndCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "approved"
        self.root.mkdir(mode=0o700)
        self.base = self.root / "qc"
        self.base.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o700)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(private, "_APPROVED_PRIVATE_ROOT", self.root))
        stack.enter_context(patch.object(private, "_TRUSTED_CANONICAL_ROOT", self.root))

    def test_exclusive_run_and_artifact_never_overwrite(self):
        run = private.create_private_run(self.base, "fictional_qc", 0)
        qc.write_private_qc(run, {"fictional": True})
        target = run.fold_dir / "training_anatomy_qc.json"
        before = target.read_bytes()
        with self.assertRaises(private.PrivateOutputError):
            qc.write_private_qc(run, {"overwrite": True})
        with self.assertRaises(private.PrivateOutputError):
            private.create_private_run(self.base, "fictional_qc", 0)
        self.assertEqual(target.read_bytes(), before)

    def test_forged_or_outside_private_output_rejected(self):
        with self.assertRaises(private.PrivateOutputError):
            qc.write_private_qc(SimpleNamespace(fold_dir=Path(self.temporary.name)), {})
        with self.assertRaises(private.PrivateOutputError):
            private.create_private_run(Path(self.temporary.name), "fictional_qc", 0)
        run = private.create_private_run(self.base, "fictional_qc", 0)
        with self.assertRaises(private.PrivateOutputError):
            qc.write_private_qc(replace(run, fold_dir=self.root), {})

    @unittest.skipUnless(os.name == "posix", "POSIX owner/mode checks require UiS")
    def test_owner_only_permissions_and_nonprivate_base_rejected(self):
        run = private.create_private_run(self.base, "fictional_modes", 0)
        qc.write_private_qc(run, {})
        for path, mode in ((run.run_dir, 0o700), (run.fold_dir, 0o700),
                           (run.fold_dir / "training_anatomy_qc.json", 0o600)):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode)
            self.assertEqual(path.stat().st_uid, os.getuid())
        os.chmod(self.base, 0o755)
        with self.assertRaises(private.PrivateOutputError):
            private.create_private_run(self.base, "fictional_unsafe", 0)

    def test_plan_only_accesses_no_files_and_ignores_supplied_private_paths(self):
        output = io.StringIO()
        with patch.object(Path, "open", side_effect=AssertionError("I/O")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("I/O")), \
                patch.object(Path, "stat", side_effect=AssertionError("I/O")), \
                patch.object(qc, "scan_training_qc", side_effect=AssertionError("images")), redirect_stdout(output):
            self.assertEqual(cli.main(["--manifest", "fictional-private.csv"]), 0)
        self.assertNotIn("fictional-private", output.getvalue())
        self.assertFalse(json.loads(output.getvalue())["patient_data_access"])

    def test_execution_guard_before_files_and_sanitized_provenance_failure(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "read_bytes", side_effect=AssertionError("I/O")), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(cli.main(["--execute-qc"]), 2)
        self.assertEqual(stdout.getvalue(), "")
        stderr = io.StringIO()
        with patch.object(cli, "_worker_available", return_value=True), \
                redirect_stderr(stderr):
            self.assertEqual(cli.main(["--execute-qc", "--private-output-dir", str(self.base),
                                       "--run-id", "fictional_cli", "--source-revision", "fictional-private-path",
                                       "--protocol", "fictional-protocol.md"]), 1)
        self.assertNotIn("fictional-private-path", stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue())["failure_stage"], "provenance")

    def test_mocked_execution_only_prints_aggregate_summary_and_private_report(self):
        from ele630 import data_contract
        contract = fictional_contract()
        profile = replace(qc.ACTIVE_PROFILE, fingerprint=contract.fingerprint)
        repository = Mock()
        repository.load.return_value = fake_volume()
        nifti_module = SimpleNamespace(NCCTRepository=Mock(return_value=repository), __file__=qc.__file__, __name__="ele630.ncct")
        protocol = self.root / "fictional-protocol.md"
        protocol.write_text("synthetic protocol", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "_worker_available", return_value=True), \
                patch.object(cli.subprocess, "run", return_value=SimpleNamespace(stdout="a" * 40)), \
                patch.object(data_contract, "load_and_validate", return_value=contract), \
                patch.object(qc, "ACTIVE_PROFILE", profile), patch.object(ncct_splits, "ACTIVE_PROFILE", profile), \
                patch.dict(sys.modules, {"ele630.ncct": nifti_module, "nibabel": SimpleNamespace(__version__="synthetic")}), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(["--execute-qc", "--private-output-dir", str(self.base),
                               "--run-id", "fictional_execution", "--source-revision", "a" * 40,
                               "--protocol", str(protocol)])
        self.assertEqual(result, 0, stderr.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("fictional-", stdout.getvalue())
        self.assertNotIn("patient_id", stdout.getvalue())
        summary = json.loads(stdout.getvalue())
        self.assertTrue(summary["integrity_complete"])
        self.assertFalse(summary["training_authorized"])
        report = json.loads((self.base / "fictional_execution" / "outer_fold_0" / "training_anatomy_qc.json").read_text())
        self.assertEqual(len(report["cases"]), 92)
        self.assertEqual(repository.load.call_count, 92)
        self.assertEqual(report["summary"], summary)
        self.assertEqual(report["provenance"]["source_revision"], "a" * 40)

    def test_invalid_contract_cli_never_constructs_image_repository(self):
        from ele630 import data_contract
        constructor = Mock(side_effect=AssertionError("image access"))
        protocol = self.root / "fictional-protocol.md"
        protocol.write_text("synthetic protocol", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "_worker_available", return_value=True), \
                patch.object(cli.subprocess, "run", return_value=SimpleNamespace(stdout="a" * 40)), \
                patch.object(data_contract, "load_and_validate", return_value=fictional_contract()), \
                patch.dict(sys.modules, {"ele630.ncct": SimpleNamespace(NCCTRepository=constructor)}), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(["--execute-qc", "--private-output-dir", str(self.base),
                               "--run-id", "fictional_bad_contract", "--source-revision", "a" * 40,
                               "--protocol", str(protocol)])
        self.assertEqual(result, 1)
        constructor.assert_not_called()
        self.assertFalse((self.base / "fictional_bad_contract").exists())
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["failure_stage"], "contract")


@unittest.skipUnless(NIBABEL_AVAILABLE, "nibabel unavailable: generated-NIfTI integration not verified locally")
class GeneratedNiftiTests(unittest.TestCase):
    def setUp(self):
        TrainingBoundaryTests.setUp(self)
        import nibabel as nib
        from ele630.ncct import NCCTRepository
        self.nib = nib
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name)
        self.patient = self.guard.plan.inner_train[0]
        directory = self.dataset / "raw_data_nifti" / self.patient.patient_id / "ses-01"
        directory.mkdir(parents=True)
        self.image_path = directory / "fictional_ses-01_ncct_skull_stripped.nii.gz"
        self.real_repo = NCCTRepository(self.contract, self.dataset)
        self.real_guard = qc.TrainingOnlyNCCTQC(self.contract, self.real_repo)

    def save(self, data, slope=1, intercept=0):
        image = self.nib.Nifti1Image(data, np.diag([2., 3., 4., 1.]))
        image.header.set_xyzt_units("mm")
        image.header.set_slope_inter(slope, intercept)
        self.nib.save(image, self.image_path)

    def test_declared_scaling_applied_once_and_preserved_via_proxy(self):
        self.save(np.array([-100, -20, 0, 50, 120, 200], dtype=np.int16).reshape(1, 2, 3), 2, -10)
        volume = self.real_guard.load(self.patient)
        record = qc.measure_volume(volume)
        self.assertEqual(record["proxy_slope"], 2)
        self.assertEqual(record["proxy_intercept"], -10)
        self.assertEqual(record["scaled_intensity_percentiles_0_1_25_50_75_99_100"][0], -210)
        self.assertAlmostEqual(record["transformed_zero_fraction"], 2/6)

    def test_actual_nifti_malformed_and_nonfinite_fail_without_path_disclosure(self):
        self.image_path.write_bytes(b"synthetic malformed NIfTI")
        with self.assertRaises(ValueError) as raised:
            self.real_guard.load(self.patient)
        self.assertNotIn(str(self.image_path), str(raised.exception))
        self.save(np.full((2, 3, 4), np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.real_guard.load(self.patient)

    def test_real_resolver_never_called_for_forbidden_or_forged_membership(self):
        with patch.object(self.real_repo, "resolve", side_effect=AssertionError("path I/O")) as resolve:
            for patient in (*self.real_guard.plan.inner_validation, *self.real_guard.plan.outer_test, replace(self.patient)):
                with self.assertRaises(qc.AnatomyQCError):
                    self.real_guard.load(patient)
        resolve.assert_not_called()

    def test_gt_nifti_is_not_requested_or_used(self):
        self.save(np.ones((2, 3, 4), dtype=np.int16))
        lesion = self.image_path.parent / "fictional_ctp_registered_mask.nii.gz"
        lesion.write_bytes(b"synthetic forbidden GT")
        original = self.nib.load
        with patch.object(self.nib, "load", wraps=original) as load:
            qc.measure_volume(self.real_guard.load(self.patient))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(Path(load.call_args.args[0]), self.image_path)


if __name__ == "__main__":
    unittest.main()
