"""Stage 5B fictional CSV/contract fixtures; never read UiS files."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630 import clinical_runner as runner, clinical_baseline, ncct_private_output
from ele630.clinical_preprocessing import FEATURE_NAMES
from ele630.config import ServerPaths
from ele630.ncct_splits import plan_outer_fold
from test_clinical_preprocessing import row
from test_ncct_stage4a import fictional_contract

script_path = Path(__file__).resolve().parents[1] / "scripts/experiments/run_clinical_baseline.py"
spec = importlib.util.spec_from_file_location("clinical_runner_cli", script_path)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

HEADER = ["patient_id", *FEATURE_NAMES, "mrs_3months", "nihss_discharge"]


def records(plan):
    return [dict(row(), patient_id=p.patient_id, mrs_3months="forbidden", nihss_discharge="forbidden")
            for p in plan.inner_train + plan.inner_validation]


def csv_text(rows, header=HEADER):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class CSVAdapterTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_outer_fold(fictional_contract(per_fold=2), 0, seed=630, validation_fraction=0.20)
        self.rows = records(self.plan)

    def test_exact_development_membership_and_only_approved_features_retained(self):
        outer = [dict.fromkeys(HEADER, "invalid-private-value") for p in self.plan.outer_test]
        for source, p in zip(outer, self.plan.outer_test):
            source["patient_id"] = p.patient_id
        selected = runner.read_authorized_clinical(io.StringIO(csv_text(self.rows + outer)), self.plan)
        self.assertEqual(set(selected), {p.patient_id for p in self.plan.inner_train + self.plan.inner_validation})
        self.assertTrue(all(set(source) == set(FEATURE_NAMES) for source in selected.values()))
        self.assertNotIn("invalid-private-value", repr(selected))

    def test_excluded_features_never_indexed_even_when_unparseable(self):
        class IDOnlyCells(list):
            def __getitem__(inner, index):
                if index != 0:
                    raise AssertionError("excluded clinical feature accessed")
                return super().__getitem__(index)

        lexical_rows = [HEADER]
        lexical_rows.extend([source[name] for name in HEADER] for source in self.rows)
        lexical_rows.extend(IDOnlyCells([p.patient_id] + [object()] * (len(HEADER) - 1))
                            for p in self.plan.outer_test)
        lexical_rows.append(IDOnlyCells([""] + [object()] * (len(HEADER) - 1)))
        with patch.object(runner.csv, "reader", return_value=iter(lexical_rows)):
            selected = runner.read_authorized_clinical(io.StringIO(), self.plan)
        self.assertEqual(len(selected), len(self.plan.inner_train + self.plan.inner_validation))

    def test_duplicate_authorized_and_excluded_ids_and_missing_rows_fail(self):
        outside = dict(row(), patient_id=self.plan.outer_test[0].patient_id)
        for sources in (self.rows + [self.rows[0]], self.rows[:-1], self.rows + [outside, outside]):
            with self.subTest(count=len(sources)), self.assertRaises(runner.ClinicalRunnerError) as caught:
                runner.read_authorized_clinical(io.StringIO(csv_text(sources)), self.plan)
            self.assertNotIn("fictional", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_schema_header_width_and_csv_quoting_fail_safely(self):
        for text in ("", csv_text(self.rows, HEADER[1:]),
                     csv_text(self.rows, HEADER + ["age"]),
                     csv_text(self.rows, HEADER + [""]),
                     csv_text(self.rows) + "fictional-malformed\n",
                     csv_text(self.rows) + '"unclosed private-value'):
            with self.subTest(text=text[:20]), self.assertRaises(runner.ClinicalRunnerError) as caught:
                runner.read_authorized_clinical(io.StringIO(text), self.plan)
            self.assertNotIn("private-value", str(caught.exception))

    def test_blank_ids_discarded_and_normalization_is_anchored(self):
        blank = dict.fromkeys(HEADER, "not-clinical-data")
        blank["patient_id"] = " "
        selected = runner.read_authorized_clinical(io.StringIO(csv_text(self.rows + [blank, blank])), self.plan)
        self.assertEqual(len(selected), len(self.rows))
        for raw in ("1_999", "01-999", " SUB-STROKE_01_999 "):
            self.assertEqual(runner._identifier(raw), "sub-stroke_01_999")
        self.assertNotEqual(runner._identifier("prefix-1_999-suffix"), "sub-stroke_01_999")

    def test_invalid_plan_rejected_before_reading_stream(self):
        stream = Mock()
        for bad in (None, replace(self.plan, outer_fold=1), replace(self.plan, inner_train=self.plan.outer_test)):
            with self.assertRaises(runner.ClinicalRunnerError):
                runner.read_authorized_clinical(stream, bad)
        self.assertFalse(stream.mock_calls)


class RestrictedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "source"
        self.dataset = self.root / "dataset"
        self.runtime = self.root / "runtime"
        self.output = self.runtime / "clinical_runs"
        for directory in (self.project, self.dataset, self.runtime, self.output):
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
        self.manifest = self.project / "fictional_manifest.csv"
        self.folds = self.project / "fictional_folds.csv"
        self.manifest.write_text("synthetic contract stub", encoding="utf-8")
        self.folds.write_text("synthetic fold stub", encoding="utf-8")
        self.paths = ServerPaths(self.project, self.manifest, self.folds, self.dataset, self.runtime)
        self.contract = fictional_contract(per_fold=2)
        self.plan = plan_outer_fold(self.contract, 0, seed=630, validation_fraction=0.20)
        self.clinical = self.dataset / "fictional_clinical.csv"
        self.clinical.write_text(csv_text(records(self.plan)), encoding="utf-8", newline="")
        self.metadata = {"python": "3.10.11", **runner._APPROVED_VERSIONS}
        self.start_patch(runner, "_execution_metadata", return_value=self.metadata)
        self.contract_loader = self.start_patch(runner, "load_and_validate", return_value=self.contract)
        self.start_patch(ncct_private_output, "_APPROVED_PRIVATE_ROOT", new=self.runtime)
        self.start_patch(ncct_private_output, "_TRUSTED_CANONICAL_ROOT", new=self.runtime)

    def start_patch(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_runner(self, run_id="synthetic_run"):
        return runner.execute_clinical_runner(self.paths, self.clinical, self.output, run_id, execute_real=True)

    def fake_estimator(self):
        class Estimator:
            classes_ = np.array([0, 1])
            coef_ = np.zeros((1, 11))
            intercept_ = np.zeros(1)

            def fit(inner, x, y):
                self.training_x, self.training_y = x.copy(), y.copy()
                return inner

            def predict_proba(inner, x):
                self.validation_x = x.copy()
                return np.tile([0.7, 0.3], (len(x), 1))
        return Estimator()

    def test_canonical_plan_uses_existing_loader_and_exact_split_arguments(self):
        with patch.object(runner, "plan_outer_fold", wraps=plan_outer_fold) as split:
            actual = runner._load_plan(self.paths)
        self.contract_loader.assert_called_once_with(self.manifest, self.folds)
        split.assert_called_once_with(self.contract, 0, seed=630, validation_fraction=0.20)
        self.assertEqual(actual, self.plan)

    def test_real_stage5a_interface_train_only_fit_and_aggregate_private_output(self):
        rows = records(self.plan)
        for i, source in enumerate(rows):
            source["age"] = str(10 * (i + 1))
        self.clinical.write_text(csv_text(rows), encoding="utf-8", newline="")
        source_before = {p.name: p.read_bytes() for p in self.project.iterdir()}
        clinical_before = self.clinical.read_bytes()
        with patch.object(clinical_baseline, "_make_logistic_regression", return_value=self.fake_estimator()):
            record = self.run_runner()
        self.assertEqual(len(self.training_y), len(self.plan.inner_train))
        np.testing.assert_array_equal(self.training_y, [0 if p.outcome == "good" else 1 for p in self.plan.inner_train])
        self.assertAlmostEqual(float(self.training_x[:, 0].mean()), 0, places=6)
        self.assertTrue(np.all(self.validation_x[:, 0] > 0))
        self.assertEqual(source_before, {p.name: p.read_bytes() for p in self.project.iterdir()})
        self.assertEqual(clinical_before, self.clinical.read_bytes())
        artifact = self.output / "synthetic_run/outer_fold_0/aggregate_training.json"
        self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), record)
        self.assertEqual([p.name for p in artifact.parent.iterdir()], ["aggregate_training.json"])
        text = json.dumps(record)
        for private in ("fictional-", str(self.dataset), "patient_id", "coef_", "intercept_"):
            self.assertNotIn(private, text)
        self.assertEqual(record["dependencies"], self.metadata)
        self.assertEqual(record["results"]["logistic_regression"]["validation_count"], len(self.plan.inner_validation))
        self.assertFalse(record["outer_test_clinical_processed"])
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(artifact.parent.stat().st_mode), 0o700)

    def test_mapping_passed_to_baseline_excludes_outer_even_if_source_contains_them(self):
        extras = [dict(row(), patient_id=p.patient_id, age="invalid-private-value") for p in self.plan.outer_test]
        self.clinical.write_text(csv_text(records(self.plan) + extras), encoding="utf-8", newline="")
        def inspect_mapping(plan, mapping):
            self.assertEqual(plan, self.plan)
            self.assertEqual(set(mapping), {p.patient_id for p in plan.inner_train + plan.inner_validation})
            self.assertNotIn("invalid-private-value", repr(mapping))
            raise RuntimeError("synthetic stop before training")
        with patch.object(runner, "run_clinical_development", side_effect=inspect_mapping) as baseline:
            with self.assertRaises(runner.ClinicalRunnerError):
                self.run_runner()
        baseline.assert_called_once()

    def test_explicit_guard_and_invalid_inputs_do_not_create_outputs(self):
        with self.assertRaises(runner.ClinicalRunnerError):
            runner.execute_clinical_runner(self.paths, self.clinical, self.output, "guard")
        self.contract_loader.assert_not_called()
        for clinical, output in ((self.manifest, self.output), (self.clinical, self.project),
                                 (self.clinical, self.dataset)):
            with self.assertRaises(runner.ClinicalRunnerError):
                runner.execute_clinical_runner(self.paths, clinical, output, "guard", execute_real=True)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_invalid_contract_or_csv_fails_before_run_creation_and_fit(self):
        with patch.object(runner, "run_clinical_development") as model:
            self.contract_loader.side_effect = ValueError("fictional-secret /private/source")
            with self.assertRaises(runner.ClinicalRunnerError) as caught:
                self.run_runner()
            self.assertNotIn("fictional-secret", str(caught.exception))
            self.contract_loader.side_effect = None
            self.clinical.write_text(csv_text(records(self.plan)[:-1]), encoding="utf-8", newline="")
            with self.assertRaises(runner.ClinicalRunnerError):
                self.run_runner()
            model.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_run_rejected_before_fit_without_overwrite(self):
        with patch.object(clinical_baseline, "_make_logistic_regression", return_value=self.fake_estimator()):
            self.run_runner()
        artifact = self.output / "synthetic_run/outer_fold_0/aggregate_training.json"
        before = artifact.read_bytes()
        with patch.object(runner, "run_clinical_development") as model:
            with self.assertRaises(runner.ClinicalRunnerError):
                self.run_runner()
            model.assert_not_called()
        self.assertEqual(before, artifact.read_bytes())

    def test_same_fixed_configuration_and_aggregates_for_separate_synthetic_runs(self):
        with patch.object(clinical_baseline, "_make_logistic_regression", side_effect=lambda: self.fake_estimator()):
            a, b = self.run_runner("run_a"), self.run_runner("run_b")
        self.assertEqual(a, b)
        self.assertEqual(a["configuration"]["C"], 1)
        self.assertEqual(a["configuration"]["threshold"], 0.5)

    def test_authorized_malformed_features_fail_without_publishing_metrics(self):
        sources = records(self.plan)
        sources[0]["smoking"] = "private-invalid-value"
        self.clinical.write_text(csv_text(sources), encoding="utf-8", newline="")
        with patch.object(clinical_baseline, "_make_logistic_regression", return_value=self.fake_estimator()):
            with self.assertRaises(runner.ClinicalRunnerError) as caught:
                self.run_runner()
        self.assertNotIn("private-invalid-value", str(caught.exception))
        self.assertEqual(list((self.output / "synthetic_run/outer_fold_0").iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec("sklearn") is not None,
                         "scikit-learn is absent; full runner real-estimator gate pending")
    def test_full_runner_with_real_estimator_on_fictional_data(self):
        # Environment/contract/output roots are fictional, estimator is genuine.
        record = self.run_runner()
        self.assertEqual(record["results"]["training_count"], len(self.plan.inner_train))
        self.assertEqual(record["results"]["logistic_regression"]["validation_count"],
                         len(self.plan.inner_validation))
        self.assertEqual(record["results"]["logistic_regression"]["auroc"], 0.5)


class ExecutionBoundaryTests(unittest.TestCase):
    def test_environment_gate_rejects_local_windows_or_login_before_imports(self):
        with patch.object(runner.importlib, "import_module") as imports:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(runner.ClinicalRunnerError):
                    runner._execution_metadata()
            imports.assert_not_called()

    def test_approved_environment_and_exact_versions_gate_without_patient_data(self):
        # All trusted environment boundaries are mocked; no real server request.
        with patch.object(runner.os, "name", "posix"), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "cpu20", "SLURMD_NODENAME": "synthetic"}), \
             patch.object(runner.socket, "gethostname", return_value="synthetic"), \
             patch.object(runner, "Path") as paths, \
             patch.object(runner, "_APPROVED_PYTHON") as approved, \
             patch.object(runner.platform, "python_version", return_value="3.10.11"), \
             patch.object(runner.importlib, "import_module") as imports:
            approved.resolve.return_value = "approved-synthetic-interpreter"
            paths.return_value.resolve.return_value = "approved-synthetic-interpreter"
            imports.side_effect = lambda name: SimpleNamespace(__version__=runner._APPROVED_VERSIONS[name])
            self.assertEqual(runner._execution_metadata(), {"python": "3.10.11", **runner._APPROVED_VERSIONS})
            imports.side_effect = lambda name: SimpleNamespace(__version__="wrong-version")
            with self.assertRaises(runner.ClinicalRunnerError):
                runner._execution_metadata()

    def test_cli_missing_acknowledgement_or_bad_arguments_are_sanitized(self):
        for arguments in (["--clinical-csv", "synthetic.csv", "--output-dir", "synthetic", "--run-id", "run"],
                          ["--private-secret-argument"]):
            stderr = io.StringIO()
            with patch.object(cli, "execute_clinical_runner") as execute, redirect_stderr(stderr):
                try:
                    status = cli.main(arguments)
                except SystemExit as exc:
                    status = exc.code
                self.assertEqual(status, 2)
                execute.assert_not_called()
            self.assertNotIn("private-secret", stderr.getvalue())

    def test_cli_real_boundary_suppresses_exception_and_traceback_details(self):
        args = ["--execute-real", "--clinical-csv", "synthetic.csv", "--output-dir", "synthetic", "--run-id", "run"]
        stderr = io.StringIO()
        with patch.object(cli, "execute_clinical_runner", side_effect=ValueError("fictional-secret /private/file")), redirect_stderr(stderr):
            self.assertEqual(cli.main(args), 1)
        self.assertNotIn("fictional-secret", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
