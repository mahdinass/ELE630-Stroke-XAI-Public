"""Fictional-only Stage 5C3 numerical, fit-boundary and artifact tests."""
from __future__ import annotations
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import patch, Mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ele630 import fusion, fusion_runtime as runtime
from ele630.fusion_synthetic import fictional_plan, fictional_rows
from ele630.clinical_preprocessing import ClinicalPreprocessor

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class GuardedRows(dict):
    def __init__(self, plan):
        super().__init__(fictional_rows(plan))
        self.plan, self.fitted = plan, False

    def __getitem__(self, key):
        if key in {p.patient_id for p in self.plan.outer_test}:
            raise AssertionError("outer clinical lookup")
        if key in {p.patient_id for p in self.plan.inner_validation} and not self.fitted:
            raise AssertionError("validation lookup before all fits")
        return super().__getitem__(key)


def fake_result():
    y, p = np.array([0, 1], dtype=int), np.array([.2, .7])
    return {
        "metrics": {m: fusion.metric_record(y, p) for m in runtime.MODELS},
        "fusion_minus_reference": {m: {k: 0. for k in ("bce", "auroc", "average_precision", "brier_score", "balanced_accuracy_0_5", "probability_sd")}
                                  for m in runtime.MODELS - {"fusion", "training_prior"}},
        "training_count": 18, "validation_count": 2, "image_components": 8,
        "clinical_features": 11, "fusion_features": 19, "outer_test_evaluated": False,
    }


class FusionSourceTests(unittest.TestCase):
    def test_embedding_contract_rejects_wrong_shape_and_nonfinite(self):
        for value in (np.zeros((18, 127)), np.zeros((17, 128)), np.full((18, 128), np.nan)):
            with self.assertRaises(fusion.FusionError):
                fusion.matrix(value, 18)

    def test_metrics_numerical_and_dispersion(self):
        result = fusion.metric_record(np.array([0, 1]), np.array([.25, .75]))
        self.assertAlmostEqual(result["bce"], -np.log(.75))
        self.assertEqual(result["probability_sd"], .25)
        self.assertEqual(result["brier_score"], .0625)
        self.assertEqual(result["auroc"], 1.)

    def test_metric_shape_and_probability_fail_closed(self):
        for p in ([.2], [np.nan, .7], [-.1, 1.1], [[.2], [.7]]):
            with self.assertRaises(Exception):
                fusion.metric_record(np.array([0, 1]), p)

    def test_preprocessor_is_training_only(self):
        plan = fictional_plan()
        rows = fictional_rows(plan)
        first = ClinicalPreprocessor().fit_training(plan, rows)
        for p in plan.inner_validation:
            rows[p.patient_id]["age"] = 999999
        second = ClinicalPreprocessor().fit_training(plan, rows)
        for a, b in zip(first._state, second._state):
            np.testing.assert_array_equal(a, b)

    def test_split_overlap_rejected_before_fit(self):
        plan = fictional_plan()
        with self.assertRaises(Exception):
            fusion.run_fusion(replace(plan, inner_validation=plan.inner_train[:6]), {},
                              np.zeros((18, 128)), np.zeros((6, 128)), [.5] * 6)

    def test_all_fits_precede_validation_access_and_outputs_are_aggregate(self):
        plan = fictional_plan()
        rows = GuardedRows(plan)
        fits = []
        class FakePCA:
            def fit(self, x):
                np.testing.assert_array_equal(x, np.ones((18, 128)))
                return self
            def transform(self, x):
                return x[:, :8]
        def fit(x, y):
            fits.append(x.copy())
            if len(fits) == 3:
                rows.fitted = True
            return Mock(predict_proba=lambda v: np.tile([.7, .3], (len(v), 1)))
        with patch.object(fusion, "ImagePCA", FakePCA), patch.object(fusion, "_fit", fit):
            result = fusion.run_fusion(plan, rows, np.ones((18, 128)), np.full((6, 128), 999), [.3] * 6)
        self.assertEqual([x.shape for x in fits], [(18, 11), (18, 19), (18, 19)])
        np.testing.assert_array_equal(fits[2][:, :8], 0)
        self.assertEqual(set(result["metrics"]), runtime.MODELS)
        expected_shrinkage = .5 * .3 + .5 * (6 / 18)
        self.assertAlmostEqual(result["metrics"]["clinical_prior_shrinkage_0_5"]["mean_predicted_probability"], expected_shrinkage)
        runtime.validate_result(result)
        text = json.dumps(result)
        for forbidden in ("fictional-", "coefficients", "embedding", "patient_id", "predictions"):
            self.assertNotIn(forbidden, text)

    def test_estimator_warnings_fail_without_retry(self):
        import warnings
        model = Mock()
        def warn(*args):
            warnings.warn("fictional nonconvergence")
        model.fit.side_effect = warn
        with patch.object(fusion, "_make_logistic_regression", return_value=model):
            with self.assertRaises(fusion.FusionError):
                fusion._fit(np.zeros((18, 19)), np.zeros(18))
        self.assertEqual(model.fit.call_count, 1)

    def test_allowlist_rejects_individual_artifacts(self):
        for section, value in (("predictions", [0.1]), ("coef", [1.]), ("embeddings", [[1.]])):
            result = fake_result()
            result[section] = value
            with self.assertRaises(ValueError):
                runtime.validate_result(result)
        result = fake_result()
        result["metrics"]["fusion"]["patient_id"] = "fictional-x"
        with self.assertRaises(ValueError):
            runtime.validate_result(result)

    def test_allowlist_rejects_arrays_and_nonfinite_metrics(self):
        for value in (np.array([1.]), float("nan"), float("inf"), "private-value"):
            result = fake_result()
            result["metrics"]["fusion"]["bce"] = value
            with self.assertRaises(ValueError):
                runtime.validate_result(result)

    def test_existing_output_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            runtime.publish(path, {"status": "PASS"})
            before = path.read_bytes()
            with self.assertRaises(Exception):
                runtime.publish(path, {"status": "FAIL"})
            self.assertEqual(path.read_bytes(), before)
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_default_private_output_forbidden_locally(self):
        from ele630.ncct_private_output import _validate_base
        with self.assertRaises(Exception):
            _validate_base(Path.cwd())

    def test_execution_flag_checked_before_any_server_access(self):
        from ele630.fusion_runner import execute
        with patch.object(runtime, "source_identity") as source:
            with self.assertRaises(ValueError):
                execute("ncct", "0" * 40)
        source.assert_not_called()

    def test_real_run_rejects_mismatched_no_image_control_before_publication(self):
        from contextlib import ExitStack
        from ele630 import fusion_runner as runner
        # Fictional pipeline boundaries only: no torch, server files or patients.
        training = types.ModuleType("ele630.ncct_training")
        training._seed_everything = Mock()
        training._membership_record = Mock(return_value={})
        training._parameter_digest = Mock(return_value="unchanged-fictional-weights")
        torch = types.ModuleType("torch")
        torch.set_num_threads = Mock()
        model = Mock()
        model.parameters.return_value = []
        keys = ("bce", "brier_score", "mean_predicted_probability")
        for key in keys:
            control = fake_result()
            control["metrics"]["clinical_only"][key] = .125
            control["metrics"]["no_image"][key] = .125 + .5e-8
            runner._validate_no_image_control(control)
            for value in (.125 + 2e-8, float("nan"), float("inf")):
                control["metrics"]["no_image"][key] = value
                with self.subTest(metric=key, invalid=value), self.assertRaises(ValueError):
                    runner._validate_no_image_control(control)
        with tempfile.TemporaryDirectory() as directory:
            fold = Path(directory)
            destination = types.SimpleNamespace(fold_dir=fold, metrics_path=fold / "aggregate_training.json")
            for mode in ("ncct", "ctp"):
                for key in keys:
                    result = fake_result()
                    result["metrics"]["imaging_only"]["bce"] = {"ncct": .614388, "ctp": .6143472169}[mode]
                    for name in ("clinical_only", "no_image"):
                        result["metrics"][name]["bce"] = .2313358984
                    result["metrics"]["no_image"][key] += 1e-5
                    # Keep schema/deltas valid so only the agreement gate rejects.
                    for name, differences in result["fusion_minus_reference"].items():
                        for metric in differences:
                            differences[metric] = result["metrics"]["fusion"][metric] - result["metrics"][name][metric]
                    runtime.validate_result(result)
                    with self.subTest(mode=mode, metric=key), ExitStack() as stack:
                        stack.enter_context(patch.dict(sys.modules, {"torch": torch, "ele630.ncct_training": training}))
                        stack.enter_context(patch.dict(os.environ))
                        stack.enter_context(patch.object(tempfile, "tempdir", tempfile.tempdir))
                        stack.enter_context(patch.object(runner.os, "umask"))
                        stack.enter_context(patch.object(runtime, "source_identity", return_value={}))
                        stack.enter_context(patch.object(runtime, "environment", return_value={}))
                        stack.enter_context(patch.object(runtime, "bridge", side_effect=[{"dependencies": {}}, result]))
                        stack.enter_context(patch.object(runner, "smoke_record"))
                        stack.enter_context(patch.object(runner, "load_plan", return_value=(Mock(), fictional_plan())))
                        stack.enter_context(patch.object(runner, "check_clinical_path"))
                        stack.enter_context(patch("ele630.ncct_private_output.create_private_run", return_value=destination))
                        for name in ("load_ncct", "load_ctp"):
                            stack.enter_context(patch(f"ele630.fusion_checkpoints.{name}", return_value=(model, {})))
                        stack.enter_context(patch.object(runner, "real_features", return_value=(np.zeros((18, 128)), np.zeros((6, 128)), np.full(6, .3))))
                        publish = stack.enter_context(patch.object(runtime, "publish"))
                        with self.assertRaisesRegex(runner.FusionRunError, "phase=matched_references"):
                            runner.execute(mode, "a" * 40, execute_real=True)
                        publish.assert_called_once()
                        self.assertEqual(publish.call_args.args[0], fold / "failure.json")
                        self.assertEqual(publish.call_args.args[1]["status"], "FAIL")
                        self.assertFalse(destination.metrics_path.exists())

    def test_source_revision_guard_rejects_unknown_revision(self):
        for revision in (None, "HEAD", "0" * 39, "z" * 40):
            with self.assertRaises(ValueError):
                runtime.source_identity(revision)

    def test_source_revision_guard_rejects_checkout_mismatch(self):
        with patch.object(runtime, "PROJECT", Path(__file__).resolve().parents[1]), patch.object(runtime, "git", return_value=b"other"):
            with self.assertRaisesRegex(ValueError, "checkout changed"):
                runtime.source_identity("0" * 40)

    def test_clinical_environment_removes_gpu_packages(self):
        with patch.dict(os.environ, {"PYTHONPATH": "fictional-gpu-path", "OPENBLAS_NUM_THREADS": "99"}):
            env = runtime.clinical_environment()
        self.assertEqual(env["PYTHONPATH"], str(runtime.PROJECT / "src"))
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_bridge_errors_never_expose_subprocess_output(self):
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=1, stderr="fictional-sensitive", stdout="private")):
            with self.assertRaises(ValueError) as error:
                runtime.bridge({"probe": True})
        self.assertNotIn("fictional-sensitive", str(error.exception))

    def test_smoke_gate_rejects_changed_source_or_missing_modality(self):
        from ele630.fusion_runner import smoke_record
        identity = {"source_revision": "a" * 40}
        record = {"status": "PASS", "mode": "smoke", "identity": identity,
                  "dependencies": {}, "patient_data_access": False,
                  "synthetic_tests": {"clinical": 28, "features": 33, "skipped": 0},
                  "results": {"ncct": fake_result(), "ctp": fake_result()}}
        self.assertEqual(runtime.RUNS, {"smoke": "stage5c3_smoke_02", "ncct": "stage5c3_ncct_01", "ctp": "stage5c3_ctp_01"})
        with patch.object(runtime, "read_private_json", return_value=record) as read:
            smoke_record(identity, {})
            read.assert_called_once_with(runtime.RUNTIME / "fusion_runs/stage5c3_smoke_02/outer_fold_0/aggregate_training.json")
            with self.assertRaises(ValueError):
                smoke_record({"source_revision": "b" * 40}, {})
            del record["results"]["ctp"]
            with self.assertRaises(ValueError):
                smoke_record(identity, {})


    def test_private_read_rejects_symlink_before_loading_content(self):
        with patch.object(runtime, "_validate_base"), patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(ValueError):
                runtime.read_private_json(Path("fictional-private.json"))

    def test_gpu_execution_guard_rejects_local_host_before_importing_dependencies(self):
        with patch.object(runtime.platform, "python_version", return_value="3.11.0"):
            with self.assertRaises(ValueError):
                runtime.environment()

    def test_preflight_failure_identifies_phase_without_sensitive_exception(self):
        from ele630.fusion_runner import preflight, FusionRunError
        with patch.object(runtime, "source_identity", return_value={}), patch.object(runtime, "environment", side_effect=ValueError("fictional-private-path")):
            with self.assertRaises(FusionRunError) as error:
                preflight("a" * 40)
        self.assertEqual(str(error.exception), "STAGE5C3_PREFLIGHT_FAIL phase=GPU_imports_on_login")

    def test_aggregate_counts_and_comparison_arithmetic_are_checked(self):
        result = fake_result()
        result["metrics"]["fusion"]["validation_count"] = 3
        with self.assertRaises(ValueError):
            runtime.validate_result(result)
        # Exercise the actual read-only CLI on fictional aggregate artifacts.
        from contextlib import redirect_stdout, redirect_stderr
        spec = importlib.util.spec_from_file_location("fictional_fusion_cli", Path(__file__).resolve().parents[1] / "scripts/experiments/run_stage5c3_fusion.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        records = [{"status": "PASS", "mode": m, "identity": {}, "outer_test_evaluated": False,
                    "provenance": {"membership": {}}, "results": fake_result()} for m in ("ncct", "ctp")]
        for expected in (0, 1):
            if expected:
                records[0]["results"]["metrics"]["fusion"]["patient_id"] = "fictional-private-identifier"
            output, errors = io.StringIO(), io.StringIO()
            with patch.object(runtime, "source_identity", return_value={}), patch.object(runtime, "read_private_json", side_effect=records), redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(cli.main(["results", "--source-revision", "a" * 40]), expected)
            self.assertNotIn("fictional-private-identifier", output.getvalue() + errors.getvalue())
            if not expected:
                self.assertEqual(set(json.loads(output.getvalue())), {"ncct", "ctp"})
        result = fake_result()
        result["fusion_minus_reference"]["clinical_only"]["bce"] = .01
        with self.assertRaises(ValueError):
            runtime.validate_result(result)

    def test_canonical_plan_uses_fixed_paths_and_rejects_replacement_membership(self):
        from ele630.fusion_runner import load_plan
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            tables = project / "outputs/tables"
            tables.mkdir(parents=True)
            for name in ("patient_manifest.csv", "outcome_cv_folds_private.csv"):
                (tables / name).write_text("fictional metadata")
            contract = Mock(fingerprint="replacement")
            with patch.object(runtime, "PROJECT", project), patch("ele630.data_contract.load_and_validate", return_value=contract) as load, patch("ele630.ncct_splits.plan_outer_fold", return_value=fictional_plan()) as split:
                with self.assertRaises(ValueError):
                    load_plan()
                load.assert_called_once_with(tables / "patient_manifest.csv", tables / "outcome_cv_folds_private.csv")
                split.assert_called_once_with(contract, 0, seed=630, validation_fraction=.20)

    def test_checkpoint_discovery_rejects_missing_ambiguous_or_wrong_stride_before_weights(self):
        from ele630 import fusion_checkpoints as checkpoints
        resnet = types.ModuleType("ele630.resnet3d")
        resnet.CompactResNet10 = Mock()
        checkpoint = types.ModuleType("ele630.ncct_checkpoint")
        checkpoint.load_ncct_checkpoint = Mock()
        paths = [Path("fictional-base") / f"run{i}/outer_fold_0/aggregate_training.json" for i in range(2)]
        summary = {"run_record": {"source_revision": runtime.NCCT_REVISION,
                                  "environment": {"slurm_job_id": "486314"},
                                  "architecture": {"stage4_stride": [2, 2, 2]}}}
        with patch.dict(sys.modules, {"ele630.resnet3d": resnet, "ele630.ncct_checkpoint": checkpoint}), patch("ele630.ncct_private_output._validate_base"), patch.object(checkpoints, "read_private_json", return_value=summary):
            for discovered in ([], paths):
                with patch.object(Path, "glob", return_value=discovered):
                    with self.assertRaises(ValueError):
                        checkpoints.load_ncct(fictional_plan())
            summary["run_record"]["architecture"]["stage4_stride"] = [1, 1, 1]
            with patch.object(Path, "glob", return_value=paths[:1]):
                with self.assertRaises(ValueError):
                    checkpoints.load_ncct(fictional_plan())
        checkpoint.load_ncct_checkpoint.assert_not_called()


@unittest.skipUnless(HAS_SKLEARN, "existing local environment has no scikit-learn/SciPy")
class FusionEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.plan = fictional_plan()
        rng = np.random.default_rng(630)
        self.train, self.val = rng.normal(size=(18, 128)), rng.normal(size=(6, 128))

    def test_pca_numeric_oracle_and_train_only_statistics(self):
        transformer = fusion.ImagePCA().fit(self.train)
        z = (self.train - self.train.mean(0)) / self.train.std(0)
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        np.testing.assert_allclose(transformer.scaler.mean_, self.train.mean(0))
        np.testing.assert_allclose(transformer.pca.singular_values_, singular[:8])
        np.testing.assert_allclose(transformer.pca.components_.T @ transformer.pca.components_, vt[:8].T @ vt[:8], atol=1e-12)
        before = transformer.scaler.mean_.copy()
        transformer.transform(self.val * 999)
        np.testing.assert_array_equal(before, transformer.scaler.mean_)
        self.assertFalse(transformer.pca.whiten)
        self.assertEqual(transformer.pca.svd_solver, "full")

    def test_pca_rejects_degenerate_or_small_training(self):
        for train in (np.ones((18, 128)), np.zeros((8, 128))):
            with self.assertRaises(fusion.FusionError):
                fusion.ImagePCA().fit(train)
        transformer = fusion.ImagePCA().fit(self.train)
        with self.assertRaises(fusion.FusionError):
            transformer.fit(self.train)

    def test_deterministic_integration_controls_and_frozen_recipe(self):
        rows = fictional_rows(self.plan)
        a = fusion.run_fusion(self.plan, rows, self.train, self.val, [.3] * 6)
        b = fusion.run_fusion(self.plan, rows, self.train, self.val, [.3] * 6)
        self.assertEqual(a, b)
        runtime.validate_result(a)
        for key in ("bce", "auroc", "brier_score"):
            self.assertAlmostEqual(a["metrics"]["no_image"][key], a["metrics"]["clinical_only"][key], places=8)
        from ele630.clinical_baseline import run_clinical_development, _make_logistic_regression
        self.assertAlmostEqual(a["metrics"]["clinical_only"]["bce"], run_clinical_development(self.plan, rows).logistic_regression.bce, places=12)
        config = _make_logistic_regression().get_params()
        self.assertEqual((config["C"], config["solver"], config["tol"], config["max_iter"]), (1., "lbfgs", 1e-8, 1000))


@unittest.skipUnless(HAS_TORCH, "existing local environment has no PyTorch")
class FusionFeatureTests(unittest.TestCase):
    def test_ncct_frozen_embedding_and_logit_match(self):
        import torch
        from ele630.resnet3d import CompactResNet10
        from ele630.fusion_features import extract, freeze
        torch.set_num_threads(1)
        model = CompactResNet10(width=16)
        image = np.random.default_rng(630).random((1, 4, 16, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            extract(model, image, "ncct")
        before = {k: v.clone() for k, v in model.state_dict().items()}
        freeze(model)
        embedding, score = extract(model, image, "ncct")
        with torch.no_grad():
            full = model(torch.from_numpy(image).unsqueeze(0)).sigmoid().item()
        self.assertEqual(score, full)
        self.assertEqual(embedding.shape, (128,))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_ctp_streamed_mean_matches_explicit_all_frame_oracle(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import extract, freeze
        from ele630.fusion_synthetic import FictionalCTP
        torch.set_num_threads(1)
        model = freeze(CompactCTPMean(width=16))
        series = FictionalCTP(630)
        embedding, score = extract(model, series, "ctp")
        self.assertEqual(series.reads, list(range(40)))
        with torch.inference_mode():
            features = torch.stack([model.encoder.forward_features(model._frame_tensor(FictionalCTP(630), i)) for i in range(40)])
            oracle = features.mean(0)
            np.testing.assert_allclose(embedding, oracle.numpy().reshape(128), rtol=1e-6, atol=1e-6)
            self.assertAlmostEqual(score, model.head(oracle).sigmoid().item(), places=6)
        series.metadata.frame_count = 39
        with self.assertRaises(ValueError):
            extract(model, series, "ctp")

    def test_outer_and_forged_patient_blocked_before_loader(self):
        from ele630.fusion_features import DevelopmentFeatures
        from ele630.resnet3d import CompactResNet10
        plan = fictional_plan()
        loader = Mock()
        features = DevelopmentFeatures(plan, loader, CompactResNet10(), "ncct")
        for patient in (plan.outer_test[0], replace(plan.inner_train[0], outcome="good")):
            with self.assertRaises(ValueError):
                features.load(patient)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
