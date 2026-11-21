"""Read-only fold-0 evaluation against fictional private runs and series."""

from __future__ import annotations

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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630 import ncct_private_output
from ele630.ncct_splits import plan_outer_fold
from test_ncct_stage4a import fictional_contract
from test_ctp_training import FakeRepository


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ele630 import ctp_continuation, ctp_evaluation, ctp_model, ctp_training
    from ele630.ctp_model import (
        CTPBackwardResult, CompactCTPMean, CompactCTPMeanStd, CompactCTPShallowTCN,
    )
    from ele630.ctp_training import CTPTrainingConfig, run_ctp_development
    from ele630.ctp_evaluation import CTPEvaluationError, evaluate_best_ctp_inner_validation


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CTPAggregateEvaluatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "approved"
        self.base = self.root / "private_ctp"
        self.base.mkdir(parents=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o700)
        for name in ("_APPROVED_PRIVATE_ROOT", "_TRUSTED_CANONICAL_ROOT"):
            substitution = patch.object(ncct_private_output, name, self.root)
            substitution.start()
            self.addCleanup(substitution.stop)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.contract = fictional_contract(per_fold=2)
        self.plan = plan_outer_fold(self.contract, 0)
        self.profile_patch = patch.object(
            ctp_evaluation, "ACTIVE_PROFILE",
            SimpleNamespace(patients=len(self.contract.patients),
                            fingerprint=self.contract.fingerprint),
        )
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)
        self.models = []

        class TinyCTP(torch.nn.Module):
            configuration = {"synthetic_frame_model": True, "frame_count": 40}

            def __init__(nested_self, width):
                super().__init__()
                if width != 16:
                    raise AssertionError("unexpected CTP width")
                nested_self.bias = torch.nn.Parameter(torch.tensor([0.1]))
                nested_self.backward_calls = 0
                nested_self.forward_calls = 0
                self.models.append(nested_self)

            def backward_series(nested_self, series, target):
                nested_self.backward_calls += 1
                nested_self.zero_grad(set_to_none=True)
                for _ in range(2):
                    for frame in range(40):
                        series.load_frame(frame)
                logit = nested_self.bias
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
                loss.backward()
                return CTPBackwardResult(loss.detach(), logit.detach())

            def forward(nested_self, series):
                if nested_self.training or torch.is_grad_enabled():
                    raise AssertionError("evaluation is not in eval/no_grad mode")
                nested_self.forward_calls += 1
                for frame in range(40):
                    series.load_frame(frame)
                return nested_self.bias * 0 + torch.tensor([series.score], dtype=torch.float32)

        self.model_patch_training = patch.object(ctp_training, "CompactCTPMean", TinyCTP)
        self.model_patch_evaluation = patch.object(ctp_evaluation, "CompactCTPMean", TinyCTP)
        self.model_patch_training.start()
        self.model_patch_evaluation.start()
        self.addCleanup(self.model_patch_training.stop)
        self.addCleanup(self.model_patch_evaluation.stop)

        class Series:
            def __init__(nested_self, score):
                nested_self.score = score
                nested_self.calls = []

            def load_frame(nested_self, frame):
                nested_self.calls.append(frame)

        class Repository:
            def __init__(nested_self):
                nested_self.opened = []
                nested_self.on_open = None

            def open_series(nested_self, patient):
                if nested_self.on_open is not None:
                    nested_self.on_open(patient)
                score = -0.5 if patient.outcome == "good" else 0.5
                series = Series(score)
                nested_self.opened.append((patient, series))
                return series

        self.repo = Repository()

    def make_completed_run(self, *, best_epoch_two=False, early_stop=False):
        max_epochs = 10 if early_stop else 2

        def training_call():
            run_ctp_development(
                self.contract, self.repo, mode="development", private_output_dir=self.base,
                run_id="fictional_complete", config=CTPTrainingConfig(max_epochs=max_epochs),
                epochs_this_job=max_epochs,
            )

        if best_epoch_two or early_stop:
            original_validation = ctp_training._validation_bce
            scheduled = iter((0.9, 0.5, 0.6, 0.7, 0.8) if early_stop else (1.0, 0.5))

            def fixed_aggregate(model, data, device):
                original_validation(model, data, device)
                return next(scheduled)

            with patch.object(ctp_training, "_validation_bce", side_effect=fixed_aggregate):
                training_call()
        else:
            training_call()
        self.repo.opened.clear()
        return self.base / "fictional_complete" / "outer_fold_0"

    def evaluate(self, **kwargs):
        return evaluate_best_ctp_inner_validation(
            self.contract, self.repo, private_output_dir=self.base,
            run_id="fictional_complete", **kwargs,
        )

    def test_only_inner_validation_all_frames_aggregate_output_and_unchanged_model(self):
        folder = self.make_completed_run()
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step")), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.evaluate()
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertEqual([patient for patient, _ in self.repo.opened],
                         list(self.plan.inner_validation))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))
        self.assertTrue(all(series.calls == list(range(40)) for _, series in self.repo.opened))
        self.assertEqual((result.validation_count, result.validation_good_count,
                          result.validation_poor_count), (2, 1, 1))
        self.assertAlmostEqual(result.auroc, 1.0)
        self.assertAlmostEqual(result.average_precision, 1.0)
        self.assertAlmostEqual(result.balanced_accuracy_0_5, 1.0)
        self.assertAlmostEqual(result.mean_predicted_probability, 0.5, places=6)
        logits = [torch.tensor([series.score]) for _, series in self.repo.opened]
        targets = [torch.tensor([0. if patient.outcome == "good" else 1.])
                   for patient, _ in self.repo.opened]
        reference_bce = sum(float(torch.nn.functional.binary_cross_entropy_with_logits(logit, target))
                            for logit, target in zip(logits, targets)) / 2
        reference_brier = sum((float(torch.sigmoid(logit)) - float(target)) ** 2
                              for logit, target in zip(logits, targets)) / 2
        self.assertAlmostEqual(result.bce, reference_bce, places=7)
        self.assertAlmostEqual(result.brier_score, reference_brier, places=7)
        evaluated_model = self.models[-1]
        self.assertEqual((evaluated_model.backward_calls, evaluated_model.forward_calls), (0, 2))
        selected = torch.load(folder / "checkpoint_epoch_001.pt", map_location="cpu",
                              weights_only=True)
        for name, value in evaluated_model.state_dict().items():
            torch.testing.assert_close(value, selected[name], rtol=0, atol=0)
        record = json.loads((folder / "aggregate_inner_validation_evaluation.json").read_text())
        self.assertEqual(record, result.as_dict())
        self.assertEqual(len(record), 9)
        contents = json.dumps(record)
        for patient in self.contract.patients:
            self.assertNotIn(patient.patient_id, contents)
        self.assertNotIn(str(self.base), contents)
        self.assertNotIn("checkpoint", contents)
        self.assertNotIn("logit", contents)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE((folder / "aggregate_inner_validation_evaluation.json").stat().st_mode), 0o600)
        with self.assertRaises(CTPEvaluationError):
            self.evaluate()  # Aggregate-only artifact is exclusive.

    def test_bad_fold_and_effective_outer_membership_fail_before_image_access(self):
        self.make_completed_run()
        with self.assertRaises(CTPEvaluationError):
            self.evaluate(outer_fold=1)
        self.assertEqual(self.repo.opened, [])
        original_data = ctp_evaluation.CTPDevelopmentData

        def contaminated_data(contract, repository, *, mode):
            data = original_data(contract, repository, mode=mode)
            data._validation = (data.plan.outer_test[0],)
            return data

        with patch.object(ctp_evaluation, "CTPDevelopmentData", side_effect=contaminated_data):
            with self.assertRaises(CTPEvaluationError):
                self.evaluate()
        self.assertEqual(self.repo.opened, [])

    def test_selected_epoch_two_is_loaded_from_best_checkpoint(self):
        folder = self.make_completed_run(best_epoch_two=True)
        self.assertEqual(json.loads((folder / "aggregate_training.json").read_text())["best_epoch"], 2)
        self.evaluate()
        selected = torch.load(folder / "checkpoint_epoch_002.pt", map_location="cpu",
                              weights_only=True)
        for name, value in self.models[-1].state_dict().items():
            torch.testing.assert_close(value, selected[name], rtol=0, atol=0)
        self.assertEqual([patient for patient, _ in self.repo.opened],
                         list(self.plan.inner_validation))

    def test_early_stop_epoch_five_evaluates_only_validation_with_epoch_two_weights(self):
        folder = self.make_completed_run(early_stop=True)
        epoch, marker, state, commits = ctp_continuation.read_committed_state(folder)
        summary = json.loads((folder / "aggregate_training.json").read_text())
        self.assertEqual((epoch, len(commits), state["completed_epoch"], state["next_epoch"]),
                         (5, 5, 5, 6))
        self.assertEqual((state["best_epoch"], state["best_validation_bce"],
                          state["stale_epochs"], state["optimizer_steps"]),
                         (2, 0.5, 3, 5 * len(self.plan.inner_train)))
        self.assertEqual([record["validation_bce"] for record in state["history"]],
                         [0.9, 0.5, 0.6, 0.7, 0.8])
        self.assertEqual((summary["status"], summary["completed_epochs"],
                          summary["next_epoch"], summary["best_epoch"]),
                         ("complete", 5, None, 2))
        self.assertEqual(marker["best_checkpoint"]["name"], "checkpoint_epoch_002.pt")
        self.assertFalse((folder / "checkpoint_epoch_005.pt").exists())

        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step")):
            self.evaluate()
        selected = torch.load(folder / "checkpoint_epoch_002.pt", map_location="cpu",
                              weights_only=True)
        for name, value in self.models[-1].state_dict().items():
            torch.testing.assert_close(value, selected[name], rtol=0, atol=0)
        self.assertEqual([patient for patient, _ in self.repo.opened],
                         list(self.plan.inner_validation))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))
        self.assertTrue(all(series.calls == list(range(40)) for _, series in self.repo.opened))

    def test_stage4r_writer_state_uses_shared_weights_only_reader(self):
        folder = self.make_completed_run()
        epoch, _, state, _ = ctp_continuation.read_committed_state(folder)
        self.assertEqual(epoch, 2)
        self.assertIsInstance(state["model_state"], dict)
        self.assertIsInstance(state["optimizer_state"], dict)
        self.assertIsInstance(state["rng"]["python"], tuple)
        self.assertIsInstance(state["rng"]["numpy"], tuple)
        self.assertIsInstance(state["rng"]["numpy"][1], list)
        self.assertEqual(len(state["rng"]["numpy"][1]), 624)
        self.assertEqual(state["rng"]["torch_cpu"].dtype, torch.uint8)
        self.assertIsNone(state["rng"]["torch_cuda"])
        self.evaluate()
        self.assertEqual([patient for patient, _ in self.repo.opened],
                         list(self.plan.inner_validation))

    def test_completed_summary_with_resume_epoch_is_rejected_before_images(self):
        folder = self.make_completed_run()
        summary_path = folder / "aggregate_training.json"
        summary = json.loads(summary_path.read_text())
        summary["next_epoch"] = 3
        summary_path.write_text(json.dumps(summary))
        with self.assertRaises(CTPEvaluationError):
            self.evaluate()
        self.assertEqual(self.repo.opened, [])

    def test_committed_state_with_null_next_epoch_is_rejected_before_images(self):
        folder = self.make_completed_run()
        _, marker, state, _ = ctp_continuation.read_committed_state(folder)
        state["next_epoch"] = None
        state_path = folder / marker["state_file"]
        torch.save(state, state_path)
        marker["state_sha256"] = ctp_continuation._digest(state_path)
        marker_path = next(folder.glob("commit_epoch_002_*.json"))
        marker_path.write_text(json.dumps(marker))
        with self.assertRaises(CTPEvaluationError):
            self.evaluate()
        self.assertEqual(self.repo.opened, [])

    def test_changed_inner_split_digest_is_rejected_before_images(self):
        self.make_completed_run()
        with patch.object(ctp_evaluation, "_split_digest", return_value="fictional-wrong-split"):
            with self.assertRaises(CTPEvaluationError):
                self.evaluate()
        self.assertEqual(self.repo.opened, [])

    def test_active_contract_and_selected_config_mismatch_fail_before_images(self):
        folder = self.make_completed_run()
        invalid_contract = replace(self.contract, fingerprint="fictional-invalid")
        with self.assertRaises(CTPEvaluationError):
            evaluate_best_ctp_inner_validation(invalid_contract, self.repo,
                                               private_output_dir=self.base,
                                               run_id="fictional_complete")
        self.assertEqual(self.repo.opened, [])
        config_path = folder / "run_config.json"
        saved = json.loads(config_path.read_text())
        saved["model"]["synthetic_frame_model"] = False
        config_path.write_text(json.dumps(saved))
        with self.assertRaises(CTPEvaluationError) as caught:
            self.evaluate()
        self.assertNotIn(str(self.base), str(caught.exception))
        self.assertEqual(self.repo.opened, [])

    def test_checkpoint_checksum_and_incomplete_run_fail_before_images(self):
        folder = self.make_completed_run()
        checkpoint = folder / "checkpoint_epoch_001.pt"
        checkpoint.write_bytes(b"fictional-broken-checkpoint")
        with self.assertRaises(CTPEvaluationError):
            self.evaluate()
        self.assertEqual(self.repo.opened, [])
        checkpoint.unlink()
        (folder / "aggregate_training.json").unlink()
        with self.assertRaises(CTPEvaluationError):
            self.evaluate()
        self.assertEqual(self.repo.opened, [])

    def test_repository_exception_does_not_disclose_fictional_patient_or_path(self):
        self.make_completed_run()

        def fail(patient):
            raise OSError(f"{patient.patient_id} /fictional/private/registered_ctp.nii.gz")

        self.repo.on_open = fail
        with self.assertRaises(CTPEvaluationError) as caught:
            self.evaluate()
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn("fictional/private", str(caught.exception))
        self.assertFalse((self.base / "fictional_complete" / "outer_fold_0" /
                          "aggregate_inner_validation_evaluation.json").exists())

    def make_architecture_run(self, model_class):
        """Use real temporal models and the real writer with a tiny spatial encoder."""
        self.model_patch_training.stop()
        self.model_patch_evaluation.stop()
        self.repo = FakeRepository()

        class TinyEncoder(torch.nn.Module):
            def __init__(self, width, pooling):
                super().__init__()
                if (width, pooling) != (16, "avg"):
                    raise AssertionError("unexpected encoder configuration")
                self.width = width
                self.projection = torch.nn.Linear(1, 8 * width)

            def forward_features(self, frame):
                return self.projection(frame.mean(dim=(1, 2, 3, 4)).unsqueeze(1))

        encoder_patch = patch.object(ctp_model, "CompactResNet10", TinyEncoder)
        encoder_patch.start()
        self.addCleanup(encoder_patch.stop)
        original_validation = ctp_training._validation_bce
        scheduled = iter((0.5, 0.6))  # Best epoch 1 differs from latest epoch 2.

        def fixed_aggregate(model, data, device):
            original_validation(model, data, device)
            return next(scheduled)

        with patch.object(ctp_training, "_validation_bce", side_effect=fixed_aggregate):
            run_ctp_development(
                self.contract, self.repo, mode="development", private_output_dir=self.base,
                run_id="fictional_complete", config=CTPTrainingConfig(max_epochs=2),
                epochs_this_job=2, model_factory=model_class,
            )
        self.repo.opened.clear()
        return self.base / "fictional_complete" / "outer_fold_0"

    def test_meanstd_opt_in_best_weights_validation_only_and_nine_private_aggregates(self):
        folder = self.make_architecture_run(CompactCTPMeanStd)
        epoch, _, latest, _ = ctp_continuation.read_committed_state(folder)
        self.assertEqual((epoch, latest["best_epoch"]), (2, 1))
        selected = torch.load(folder / "checkpoint_epoch_001.pt", map_location="cpu", weights_only=True)
        self.assertTrue(any(not torch.equal(value, latest["model_state"][name])
                            for name, value in selected.items()))
        original_files = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
        evaluated = []
        strict_loads = []
        original_forward = CompactCTPMeanStd.forward
        original_load = CompactCTPMeanStd.load_state_dict

        def checked_forward(model, series):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            evaluated.append(model)
            return original_forward(model, series)

        def checked_load(model, weights, strict=True, **kwargs):
            strict_loads.append(strict)
            return original_load(model, weights, strict=strict, **kwargs)

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(CompactCTPMeanStd, "forward", checked_forward), \
             patch.object(CompactCTPMeanStd, "load_state_dict", checked_load), \
             patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step")), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.evaluate(model_factory=CompactCTPMeanStd)
        self.assertEqual(strict_loads, [True])
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertEqual([patient for patient, _ in self.repo.opened], list(self.plan.inner_validation))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))
        self.assertTrue(all(series.calls == list(range(40)) for _, series in self.repo.opened))
        for name, value in evaluated[0].state_dict().items():
            torch.testing.assert_close(value, selected[name], rtol=0, atol=0)
        for path, original in original_files.items():
            self.assertEqual(path.read_bytes(), original)
        output = folder / "aggregate_inner_validation_evaluation.json"
        record = json.loads(output.read_text())
        self.assertEqual(record, result.as_dict())
        self.assertEqual(set(record), {
            "bce", "auroc", "average_precision", "brier_score", "balanced_accuracy_0_5",
            "mean_predicted_probability", "validation_count", "validation_good_count",
            "validation_poor_count",
        })
        self.assertEqual((record["validation_count"], record["validation_good_count"],
                          record["validation_poor_count"]), (2, 1, 1))
        contents = json.dumps(record)
        for patient in self.contract.patients:
            self.assertNotIn(patient.patient_id, contents)
        self.assertNotIn(str(self.base), contents)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.repo.opened.clear()
        with self.assertRaises(CTPEvaluationError):
            self.evaluate(model_factory=CompactCTPMeanStd)
        self.assertEqual(self.repo.opened, [])

    def test_tcn_opt_in_best_checkpoint_ordered_validation_aggregate_only_and_mismatch_guards(self):
        # Real TCN, runner/writer and evaluator; reuse the existing tiny spatial
        # encoder fixture so this CPU gate stays small and entirely fictional.
        folder = self.make_architecture_run(CompactCTPShallowTCN)
        epoch, _, latest, _ = ctp_continuation.read_committed_state(folder)
        self.assertEqual((epoch, latest["best_epoch"]), (2, 1))
        selected = torch.load(folder / "checkpoint_epoch_001.pt", map_location="cpu", weights_only=True)
        self.assertTrue(any(not torch.equal(value, latest["model_state"][name])
                            for name, value in selected.items()))
        config_path = folder / "run_config.json"
        original_config = config_path.read_bytes()
        evaluated = []
        original_forward = CompactCTPShallowTCN.forward

        def checked_forward(model, series):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(model.configuration, json.loads(original_config)["model"])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, selected[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            evaluated.append(model)
            return original_forward(model, series)

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            for factory in (CompactCTPMean, CompactCTPMeanStd):
                with self.subTest(wrong_architecture=factory.__name__):
                    with self.assertRaises(CTPEvaluationError) as caught:
                        self.evaluate(model_factory=factory)
                    self.assertEqual(str(caught.exception), "CTP inner-validation evaluation stopped")
                    self.assertIsNone(caught.exception.__cause__)
                    self.assertEqual(self.repo.opened, [])
            changed = json.loads(original_config)
            changed["model"]["frame_order"] = "fictional_reversed_order"
            config_path.write_text(json.dumps(changed))
            try:
                with self.assertRaises(CTPEvaluationError):
                    self.evaluate(model_factory=CompactCTPShallowTCN)
                self.assertEqual(self.repo.opened, [])
            finally:
                config_path.write_bytes(original_config)
            self.assertFalse((folder / "aggregate_inner_validation_evaluation.json").exists())
            originals = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
            with patch.object(CompactCTPShallowTCN, "forward", checked_forward), \
                 patch.object(CompactCTPShallowTCN, "backward_series", side_effect=AssertionError("backward")), \
                 patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step")):
                result = self.evaluate(model_factory=CompactCTPShallowTCN)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertEqual(len(evaluated), len(self.plan.inner_validation))
        self.assertEqual([patient for patient, _ in self.repo.opened], list(self.plan.inner_validation))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))
        self.assertTrue(all(series.calls == list(range(40)) for _, series in self.repo.opened))
        for name, value in evaluated[0].state_dict().items():
            torch.testing.assert_close(value, selected[name], rtol=0, atol=0)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
        output = folder / "aggregate_inner_validation_evaluation.json"
        self.assertEqual({path for path in folder.iterdir() if path.is_file()}, set(originals) | {output})
        record = json.loads(output.read_text())
        self.assertEqual(record, result.as_dict())
        self.assertEqual(set(record), {
            "bce", "auroc", "average_precision", "brier_score", "balanced_accuracy_0_5",
            "mean_predicted_probability", "validation_count", "validation_good_count",
            "validation_poor_count",
        })
        self.assertEqual((record["validation_count"], record["validation_good_count"],
                          record["validation_poor_count"]), (2, 1, 1))
        self.assertTrue(all(type(value) in (int, float) for value in record.values()))
        contents = json.dumps(record)
        for patient in self.contract.patients:
            self.assertNotIn(patient.patient_id, contents)
        self.assertNotIn(str(self.base), contents)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_mean_run_requested_as_meanstd_is_rejected_before_images(self):
        self.make_architecture_run(CompactCTPMean)
        with self.assertRaises(CTPEvaluationError):
            self.evaluate(model_factory=CompactCTPMeanStd)
        self.assertEqual(self.repo.opened, [])

    def test_meanstd_run_requested_as_default_mean_is_rejected_before_images(self):
        self.make_architecture_run(CompactCTPMeanStd)
        with self.assertRaises(CTPEvaluationError):
            self.evaluate()
        self.assertEqual(self.repo.opened, [])

    def test_meanstd_changed_model_configuration_is_rejected_before_images(self):
        folder = self.make_architecture_run(CompactCTPMeanStd)
        config_path = folder / "run_config.json"
        saved = json.loads(config_path.read_text())
        saved["model"]["std_correction"] = 1
        config_path.write_text(json.dumps(saved))
        with self.assertRaises(CTPEvaluationError):
            self.evaluate(model_factory=CompactCTPMeanStd)
        self.assertEqual(self.repo.opened, [])

    def test_meanstd_completion_checksum_split_and_contract_guards_before_images(self):
        folder = self.make_architecture_run(CompactCTPMeanStd)

        def rejected(**kwargs):
            with self.assertRaises(CTPEvaluationError):
                self.evaluate(model_factory=CompactCTPMeanStd, **kwargs)
            self.assertEqual(self.repo.opened, [])
            self.assertFalse((folder / "aggregate_inner_validation_evaluation.json").exists())

        rejected(outer_fold=1)
        with patch.object(ctp_evaluation, "_split_digest", return_value="fictional-wrong-split"):
            rejected()
        with patch.object(ctp_evaluation, "ACTIVE_PROFILE", SimpleNamespace(
                patients=len(self.contract.patients), fingerprint="fictional-wrong-contract")):
            rejected()
        summary_path = folder / "aggregate_training.json"
        summary_bytes = summary_path.read_bytes()
        summary = json.loads(summary_bytes)
        summary["status"] = "paused"
        summary_path.write_text(json.dumps(summary))
        rejected()
        summary_path.write_bytes(summary_bytes)
        _, marker, _, _ = ctp_continuation.read_committed_state(folder)
        for path in (folder / "checkpoint_epoch_001.pt", folder / marker["state_file"]):
            with self.subTest(artifact=path.name):
                original = path.read_bytes()
                path.write_bytes(b"fictional-corrupt-artifact")
                rejected()
                path.write_bytes(original)

    def test_meanstd_repository_and_frame_errors_remain_sanitized(self):
        folder = self.make_architecture_run(CompactCTPMeanStd)
        message = f"{self.plan.inner_validation[0].patient_id} /fictional/private/image"
        with patch.object(self.repo, "open_series", side_effect=OSError(message)):
            with self.assertRaises(CTPEvaluationError) as caught:
                self.evaluate(model_factory=CompactCTPMeanStd)
        self.assertEqual(str(caught.exception), "CTP inner-validation evaluation stopped")
        self.assertIsNone(caught.exception.__cause__)

        def fail_frame(series, index):
            raise OSError(message)

        self.repo.on_frame = fail_frame
        with self.assertRaises(CTPEvaluationError) as caught:
            self.evaluate(model_factory=CompactCTPMeanStd)
        self.assertEqual(str(caught.exception), "CTP inner-validation evaluation stopped")
        self.assertIsNone(caught.exception.__cause__)
        self.assertFalse((folder / "aggregate_inner_validation_evaluation.json").exists())

    def test_invalid_model_factory_is_rejected_before_image_access(self):
        with self.assertRaises(CTPEvaluationError):
            self.evaluate(model_factory="meanstd")
        self.assertEqual(self.repo.opened, [])


if __name__ == "__main__":
    unittest.main()
