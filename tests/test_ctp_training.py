"""Fictional contracts, tiny CPU arrays and temporary private-output fixtures."""

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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.ctp_development import CTPDevelopmentData
from ele630.ctp_input import prepare_ctp_frame_input
from ele630.data_contract import OutcomePatient
from ele630 import ncct_private_output
from ele630.ncct_private_output import PrivateOutputError, create_private_run, open_private_binary
from ele630.ncct_splits import SplitError, binary_outcome, plan_outer_fold
from test_ncct_stage4a import fictional_contract


class FakeSeries:
    def __init__(self, repository):
        self.repository = repository
        self.metadata = SimpleNamespace(frame_count=40, spatial_shape=(5, 7, 5), affine=np.diag([-1., -1., 3., 1.]))
        self.calls = []

    def load_frame(self, index):
        self.calls.append(index)
        if self.repository.on_frame is not None:
            self.repository.on_frame(self, index)
        native = (np.arange(175, dtype=np.float32).reshape(5, 7, 5) + 3 * index) % 255
        return SimpleNamespace(input=prepare_ctp_frame_input(native, self.metadata.affine, index))


class FakeRepository:
    def __init__(self):
        self.opened = []
        self.on_frame = None

    def open_series(self, patient):
        series = FakeSeries(self)
        self.opened.append((patient, series))
        return series


class CTPDevelopmentAccessTests(unittest.TestCase):
    def setUp(self):
        self.contract = fictional_contract(per_fold=2)
        self.repo = FakeRepository()

    def test_existing_fold_zero_plan_and_contract_labels_are_used(self):
        data = CTPDevelopmentData(self.contract, self.repo, mode="development")
        self.assertEqual(data.plan, plan_outer_fold(self.contract, 0, seed=630, validation_fraction=0.20))
        for patients, opener in ((data.training_patients, data.open_training),
                                 (data.validation_patients, data.open_validation)):
            for patient in patients:
                sample = opener(patient)
                self.assertEqual(sample.target, 0 if patient.outcome == "good" else 1)
                self.assertEqual(sample.series.calls, [])  # No eager frame reads.
        self.assertTrue(all(p.fold != 0 for p, _ in self.repo.opened))

    def test_membership_and_partition_enforced_before_image_access(self):
        data = CTPDevelopmentData(self.contract, self.repo, mode="development")
        outsider = OutcomePatient("fictional-outsider", "A", "good", False, 1)
        changed = replace(data.training_patients[0], outcome="poor" if data.training_patients[0].outcome == "good" else "good")
        for patient in (*data.plan.outer_test, outsider, changed, "fictional"):
            for opener in (data.open_training, data.open_validation):
                with self.assertRaisesRegex(SplitError, "authorized inner partition"):
                    opener(patient)
        with self.assertRaises(SplitError):
            data.open_training(data.validation_patients[0])
        with self.assertRaises(SplitError):
            data.open_validation(data.training_patients[0])
        self.assertEqual(self.repo.opened, [])

    def test_smoke_allows_only_one_predeclared_inner_training_patient(self):
        data = CTPDevelopmentData(self.contract, self.repo, mode="smoke")
        selected = data.plan.inner_train[0]
        self.assertEqual(data.training_patients, (selected,))
        self.assertEqual(data.validation_patients, ())
        data.open_training(selected)
        for patient in self.contract.patients:
            if patient != selected:
                with self.assertRaises(SplitError):
                    data.open_training(patient)
            with self.assertRaises(SplitError):
                data.open_validation(patient)
        self.assertEqual([p for p, _ in self.repo.opened], [selected])

    def test_bad_contract_and_mode_rejected_before_image_access(self):
        for mode in (None, "all_folds", True):
            with self.assertRaisesRegex(ValueError, "mode"):
                CTPDevelopmentData(self.contract, self.repo, mode=mode)
        with self.assertRaisesRegex(SplitError, "fingerprint"):
            CTPDevelopmentData(replace(self.contract, fingerprint="fictional-invalid"), self.repo, mode="development")
        self.assertEqual(self.repo.opened, [])

    def test_open_failure_omits_patient_and_path(self):
        data = CTPDevelopmentData(self.contract, self.repo, mode="smoke")
        patient = data.training_patients[0]
        with patch.object(self.repo, "open_series", side_effect=OSError(f"{patient.patient_id} /private/image")):
            with self.assertRaisesRegex(RuntimeError, "^CTP series could not be opened$") as caught:
                data.open_training(patient)
        self.assertIsNone(caught.exception.__cause__)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ele630 import ctp_training
    from ele630.ctp_model import CTPBackwardResult, CompactCTPMean
    from ele630.ctp_training import CTPTrainingConfig, CTPTrainingError, run_ctp_development


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CTPTrainingTests(unittest.TestCase):
    def setUp(self):
        self.contract = fictional_contract(per_fold=2)
        self.plan = plan_outer_fold(self.contract, 0)
        self.repo = FakeRepository()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "approved"
        self.base = self.root / "private_ctp"
        self.base.mkdir(parents=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o700)
        # Private module constants are substituted only for synthetic tests.
        for name in ("_APPROVED_PRIVATE_ROOT", "_TRUSTED_CANONICAL_ROOT"):
            substitution = patch.object(ncct_private_output, name, self.root)
            substitution.start()
            self.addCleanup(substitution.stop)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def run_fixture(self, run_id, mode="development", max_epochs=1, job_epochs=1, resume=False):
        return run_ctp_development(self.contract, self.repo, mode=mode, private_output_dir=self.base,
                                   run_id=run_id, config=CTPTrainingConfig(max_epochs=max_epochs),
                                   epochs_this_job=job_epochs, resume=resume)

    def test_fixed_configuration_and_epoch_bound(self):
        config = CTPTrainingConfig()
        self.assertEqual((config.learning_rate, config.weight_decay, config.seed, config.patience), (1e-4, 1e-4, 630, 3))
        for epochs in (0, 11, True, 1.5):
            with self.assertRaisesRegex(ValueError, "max_epochs"):
                CTPTrainingConfig(max_epochs=epochs)
        with self.assertRaises(TypeError):
            CTPTrainingConfig(learning_rate=1e-3)

    def test_complete_patient_steps_and_separate_no_gradient_validation(self):
        holder, snapshots, targets, validation_step_counts = {}, {}, [], []
        steps = []
        real_step = torch.optim.AdamW.step

        def factory(width):
            self.assertEqual(width, 16)
            # Smaller channels only in this multi-patient synthetic test.
            model = CompactCTPMean(width=4)
            holder["model"] = model
            backward = model.backward_series

            def checked_backward(series, target):
                targets.append((self.repo.opened[-1][0], int(target.item())))
                return backward(series, target)

            model.backward_series = checked_backward
            return model

        def inspect_frame(series, index):
            model = holder["model"]
            if series not in snapshots:
                snapshots[series] = {name: p.detach().clone() for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, snapshots[series][name], rtol=0, atol=0)
            if model.training:
                self.assertEqual(torch.is_grad_enabled(), len(series.calls) > 40)
            else:
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                validation_step_counts.append(len(steps))

        def checked_step(optimizer, *args, **kwargs):
            model = holder["model"]
            series = self.repo.opened[-1][1]
            self.assertEqual(series.calls, list(range(40)) * 2)
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, snapshots[series][name], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(p.grad).all().item())
            outcome = real_step(optimizer, *args, **kwargs)
            self.assertTrue(all(torch.isfinite(p).all().item() for p in model.parameters()))
            self.assertFalse(torch.equal(model.encoder.stem[0].weight, snapshots[series]["encoder.stem.0.weight"]))
            self.assertFalse(torch.equal(model.head.weight, snapshots[series]["head.weight"]))
            steps.append(1)
            return outcome

        self.repo.on_frame = inspect_frame
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(ctp_training, "CompactCTPMean", side_effect=factory), \
             patch.object(torch.optim.AdamW, "step", checked_step), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.run_fixture("one_epoch")
        self.assertEqual(result.optimizer_steps, len(self.plan.inner_train))
        self.assertEqual(len(steps), len(self.plan.inner_train))
        self.assertEqual(set(validation_step_counts), {len(steps)})
        for patient, series in self.repo.opened:
            self.assertNotIn(patient, self.plan.outer_test)
            expected = list(range(40)) * (2 if patient in self.plan.inner_train else 1)
            self.assertEqual(series.calls, expected)
        for patient, target in targets:
            self.assertEqual(target, binary_outcome(patient))
        final_parameters = dict(holder["model"].named_parameters())
        for patient, series in self.repo.opened:
            if patient in self.plan.inner_validation:
                for name, expected in snapshots[series].items():
                    torch.testing.assert_close(final_parameters[name], expected, rtol=0, atol=0)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        metrics = json.loads(result.metrics_path.read_text())
        self.assertEqual(metrics["validation_patient_visits"], len(self.plan.inner_validation))
        config = json.loads((result.metrics_path.parent / "run_config.json").read_text())
        self.assertEqual(config["optimizer"]["name"], "AdamW")
        self.assertEqual(config["optimizer"]["lr"], 1e-4)
        self.assertEqual(config["optimizer"]["weight_decay"], 1e-4)
        self.assertEqual(config["model"]["intensity_divisor"], 255.0)
        for record in result.metrics_path.parent.glob("*.json"):
            contents = record.read_text()
            self.assertNotIn(str(self.base), contents)
            for patient in self.contract.patients:
                self.assertNotIn(patient.patient_id, contents)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(result.checkpoint_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(result.metrics_path.parent.stat().st_mode), 0o700)

    def test_width_16_smoke_is_one_complete_update_without_validation(self):
        models, before = [], {}

        def factory(width):
            self.assertEqual(width, 16)
            model = CompactCTPMean(width=width)
            models.append(model)
            before.update({name: p.detach().clone() for name, p in model.named_parameters()})
            return model

        with patch.object(ctp_training, "CompactCTPMean", side_effect=factory), \
             patch.object(ctp_training, "_validation_bce", side_effect=AssertionError("validation accessed")):
            result = self.run_fixture("single_patient", mode="smoke", max_epochs=10)
        self.assertEqual([p for p, _ in self.repo.opened], [self.plan.inner_train[0]])
        self.assertEqual(self.repo.opened[0][1].calls, list(range(40)) * 2)
        self.assertEqual((result.optimizer_steps, result.epochs, result.best_epoch), (1, (), None))
        self.assertFalse(torch.equal(models[0].encoder.stem[0].weight, before["encoder.stem.0.weight"]))
        self.assertFalse(torch.equal(models[0].head.weight, before["head.weight"]))
        self.assertTrue(all(torch.isfinite(p).all().item() for p in models[0].parameters()))
        metrics = json.loads(result.metrics_path.read_text())
        self.assertEqual(metrics["validation_patient_visits"], 0)
        self.assertIsNone(metrics["best_validation_bce"])
        self.assertEqual(metrics["epochs"], [])  # No individual smoke loss/logit.
        saved = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(saved[name], value, rtol=0, atol=0)

    def test_private_location_and_existing_run_rejected_before_model_or_images(self):
        paths = create_private_run(self.base, "existing", 0)
        with open_private_binary(paths.metrics_path) as stream:
            stream.write(b"original")
        with patch.object(ctp_training, "CompactCTPMean") as factory:
            with self.assertRaisesRegex(PrivateOutputError, "already exists"):
                self.run_fixture("existing", mode="smoke")
            for location in (self.root, "relative", Path(__file__).resolve().parents[1] / "docs"):
                with self.assertRaises(PrivateOutputError):
                    run_ctp_development(self.contract, self.repo, mode="smoke", private_output_dir=location, run_id="blocked")
            factory.assert_not_called()
        self.assertEqual(paths.metrics_path.read_bytes(), b"original")
        self.assertEqual(self.repo.opened, [])

    def test_second_pass_failure_clears_gradients_and_sanitizes_error(self):
        models = []

        def factory(width):
            model = CompactCTPMean(width=4)
            models.append(model)
            return model

        def fail_on_recompute(series, index):
            if len(series.calls) == 41:
                raise OSError("fictional-private-image /private/path")

        self.repo.on_frame = fail_on_recompute
        with patch.object(ctp_training, "CompactCTPMean", side_effect=factory), \
             patch.object(torch.optim.AdamW, "step") as step:
            with self.assertRaises(CTPTrainingError) as caught:
                self.run_fixture("failed_recompute", mode="smoke")
            step.assert_not_called()
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn("fictional-private", str(caught.exception))
        self.assertTrue(all(p.grad is None for p in models[0].parameters()))
        folder = self.base / "failed_recompute" / "outer_fold_0"
        self.assertEqual(json.loads((folder / "failed.json").read_text())["optimizer_steps_attempted"], 0)
        self.assertFalse(list(folder.glob("*.pt")))
        self.assertFalse((folder / "aggregate_training.json").exists())

    def test_nonfinite_gradient_prevents_update(self):
        def factory(width):
            model = CompactCTPMean(width=4)
            next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            return model

        with patch.object(ctp_training, "CompactCTPMean", side_effect=factory), \
             patch.object(torch.optim.AdamW, "step") as step:
            with self.assertRaises(CTPTrainingError):
                self.run_fixture("nonfinite", mode="smoke")
            step.assert_not_called()
        self.assertFalse(list((self.base / "nonfinite").rglob("*.pt")))

    def test_checkpoint_ties_keep_earlier_epoch_and_patience_stops(self):
        class TinyModel(torch.nn.Module):
            configuration = {"synthetic_checkpoint_fixture": True}

            def __init__(self, width):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))

            def backward_series(self, series, target):
                self.zero_grad(set_to_none=True)
                for _ in range(2):
                    for index in range(40):
                        series.load_frame(index)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.bias, target)
                loss.backward()
                return CTPBackwardResult(loss.detach(), self.bias.detach())

        with patch.object(ctp_training, "CompactCTPMean", TinyModel), \
             patch.object(ctp_training, "_validation_bce", side_effect=[1.0, 1.0, 0.9, 0.9, 0.9, 0.9]):
            result = self.run_fixture("selection", max_epochs=10, job_epochs=10)
        self.assertEqual(result.best_epoch, 3)
        self.assertEqual(len(result.epochs), 6)
        self.assertEqual(result.optimizer_steps, 6 * len(self.plan.inner_train))
        self.assertEqual(result.checkpoint_path.name, "checkpoint_epoch_003.pt")
        checkpoints = sorted(p.name for p in result.metrics_path.parent.glob("checkpoint_epoch_*.pt"))
        self.assertEqual(checkpoints, ["checkpoint_epoch_001.pt", "checkpoint_epoch_003.pt"])


if __name__ == "__main__":
    unittest.main()
