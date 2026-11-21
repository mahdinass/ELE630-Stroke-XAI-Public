"""Fictional, small-frame TCN training and exact continuation tests."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import stat
import unittest
from unittest.mock import patch

import numpy as np

from test_ctp_training import FakeRepository
import test_ctp_meanstd_training as fixture

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from ele630 import ctp_training
    from ele630.ctp_model import CompactCTPShallowTCN
    from ele630.ctp_training import CTPTrainingConfig, CTPTrainingError, run_ctp_development


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CTPTCNTrainingTests(unittest.TestCase):
    # Reuse only fictional-contract/private-directory fixture helpers. No
    # inheritance of legacy tests and no changes to their model factories.
    def setUp(self):
        fixture.CTPMeanStdTrainingTests.setUp(self)
        self.initial_states = []

        class TinyEncoder(nn.Module):
            def __init__(nested_self):
                super().__init__()
                nested_self.width = 16
                nested_self.projection = nn.Linear(1, 128)

            def forward_features(nested_self, frame):
                return nested_self.projection(frame.mean(dim=(1, 2, 3, 4)).unsqueeze(1))

        def tcn_factory(width):
            model = CompactCTPShallowTCN(width=width)
            # The real ResNet gradient test is in test_ctp_tcn.py. This small
            # deterministic substitute keeps multi-epoch runner tests cheap.
            model.encoder = TinyEncoder()
            self.initial_states.append({name: value.clone() for name, value in model.state_dict().items()})
            self.models.append(model)
            return model

        self.tcn_factory = tcn_factory

    latest_state = fixture.CTPMeanStdTrainingTests.latest_state
    assert_nested_equal = fixture.CTPMeanStdTrainingTests.assert_nested_equal

    def run_job(self, run_id, *, epochs=1, max_epochs=2, resume=False,
                repository=None, factory=None):
        return run_ctp_development(
            self.contract, self.repo if repository is None else repository,
            mode="development", private_output_dir=self.base, run_id=run_id,
            config=CTPTrainingConfig(max_epochs=max_epochs), epochs_this_job=epochs,
            resume=resume, model_factory=self.tcn_factory if factory is None else factory,
        )

    def test_fresh_training_steps_only_after_complete_backward_and_validation_is_no_grad(self):
        steps = []
        real_step = torch.optim.AdamW.step

        def checked_step(optimizer, *args, **kwargs):
            self.assertEqual(self.repo.opened[-1][1].calls, list(range(40)) * 2)
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in self.models[-1].parameters()))
            steps.append(len(self.repo.opened))
            return real_step(optimizer, *args, **kwargs)

        real_forward = CompactCTPShallowTCN.forward

        def checked_forward(model, series):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            before = {name: value.clone() for name, value in model.state_dict().items()}
            output = real_forward(model, series)
            self.assert_nested_equal(before, model.state_dict())
            return output

        with patch.object(torch.optim.AdamW, "step", checked_step), \
             patch.object(CompactCTPShallowTCN, "forward", checked_forward):
            result = self.run_job("tcn_fresh", max_epochs=1)
        training_count = len(self.plan.inner_train)
        self.assertEqual(steps, list(range(1, training_count + 1)))
        self.assertEqual(result.optimizer_steps, training_count)
        self.assertEqual({patient for patient, _ in self.repo.opened[:training_count]},
                         set(self.plan.inner_train))
        self.assertEqual([patient for patient, _ in self.repo.opened[training_count:]],
                         list(self.plan.inner_validation))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))
        self.assertTrue(all(series.calls == list(range(40)) * 2
                            for _, series in self.repo.opened[:training_count]))
        self.assertTrue(all(series.calls == list(range(40))
                            for _, series in self.repo.opened[training_count:]))
        for prefix in ("encoder.", "temporal.0.", "temporal.2.", "head."):
            self.assertTrue(any(not torch.equal(value, self.initial_states[0][name])
                                for name, value in self.models[-1].state_dict().items()
                                if name.startswith(prefix)))
        folder = self.base / "tcn_fresh" / "outer_fold_0"
        saved = json.loads((folder / "run_config.json").read_text())
        self.assertEqual(saved["model"], self.models[-1].configuration)
        self.assertEqual(saved["training"], {"max_epochs": 1, "device": "cpu", "patience": 3,
                                             "learning_rate": 1e-4, "weight_decay": 1e-4, "seed": 630})
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
            self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600
                                for path in folder.iterdir() if path.is_file()))

    def test_continuation_is_exact_for_weights_optimizer_rng_history_order_and_selection(self):
        original_validation = ctp_training._validation_bce

        def run_sequence(run_id, bounds):
            repository = FakeRepository()

            def validation_with_rng(model, data, device):
                value = original_validation(model, data, device)
                return value + 0.001 * (random.random() + float(np.random.random()) +
                                        float(torch.rand(()).item()))

            with patch.object(ctp_training, "_validation_bce", side_effect=validation_with_rng):
                for job, bound in enumerate(bounds):
                    result = self.run_job(run_id, epochs=bound, resume=job > 0, repository=repository)
            return result, repository

        continuous, continuous_repo = run_sequence("tcn_continuous", [2])
        divided, divided_repo = run_sequence("tcn_divided", [1, 1])
        self.assertEqual(continuous.epochs, divided.epochs)
        self.assertEqual(continuous.best_epoch, divided.best_epoch)
        self.assertEqual(continuous.optimizer_steps, divided.optimizer_steps)
        self.assertEqual([patient for patient, _ in continuous_repo.opened],
                         [patient for patient, _ in divided_repo.opened])
        left = self.latest_state("tcn_continuous")
        right = self.latest_state("tcn_divided")
        for field in ("model_state", "optimizer_state", "rng", "history", "optimizer_steps",
                      "best_epoch", "best_validation_bce", "stale_epochs", "completed_epoch",
                      "next_epoch", "experiment_signature"):
            self.assert_nested_equal(left[field], right[field])
        self.assertEqual(json.loads(divided.metrics_path.read_text())["status"], "complete")

    def test_interrupted_recomputation_recovers_from_last_complete_commit(self):
        self.run_job("tcn_intact", epochs=2)
        self.run_job("tcn_interrupted")
        repository = FakeRepository()

        def fail_during_recomputation(series, index):
            if len(series.calls) == 41:
                raise OSError("fictional interrupted frame")

        repository.on_frame = fail_during_recomputation
        with self.assertRaises(CTPTrainingError):
            self.run_job("tcn_interrupted", resume=True, repository=repository)
        folder = self.base / "tcn_interrupted" / "outer_fold_0"
        self.assertEqual(len(list(folder.glob("commit_epoch_*.json"))), 1)
        self.run_job("tcn_interrupted", resume=True, repository=FakeRepository())
        intact = self.latest_state("tcn_intact")
        recovered = self.latest_state("tcn_interrupted")
        for field in ("model_state", "optimizer_state", "rng", "history", "optimizer_steps",
                      "best_epoch", "best_validation_bce", "stale_epochs", "next_epoch"):
            self.assert_nested_equal(intact[field], recovered[field])

    def test_all_cross_architecture_resumes_fail_before_image_access(self):
        directions = ((self.tcn_factory, self.mean_factory),
                      (self.tcn_factory, self.meanstd_factory),
                      (self.mean_factory, self.tcn_factory),
                      (self.meanstd_factory, self.tcn_factory))
        for index, (source, destination) in enumerate(directions):
            with self.subTest(direction=index):
                run_id = f"fictional_architecture_{index}"
                self.run_job(run_id, factory=source)
                self.repo.opened.clear()
                with self.assertRaises(CTPTrainingError) as caught:
                    self.run_job(run_id, factory=destination, resume=True)
                self.assertEqual(self.repo.opened, [])
                self.assertIsNone(caught.exception.__cause__)
                self.assertNotIn(str(self.base), str(caught.exception))
                for patient in self.contract.patients:
                    self.assertNotIn(patient.patient_id, str(caught.exception))

    def test_nonfinite_gradient_stops_before_step_with_sanitized_error(self):
        def faulty_factory(width):
            model = self.tcn_factory(width)
            original = model.backward_series

            def corrupt(series, target):
                result = original(series, target)
                model.temporal[0].weight.grad.fill_(float("nan"))
                return result

            model.backward_series = corrupt
            return model

        with patch.object(torch.optim.AdamW, "step") as step:
            with self.assertRaises(CTPTrainingError) as caught:
                self.run_job("tcn_bad_gradient", factory=faulty_factory)
        step.assert_not_called()
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(str(self.base), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
