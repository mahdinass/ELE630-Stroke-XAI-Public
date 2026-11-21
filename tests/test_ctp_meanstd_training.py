"""Fictional fold-0 training and continuation tests for the mean+std challenger."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630 import ncct_private_output
from ele630.ncct_splits import plan_outer_fold
from test_ctp_training import FakeRepository
from test_ncct_stage4a import fictional_contract

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from ele630 import ctp_continuation, ctp_training
    from ele630.ctp_model import CompactCTPMean, CompactCTPMeanStd
    from ele630.ctp_training import CTPTrainingConfig, CTPTrainingError, run_ctp_development


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CTPMeanStdTrainingTests(unittest.TestCase):
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
        self.repo = FakeRepository()
        self.models = []

        class TinyEncoder(nn.Module):
            def __init__(nested_self, width):
                super().__init__()
                nested_self.width = width
                nested_self.projection = nn.Linear(1, 8 * width)

            def forward_features(nested_self, frame):
                return nested_self.projection(frame.mean(dim=(1, 2, 3, 4)).unsqueeze(1))

        def meanstd_factory(width):
            model = CompactCTPMeanStd(width=width)
            model.encoder = TinyEncoder(width)
            self.models.append(model)
            return model

        def mean_factory(width):
            model = CompactCTPMean(width=width)
            model.encoder = TinyEncoder(width)
            return model

        self.meanstd_factory = meanstd_factory
        self.mean_factory = mean_factory

    def run_job(self, run_id, *, epochs=1, max_epochs=2, resume=False,
                repository=None, factory=None):
        return run_ctp_development(
            self.contract, self.repo if repository is None else repository,
            mode="development", private_output_dir=self.base, run_id=run_id,
            config=CTPTrainingConfig(max_epochs=max_epochs), epochs_this_job=epochs,
            resume=resume, model_factory=self.meanstd_factory if factory is None else factory,
        )

    def latest_state(self, run_id):
        folder = self.base / run_id / "outer_fold_0"
        return ctp_continuation.read_committed_state(folder)[2]

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_development_uses_same_split_all_frames_one_step_and_no_grad_validation(self):
        steps = []
        real_step = torch.optim.AdamW.step

        def checked_step(optimizer, *args, **kwargs):
            self.assertEqual(self.repo.opened[-1][1].calls, list(range(40)) * 2)
            steps.append(len(self.repo.opened))
            return real_step(optimizer, *args, **kwargs)

        original_forward = CompactCTPMeanStd.forward

        def checked_forward(model, series):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return original_forward(model, series)

        with patch.object(torch.optim.AdamW, "step", checked_step), \
             patch.object(CompactCTPMeanStd, "forward", checked_forward):
            result = self.run_job("meanstd_one_epoch", max_epochs=1)
        self.assertEqual(steps, list(range(1, len(self.plan.inner_train) + 1)))
        self.assertEqual(result.optimizer_steps, len(self.plan.inner_train))
        self.assertEqual(len(self.repo.opened),
                         len(self.plan.inner_train) + len(self.plan.inner_validation))
        self.assertEqual({patient for patient, _ in self.repo.opened[:len(self.plan.inner_train)]},
                         set(self.plan.inner_train))
        self.assertEqual([patient for patient, _ in self.repo.opened[len(self.plan.inner_train):]],
                         list(self.plan.inner_validation))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))
        self.assertTrue(all(series.calls == list(range(40)) * 2
                            for _, series in self.repo.opened[:len(self.plan.inner_train)]))
        self.assertTrue(all(series.calls == list(range(40))
                            for _, series in self.repo.opened[len(self.plan.inner_train):]))
        saved = json.loads((self.base / "meanstd_one_epoch" / "outer_fold_0" / "run_config.json").read_text())
        self.assertEqual(saved["model"]["temporal_pooling"], "mean_std_population")
        self.assertEqual(saved["model"]["head_input_dim"], 256)
        self.assertEqual(saved["training"], {"max_epochs": 1, "device": "cpu", "patience": 3,
                                             "learning_rate": 1e-4, "weight_decay": 1e-4, "seed": 630})
        folder = self.base / "meanstd_one_epoch" / "outer_fold_0"
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
            self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600
                                for path in folder.iterdir() if path.is_file()))

    def test_nonfinite_gradient_prevents_optimizer_step_and_sanitizes_error(self):
        count = 0

        def faulty_factory(width):
            model = self.meanstd_factory(width)
            original = model.backward_series

            def corrupt(series, target):
                result = original(series, target)
                next(model.parameters()).grad.fill_(float("nan"))
                return result

            model.backward_series = corrupt
            return model

        real_step = torch.optim.AdamW.step

        def counted_step(optimizer, *args, **kwargs):
            nonlocal count
            count += 1
            return real_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.AdamW, "step", counted_step):
            with self.assertRaises(CTPTrainingError) as caught:
                self.run_job("meanstd_bad_gradient", factory=faulty_factory)
        self.assertEqual(count, 0)
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(str(self.base), str(caught.exception))
        for patient in self.contract.patients:
            self.assertNotIn(patient.patient_id, str(caught.exception))
        self.assertTrue(all(patient not in self.plan.outer_test for patient, _ in self.repo.opened))

    def test_continuation_matches_uninterrupted_weights_optimizer_rng_and_order(self):
        original_validation = ctp_training._validation_bce

        def run_sequence(run_id, bounds):
            repository = FakeRepository()

            def validation_with_rng(model, data, device):
                value = original_validation(model, data, device)
                return value + 0.001 * (random.random() + float(np.random.random()) +
                                        float(torch.rand(()).item()))

            with patch.object(ctp_training, "_validation_bce", side_effect=validation_with_rng):
                result = None
                for job, bound in enumerate(bounds):
                    result = self.run_job(run_id, epochs=bound, max_epochs=2,
                                          resume=job > 0, repository=repository)
            return result, repository

        continuous, continuous_repo = run_sequence("meanstd_continuous", [2])
        divided, divided_repo = run_sequence("meanstd_divided", [1, 1])
        self.assertEqual(continuous.epochs, divided.epochs)
        self.assertEqual(continuous.best_epoch, divided.best_epoch)
        self.assertEqual(continuous.optimizer_steps, divided.optimizer_steps)
        self.assertEqual([patient for patient, _ in continuous_repo.opened],
                         [patient for patient, _ in divided_repo.opened])
        left_state = self.latest_state("meanstd_continuous")
        right_state = self.latest_state("meanstd_divided")
        for field in ("model_state", "optimizer_state", "rng", "history", "optimizer_steps",
                      "best_epoch", "best_validation_bce", "stale_epochs", "completed_epoch",
                      "next_epoch", "experiment_signature"):
            self.assert_nested_equal(left_state[field], right_state[field])
        self.assertEqual(json.loads(divided.metrics_path.read_text())["status"], "complete")

    def test_mean_checkpoint_cannot_resume_as_meanstd_before_image_access(self):
        with patch.object(ctp_training, "CompactCTPMean", self.mean_factory):
            run_ctp_development(
                self.contract, self.repo, mode="development", private_output_dir=self.base,
                run_id="fictional_mean", config=CTPTrainingConfig(max_epochs=2), epochs_this_job=1,
            )
        self.repo.opened.clear()
        with self.assertRaises(CTPTrainingError) as caught:
            self.run_job("fictional_mean", resume=True)
        self.assertEqual(self.repo.opened, [])
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(str(self.base), str(caught.exception))

    def test_interrupted_epoch_restarts_from_last_complete_meanstd_commit(self):
        self.run_job("meanstd_intact", epochs=2)
        self.run_job("meanstd_interrupted", epochs=1)
        failed_repo = FakeRepository()

        def fail_during_recomputation(series, index):
            if len(series.calls) == 41:
                raise OSError("fictional interrupted frame")

        failed_repo.on_frame = fail_during_recomputation
        with self.assertRaises(CTPTrainingError):
            self.run_job("meanstd_interrupted", resume=True, repository=failed_repo)
        folder = self.base / "meanstd_interrupted" / "outer_fold_0"
        self.assertEqual(len(list(folder.glob("commit_epoch_*.json"))), 1)
        self.run_job("meanstd_interrupted", resume=True, repository=FakeRepository())
        intact = self.latest_state("meanstd_intact")
        recovered = self.latest_state("meanstd_interrupted")
        for field in ("model_state", "optimizer_state", "rng", "history", "optimizer_steps"):
            self.assert_nested_equal(intact[field], recovered[field])


if __name__ == "__main__":
    unittest.main()
