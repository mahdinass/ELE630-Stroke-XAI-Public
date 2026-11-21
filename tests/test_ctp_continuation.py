"""Fictional, tiny CPU series for bounded CTP continuation semantics."""

from __future__ import annotations

import importlib.util
import copy
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
from ele630.ncct_private_output import PrivateOutputError, create_private_run, open_private_run
from test_ctp_training import FakeRepository, fictional_contract


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ele630 import ctp_continuation, ctp_training
    from ele630.ctp_model import CTPBackwardResult
    from ele630.ctp_training import CTPTrainingConfig, CTPTrainingError, run_ctp_development


class PrivateReopenTests(unittest.TestCase):
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

    def test_existing_run_is_opened_without_creating_or_overwriting(self):
        created = create_private_run(self.base, "fictional_run", 0)
        self.assertEqual(open_private_run(self.base, "fictional_run", 0), created)
        with self.assertRaisesRegex(PrivateOutputError, "already exists"):
            create_private_run(self.base, "fictional_run", 0)
        with self.assertRaises(PrivateOutputError):
            open_private_run(self.base, "missing", 0)
        with self.assertRaises(PrivateOutputError):
            open_private_run(self.root, "fictional_run", 0)
        with self.assertRaises(PrivateOutputError):
            open_private_run(self.base / "..", "fictional_run", 0)

    def test_symlinked_run_and_fold_paths_rejected(self):
        outside = self.root / "outside"
        outside.mkdir(mode=0o700)
        for target, name in ((outside, "escape"), (self.base / "safe", "inside")):
            if name == "inside":
                target.mkdir(mode=0o700)
            link = self.base / name
            try:
                link.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("local filesystem does not permit synthetic symlinks")
            with self.assertRaises(PrivateOutputError):
                open_private_run(self.base, name, 0)
        created = create_private_run(self.base, "normal", 0)
        self.assertEqual(open_private_run(self.base, "normal", 0), created)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CTPContinuationTests(unittest.TestCase):
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
        self.repo = FakeRepository()
        self.models = []

        class TinyTwoPass(torch.nn.Module):
            configuration = {"synthetic_two_pass": True, "frame_count": 40}

            def __init__(nested_self, width):
                super().__init__()
                if width != 16:
                    raise AssertionError("runner changed model width")
                nested_self.encoder = torch.nn.Linear(1, 2)
                nested_self.head = torch.nn.Linear(2, 1)
                self.models.append(nested_self)

            @staticmethod
            def _input(frame):
                return torch.tensor([[float(frame.input.tensor.mean())]], dtype=torch.float32)

            @staticmethod
            def _noise():
                return (random.random() + float(np.random.random()) +
                        float(torch.rand(()).item())) * 0.01

            def _mean(nested_self, series):
                total = torch.zeros((1, 2))
                for index in range(40):
                    total += nested_self.encoder(nested_self._input(series.load_frame(index)))
                return total / 40

            def forward(nested_self, series):
                if torch.is_grad_enabled():
                    raise AssertionError("validation retained a graph")
                return nested_self.head(nested_self._mean(series)).view(1) + nested_self._noise()

            def backward_series(nested_self, series, target):
                nested_self.zero_grad(set_to_none=True)
                with torch.no_grad():
                    mean = nested_self._mean(series)
                mean.requires_grad_(True)
                logit = nested_self.head(mean).view(1) + nested_self._noise()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
                loss.backward()
                gradient = mean.grad.detach() / 40
                for index in range(40):
                    nested_self.encoder(nested_self._input(series.load_frame(index))).backward(gradient)
                return CTPBackwardResult(loss.detach(), logit.detach())

        self.model_patch = patch.object(ctp_training, "CompactCTPMean", TinyTwoPass)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def run_job(self, run_id, *, epochs=1, resume=False, max_epochs=4, repository=None,
                contract=None, config=None):
        return run_ctp_development(
            self.contract if contract is None else contract,
            self.repo if repository is None else repository,
            mode="development", private_output_dir=self.base, run_id=run_id,
            config=config if config is not None else CTPTrainingConfig(max_epochs=max_epochs),
            epochs_this_job=epochs, resume=resume,
        )

    def latest_state(self, run_id):
        folder = self.base / run_id / "outer_fold_0"
        markers = sorted(folder.glob("commit_epoch_*.json"))
        self.assertTrue(markers)
        marker = json.loads(markers[-1].read_text())
        state = torch.load(folder / marker["state_file"], map_location="cpu", weights_only=True)
        return marker, state

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_uninterrupted_and_split_jobs_match_exactly(self):
        original_validation = ctp_training._validation_bce
        schedule = [1.0, 1.0, 0.9, 0.9, 0.9, 0.9]

        def comparison(run_id, job_limits):
            repository = FakeRepository()
            values = iter(schedule)

            def checked_validation(model, data, device):
                original_validation(model, data, device)  # All 40 frames, once.
                return next(values)

            with patch.object(ctp_training, "_validation_bce", side_effect=checked_validation):
                result = None
                for number, limit in enumerate(job_limits):
                    result = self.run_job(run_id, epochs=limit, resume=number > 0,
                                          max_epochs=10, repository=repository)
            return result, repository

        continuous, continuous_repo = comparison("continuous", [10])
        divided, divided_repo = comparison("divided", [1, 2, 3])
        self.assertEqual(len(continuous.epochs), 6)
        self.assertEqual(continuous.epochs, divided.epochs)
        self.assertEqual(continuous.optimizer_steps, divided.optimizer_steps)
        self.assertEqual(continuous.best_epoch, divided.best_epoch)
        self.assertEqual(continuous.best_validation_bce, divided.best_validation_bce)
        self.assertEqual(continuous.best_epoch, 3)
        self.assertEqual(json.loads(divided.metrics_path.read_text())["status"], "complete")
        self.assertEqual([p for p, _ in continuous_repo.opened],
                         [p for p, _ in divided_repo.opened])
        for repository in (continuous_repo, divided_repo):
            self.assertTrue(all(patient.fold != 0 for patient, _ in repository.opened))
            for _, series in repository.opened:
                self.assertIn(series.calls, (list(range(40)), list(range(40)) * 2))
        left_marker, left = self.latest_state("continuous")
        right_marker, right = self.latest_state("divided")
        for field in ("model_state", "optimizer_state", "rng", "history",
                      "optimizer_steps", "best_epoch", "best_validation_bce", "stale_epochs",
                      "completed_epoch", "next_epoch", "experiment_signature"):
            self.assert_nested_equal(left[field], right[field])
        self.assertEqual(left["stale_epochs"], 3)
        self.assertEqual(left["optimizer_steps"], 6 * 6)
        self.assertEqual(sorted(m.stem.split("_")[2] for m in (self.base / "continuous" / "outer_fold_0").glob("checkpoint_epoch_*.pt")),
                         ["001", "003"])
        self.assertEqual(left_marker["checkpoint_selected"], right_marker["checkpoint_selected"])

    def test_interruption_restarts_only_from_last_complete_epoch(self):
        intact_repo = FakeRepository()
        intact = self.run_job("intact", epochs=2, max_epochs=3, repository=intact_repo)
        self.assertEqual(len(intact.epochs), 2)
        self.run_job("interrupted", max_epochs=3)
        failed_repo = FakeRepository()

        def interrupt_second_pass(series, index):
            if len(series.calls) == 41:
                raise OSError("fictional interrupted image")

        failed_repo.on_frame = interrupt_second_pass
        with self.assertRaises(CTPTrainingError):
            self.run_job("interrupted", epochs=1, resume=True, max_epochs=3,
                         repository=failed_repo)
        self.assertEqual(len(list((self.base / "interrupted" / "outer_fold_0").glob("commit_epoch_*.json"))), 1)
        recovered = self.run_job("interrupted", epochs=1, resume=True, max_epochs=3)
        self.assertEqual(len(recovered.epochs), 2)
        _, expected = self.latest_state("intact")
        _, actual = self.latest_state("interrupted")
        for field in ("model_state", "optimizer_state", "rng", "history", "optimizer_steps"):
            self.assert_nested_equal(expected[field], actual[field])

    def test_orphan_state_after_commit_interruption_is_ignored(self):
        self.run_job("orphan", max_epochs=3)
        real_publish = ctp_continuation.publish_json

        def stop_before_commit(directory, name, value):
            if name.startswith("commit_epoch_002"):
                raise OSError("fictional Slurm interruption")
            return real_publish(directory, name, value)

        with patch.object(ctp_continuation, "publish_json", side_effect=stop_before_commit):
            with self.assertRaises(CTPTrainingError):
                self.run_job("orphan", max_epochs=3, resume=True)
        folder = self.base / "orphan" / "outer_fold_0"
        self.assertEqual(len(list(folder.glob("state_epoch_002_*.pt"))), 1)
        self.assertEqual(len(list(folder.glob("commit_epoch_*.json"))), 1)
        recovered = self.run_job("orphan", max_epochs=3, resume=True)
        self.assertEqual(len(recovered.epochs), 2)
        self.assertEqual(len(list(folder.glob("commit_epoch_002_*.json"))), 1)

    def test_incompatible_and_incomplete_state_rejected_before_images(self):
        self.run_job("mismatch", max_epochs=3)
        folder = self.base / "mismatch" / "outer_fold_0"
        config = folder / "run_config.json"
        original = config.read_bytes()
        value = json.loads(original)
        value["training"]["seed"] = 999
        config.write_text(json.dumps(value), encoding="utf-8")
        before = len(self.repo.opened)
        with self.assertRaises(CTPTrainingError):
            self.run_job("mismatch", resume=True, max_epochs=3)
        self.assertEqual(len(self.repo.opened), before)
        config.write_bytes(original)
        marker, _ = self.latest_state("mismatch")
        state_path = folder / marker["state_file"]
        state_path.write_bytes(b"broken")
        with self.assertRaises(CTPTrainingError):
            self.run_job("mismatch", resume=True, max_epochs=3)
        self.assertEqual(len(self.repo.opened), before)
        with self.assertRaises(CTPTrainingError):
            self.run_job("mismatch", resume=True, max_epochs=3,
                         config=CTPTrainingConfig(max_epochs=2))
        self.assertEqual(len(self.repo.opened), before)

    def test_validly_encoded_but_invalid_epoch_optimizer_and_rng_rejected(self):
        self.run_job("invalid_state", max_epochs=3)
        folder = self.base / "invalid_state" / "outer_fold_0"
        marker_path = next(folder.glob("commit_epoch_*.json"))
        original_marker = json.loads(marker_path.read_text())
        state_path = folder / original_marker["state_file"]
        original_state = torch.load(state_path, map_location="cpu", weights_only=True)
        before = len(self.repo.opened)

        def wrong_epoch(state):
            state["next_epoch"] = 9

        def wrong_optimizer(state):
            state["optimizer_state"]["param_groups"][0]["lr"] = 0.01

        def wrong_rng(state):
            state["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)

        for mutate in (wrong_epoch, wrong_optimizer, wrong_rng):
            state = copy.deepcopy(original_state)
            mutate(state)
            torch.save(state, state_path)
            marker = dict(original_marker)
            marker["state_sha256"] = ctp_continuation._digest(state_path)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaises(CTPTrainingError):
                self.run_job("invalid_state", resume=True, max_epochs=3)
            self.assertEqual(len(self.repo.opened), before)

    def test_epoch_bound_and_private_reopen(self):
        for value in (0, 11, True, 1.5):
            with self.assertRaises(ValueError):
                self.run_job("bad_bound", epochs=value)
        self.assertEqual(self.repo.opened, [])
        self.run_job("private", max_epochs=3)
        paths = open_private_run(self.base, "private", 0)
        self.assertEqual(paths.fold_dir.name, "outer_fold_0")
        with self.assertRaises(PrivateOutputError):
            self.run_job("private", max_epochs=3)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(paths.fold_dir.stat().st_mode), 0o700)
            for path in paths.fold_dir.iterdir():
                if path.is_file():
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            os.chmod(paths.fold_dir, 0o755)
            with self.assertRaises(PrivateOutputError):
                self.run_job("private", max_epochs=3, resume=True)
        self.assertEqual(len(self.repo.opened), 8)  # Six training and two validation.


if __name__ == "__main__":
    unittest.main()
