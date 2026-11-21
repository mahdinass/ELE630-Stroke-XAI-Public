"""Fictional patient records and tiny synthetic NCCT arrays only."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.data_contract import DatasetContract, FoldSummary, OutcomePatient, _fingerprint
from ele630.ncct_samples import LazyNCCTSamples
from ele630 import ncct_private_output
from ele630.ncct_private_output import PrivateOutputError, create_private_run, open_private_binary
from ele630.ncct_splits import (
    SplitError, inner_train_positive_weight, plan_outer_fold,
)


def fictional_contract(per_fold: int = 8) -> DatasetContract:
    patients = tuple(
        OutcomePatient(
            f"fictional-{fold}-{index}",
            "A" if index < per_fold // 2 else "B",
            "good" if index % 2 == 0 else "poor",
            index % 3 != 0,
            fold,
        )
        for fold in range(5) for index in range(per_fold)
    )
    return contract_from_patients(patients)


def contract_from_patients(patients: tuple[OutcomePatient, ...]) -> DatasetContract:
    summaries = []
    for fold in range(5):
        subset = [p for p in patients if p.fold == fold]
        summaries.append(FoldSummary(
            fold, len(subset), sum(p.outcome == "good" for p in subset),
            sum(p.outcome == "poor" for p in subset),
            tuple(sorted(Counter(p.cohort for p in subset).items())),
        ))
    return DatasetContract(
        patients, tuple(summaries),
        sum(p.outcome == "good" for p in patients),
        sum(p.outcome == "poor" for p in patients),
        sum(p.has_gt for p in patients),
        _fingerprint({p.patient_id: p.fold for p in patients}), 630,
    )


class FakeNCCTRepository:
    def __init__(self, patients: tuple[OutcomePatient, ...]):
        self.allowed = {p.patient_id for p in patients}
        self.calls: list[str] = []
        self.arrays: dict[str, np.ndarray] = {}

    def load(self, patient: OutcomePatient):
        if patient.patient_id not in self.allowed:
            raise AssertionError("unexpected patient")
        self.calls.append(patient.patient_id)
        index = int(patient.patient_id.rsplit("-", 1)[-1])
        shape = (8 + index % 2, 8, 4 + index % 2)
        data = np.full(shape, 43, dtype=np.int16)
        data[0, 0, 0] = -1024
        data[-1, -1, -1] = 3071
        self.arrays[patient.patient_id] = data
        return SimpleNamespace(data=data, affine=np.diag([-2.0, -2.0, 3.0, 1.0]))


class Stage4ASplitAndSampleTests(unittest.TestCase):
    def setUp(self):
        self.contract = fictional_contract()

    def test_all_five_plans_preserve_outer_assignments_and_exact_coverage(self):
        all_ids = {p.patient_id for p in self.contract.patients}
        for fold in range(5):
            with self.subTest(fold=fold):
                plan = plan_outer_fold(self.contract, fold)
                train = {p.patient_id for p in plan.inner_train}
                val = {p.patient_id for p in plan.inner_validation}
                test = {p.patient_id for p in plan.outer_test}
                self.assertEqual((len(train), len(val), len(test)), (26, 6, 8))
                self.assertEqual(train | val | test, all_ids)
                self.assertFalse(train & val or train & test or val & test)
                self.assertTrue(all(p.fold == fold for p in plan.outer_test))
                self.assertTrue(all(p.fold != fold for p in plan.inner_train + plan.inner_validation))
                for part in (plan.inner_train, plan.inner_validation, plan.outer_test):
                    self.assertEqual({p.outcome for p in part}, {"good", "poor"})
                self.assertEqual(plan.cohort_stratified_outcomes, ("good", "poor"))

    def test_same_seed_repeats_and_different_seed_changes_inner_membership(self):
        first = plan_outer_fold(self.contract, 2, seed=630)
        self.assertEqual(first, plan_outer_fold(self.contract, 2, seed=630))
        variations = [plan_outer_fold(self.contract, 2, seed=s) for s in range(631, 640)]
        self.assertTrue(any(p.inner_validation != first.inner_validation for p in variations))
        self.assertTrue(all(p.outer_test == first.outer_test for p in variations))

    def test_outer_test_labels_cannot_change_inner_split_or_weight(self):
        original = plan_outer_fold(self.contract, 0)
        changed = tuple(
            replace(p, outcome="poor") if p.fold == 0 and p.outcome == "good" and p.patient_id.endswith("-0") else p
            for p in self.contract.patients
        )
        revised = plan_outer_fold(contract_from_patients(changed), 0)
        self.assertEqual(original.inner_train, revised.inner_train)
        self.assertEqual(original.inner_validation, revised.inner_validation)
        self.assertEqual(inner_train_positive_weight(original), inner_train_positive_weight(revised))
        self.assertEqual(original.outer_test[0].outcome, "good")
        self.assertEqual(revised.outer_test[0].outcome, "poor")

    def test_rejects_bad_fold_contract_label_and_composition(self):
        for fold in (-1, 5, True, "0"):
            with self.subTest(fold=fold), self.assertRaisesRegex(SplitError, "outer_fold"):
                plan_outer_fold(self.contract, fold)
        with self.assertRaisesRegex(SplitError, "aggregate counts"):
            plan_outer_fold(replace(self.contract, patients=self.contract.patients[:-1]), 0)
        with self.assertRaisesRegex(SplitError, "fingerprint"):
            plan_outer_fold(replace(self.contract, fingerprint="6a6a203ef27bab5b"), 0)
        invalid = (replace(self.contract.patients[0], outcome="unknown"),) + self.contract.patients[1:]
        with self.assertRaisesRegex(SplitError, "outcome"):
            plan_outer_fold(contract_from_patients(invalid), 0)
        duplicate = self.contract.patients + (self.contract.patients[0],)
        with self.assertRaisesRegex(SplitError, "duplicate"):
            plan_outer_fold(contract_from_patients(duplicate), 0)
        sparse = tuple(
            replace(p, outcome="poor" if p.patient_id in {"fictional-0-1", "fictional-1-1"} else "good")
            for p in self.contract.patients
        )
        with self.assertRaisesRegex(SplitError, "at least two"):
            plan_outer_fold(contract_from_patients(sparse), 0)
        # A fictional 143-row contract cannot masquerade as the active cohort.
        enlarged = self.contract.patients + tuple(
            OutcomePatient(f"fictional-extra-{index}", "A", "good", False, index % 5)
            for index in range(103)
        )
        with self.assertRaisesRegex(SplitError, "active benchmark"):
            plan_outer_fold(contract_from_patients(enlarged), 0)

    def test_small_cohort_strata_fall_back_without_excluding_patients(self):
        patients = tuple(replace(p, cohort="tiny" if p.patient_id == "fictional-1-0" else "main")
                         for p in self.contract.patients)
        plan = plan_outer_fold(contract_from_patients(patients), 0)
        self.assertEqual(len(plan.inner_train) + len(plan.inner_validation) + len(plan.outer_test), 40)
        self.assertNotIn("good", plan.cohort_stratified_outcomes)

    def test_lazy_loading_correct_label_native_shape_bounds_and_no_mutation(self):
        repository = FakeNCCTRepository(self.contract.patients)
        samples = LazyNCCTSamples(self.contract, repository, lower=-20, upper=120)
        self.assertEqual(repository.calls, [])
        good, poor = self.contract.patients[0], self.contract.patients[1]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            first, second = samples.load(good), samples.load(poor)
        self.assertEqual((first.image.shape, second.image.shape), ((1, 4, 8, 8), (1, 5, 8, 9)))
        self.assertEqual((first.target, second.target), (np.float32(0), np.float32(1)))
        self.assertEqual((type(first.target), first.image.dtype), (np.float32, np.dtype(np.float32)))
        self.assertAlmostEqual(float(first.image[0, 1, 1, 1]), 63 / 140, places=6)
        self.assertEqual(float(first.image[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(first.image[0, -1, -1, -1]), 1.0)
        self.assertEqual(repository.arrays[good.patient_id][0, 0, 0], -1024)
        self.assertEqual(repository.arrays[good.patient_id][-1, -1, -1], 3071)
        self.assertFalse(np.shares_memory(first.image, repository.arrays[good.patient_id]))
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertEqual(len(repository.calls), 2)

    def test_sample_rejects_patient_outside_contract_before_repository_access(self):
        repository = FakeNCCTRepository(self.contract.patients)
        samples = LazyNCCTSamples(self.contract, repository, lower=-20, upper=120)
        with self.assertRaisesRegex(SplitError, "not a patient"):
            samples.load(OutcomePatient("fictional-outsider", "A", "good", True, 0))
        with self.assertRaisesRegex(SplitError, "not a patient"):
            samples.load(replace(self.contract.patients[0], outcome="poor"))
        self.assertEqual(repository.calls, [])


class PrivateOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "approved"
        self.root.mkdir(mode=0o700)
        self.base = self.root / "private_training"
        self.base.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o700)
        self.alias_patch = patch.object(ncct_private_output, "_APPROVED_PRIVATE_ROOT", self.root)
        self.canonical_patch = patch.object(ncct_private_output, "_TRUSTED_CANONICAL_ROOT", self.root)
        self.alias_patch.start()
        self.canonical_patch.start()

    def tearDown(self):
        self.canonical_patch.stop()
        self.alias_patch.stop()
        self.temporary.cleanup()

    def test_verified_alias_and_canonical_spellings_resolve_to_same_private_base(self):
        alias = Path(self.temporary.name) / "site_bhome_alias"
        try:
            alias.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable locally")
        with patch.object(ncct_private_output, "_APPROVED_PRIVATE_ROOT", alias):
            via_alias = create_private_run(alias / "private_training", "alias_run", 0)
            via_canonical = create_private_run(self.base, "canonical_run", 0)
        self.assertEqual(via_alias.run_dir.parent, self.base)
        self.assertEqual(via_canonical.run_dir.parent, self.base)
        self.assertTrue(via_alias.run_dir.is_dir())
        self.assertTrue(via_canonical.run_dir.is_dir())

    def test_alias_must_resolve_to_exact_trusted_canonical_root(self):
        wrong = Path(self.temporary.name) / "wrong_root"
        wrong.mkdir(mode=0o700)
        alias = Path(self.temporary.name) / "wrong_alias"
        try:
            alias.symlink_to(wrong, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable locally")
        with patch.object(ncct_private_output, "_APPROVED_PRIVATE_ROOT", alias):
            with self.assertRaisesRegex(PrivateOutputError, "alias does not match"):
                create_private_run(self.base, "must_fail", 0)
        self.assertFalse((self.base / "must_fail").exists())

    def test_trusted_canonical_spelling_cannot_itself_be_a_symlink(self):
        link = Path(self.temporary.name) / "canonical_link"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable locally")
        with patch.object(ncct_private_output, "_TRUSTED_CANONICAL_ROOT", link):
            with self.assertRaisesRegex(PrivateOutputError, "trusted canonical root"):
                create_private_run(self.base, "must_fail", 0)
        self.assertFalse((self.base / "must_fail").exists())

    def test_approved_base_and_restrictive_exclusive_artifacts(self):
        paths = create_private_run(self.base, "synthetic_run_01", 2)
        self.assertEqual(paths.fold_dir.parent, paths.run_dir)
        self.assertEqual(paths.run_dir.parent, self.base)
        checkpoint = paths.fold_dir / "checkpoint_epoch_001.pt"
        with open_private_binary(checkpoint) as stream:
            stream.write(b"synthetic only")
        self.assertEqual(checkpoint.read_bytes(), b"synthetic only")
        with self.assertRaisesRegex(PrivateOutputError, "exclusively"):
            open_private_binary(checkpoint)
        if os.name == "posix":
            for directory in (paths.run_dir, paths.fold_dir):
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(checkpoint.stat().st_mode), 0o600)

    def test_outside_or_tracked_destination_is_rejected(self):
        outside = Path(self.temporary.name) / "public"
        outside.mkdir()
        for location in (outside, self.root, Path(__file__).resolve().parents[1] / "docs", "relative"):
            with self.subTest(location=location), self.assertRaises(PrivateOutputError):
                create_private_run(location, "synthetic_run_02", 0)
        self.assertFalse((outside / "synthetic_run_02").exists())
        self.assertFalse((self.base / "synthetic_run_02").exists())

    def test_run_id_cannot_traverse_or_contain_separators(self):
        for run_id in ("../escape", "nested/run", ".", "", "contains space"):
            with self.subTest(run_id=run_id), self.assertRaisesRegex(PrivateOutputError, "run_id"):
                create_private_run(self.base, run_id, 0)
        self.assertEqual(list(self.base.iterdir()), [])

    def test_symlink_escape_is_rejected(self):
        outside = Path(self.temporary.name) / "external_private"
        outside.mkdir(mode=0o700)
        link = self.root / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable locally")
        with self.assertRaisesRegex(PrivateOutputError, "symlinks|approved"):
            create_private_run(link, "synthetic_run_03", 0)
        self.assertFalse((outside / "synthetic_run_03").exists())

    def test_symlink_back_inside_tree_is_rejected(self):
        link = self.root / "inside_link"
        try:
            link.symlink_to(self.base, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable locally")
        with self.assertRaisesRegex(PrivateOutputError, "symlinks"):
            create_private_run(link, "must_fail", 0)
        self.assertFalse((self.base / "must_fail").exists())

    def test_parent_traversal_is_rejected_even_when_it_lands_inside_root(self):
        traversal = self.base / ".." / "private_training"
        with self.assertRaisesRegex(PrivateOutputError, "parent traversal"):
            create_private_run(traversal, "must_fail", 0)
        self.assertFalse((self.base / "must_fail").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX ownership and mode checks")
    def test_canonical_root_and_base_require_owner_only_permissions(self):
        os.chmod(self.root, 0o750)
        with self.assertRaisesRegex(PrivateOutputError, "canonical runtime root must be owner-only"):
            create_private_run(self.base, "must_fail_root", 0)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o750)
        with self.assertRaisesRegex(PrivateOutputError, "base must be owner-only"):
            create_private_run(self.base, "must_fail_base", 0)
        os.chmod(self.base, 0o700)
        current_uid = os.getuid()
        with patch.object(ncct_private_output.os, "getuid", return_value=current_uid + 1):
            with self.assertRaisesRegex(PrivateOutputError, "canonical runtime root must be owner-only"):
                create_private_run(self.base, "must_fail_owner", 0)
        self.assertEqual(list(self.base.iterdir()), [])

    def test_existing_run_is_rejected_without_overwrite(self):
        first = create_private_run(self.base, "repeat", 0)
        with open_private_binary(first.metrics_path) as stream:
            stream.write(b"original aggregate")
        with self.assertRaisesRegex(PrivateOutputError, "already exists"):
            create_private_run(self.base, "repeat", 0)
        self.assertEqual(first.metrics_path.read_bytes(), b"original aggregate")
        self.assertEqual(list(self.base.iterdir()), [first.run_dir])

    def test_separate_runs_cannot_overwrite_each_others_files(self):
        first = create_private_run(self.base, "run_a", 0)
        second = create_private_run(self.base, "run_b", 0)
        for paths, content in ((first, b"A"), (second, b"B")):
            with open_private_binary(paths.metrics_path) as stream:
                stream.write(content)
            with open_private_binary(paths.fold_dir / "checkpoint_epoch_001.pt") as stream:
                stream.write(content)
        self.assertNotEqual(first.metrics_path, second.metrics_path)
        self.assertEqual(first.metrics_path.read_bytes(), b"A")
        self.assertEqual(second.metrics_path.read_bytes(), b"B")
        self.assertEqual((first.fold_dir / "checkpoint_epoch_001.pt").read_bytes(), b"A")
        self.assertEqual((second.fold_dir / "checkpoint_epoch_001.pt").read_bytes(), b"B")


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ele630 import ncct_training
    from ele630.ncct_training import TrainingConfig, run_one_fold


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class Stage4ATrainingTests(unittest.TestCase):
    def setUp(self):
        self.contract = fictional_contract(per_fold=2)
        self.repository = FakeNCCTRepository(self.contract.patients)
        self.samples = LazyNCCTSamples(self.contract, self.repository, lower=-20, upper=120)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "approved"
        self.root.mkdir(mode=0o700)
        self.base = self.root / "private_training"
        self.base.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o700)
        self.alias_patch = patch.object(ncct_private_output, "_APPROVED_PRIVATE_ROOT", self.root)
        self.canonical_patch = patch.object(ncct_private_output, "_TRUSTED_CANONICAL_ROOT", self.root)
        self.alias_patch.start()
        self.canonical_patch.start()

    def tearDown(self):
        self.canonical_patch.stop()
        self.alias_patch.stop()
        self.temporary.cleanup()

    def test_pooling_configuration_defaults_to_avg_and_rejects_invalid_values(self):
        self.assertEqual(TrainingConfig().pooling, "avg")
        self.assertEqual(TrainingConfig(pooling="avgmax").pooling, "avgmax")
        for value in ("max", "AVG", "", None, 1, True):
            with self.subTest(pooling=value), self.assertRaisesRegex(ValueError, "pooling"):
                TrainingConfig(pooling=value)

    def test_train_only_weight_and_no_outer_test_load_for_one_epoch(self):
        plan = plan_outer_fold(self.contract, 0)
        test_ids = {p.patient_id for p in plan.outer_test}
        self.assertEqual(inner_train_positive_weight(plan), 1.0)
        stdout, stderr = io.StringIO(), io.StringIO()
        model_factory = ncct_training.CompactResNet10
        with patch.object(ncct_training, "CompactResNet10", wraps=model_factory) as models, \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = run_one_fold(
                plan, self.samples, private_output_dir=self.base, run_id="synthetic_smoke",
                config=TrainingConfig(max_epochs=1, positive_weighting=True),
            )
        models.assert_called_once_with(width=16, pooling="avg", stage4_stride=(2, 2, 2))
        self.assertEqual(result.optimizer_steps, len(plan.inner_train))
        self.assertEqual(len(result.epochs), 1)
        self.assertEqual(result.best_epoch, 1)
        self.assertTrue(np.isfinite(result.epochs[0].training_loss))
        self.assertEqual(result.epochs[0].training_loss_kind, "positive_weighted_bce")
        self.assertTrue(np.isfinite(result.epochs[0].validation_bce))
        self.assertTrue(result.checkpoint_path.is_file())
        self.assertTrue(result.metrics_path.is_file())
        self.assertEqual(result.checkpoint_path.parent,
                         self.base / "synthetic_smoke" / "outer_fold_0")
        self.assertFalse(test_ids & set(self.repository.calls))
        self.assertEqual(set(self.repository.calls),
                         {p.patient_id for p in plan.inner_train + plan.inner_validation})
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        metrics = json.loads(result.metrics_path.read_text())
        self.assertEqual(metrics["pooling"], "avg")
        self.assertEqual(metrics["optimizer_steps"], result.optimizer_steps)
        self.assertEqual(metrics["epochs"][0]["training_loss_kind"], "positive_weighted_bce")
        self.assertNotIn("fictional-", result.metrics_path.read_text())
        state = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
        self.assertIn("classifier.weight", state)
        self.assertTrue(torch.isfinite(state["classifier.weight"]).all().item())

    def test_explicit_avgmax_is_constructed_and_recorded_in_private_metrics(self):
        plan = plan_outer_fold(self.contract, 0)
        model_factory = ncct_training.CompactResNet10
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(ncct_training, "CompactResNet10", wraps=model_factory) as models, \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = run_one_fold(
                plan, self.samples, private_output_dir=self.base, run_id="avgmax_smoke",
                config=TrainingConfig(max_epochs=1, pooling="avgmax"),
            )
        models.assert_called_once_with(width=16, pooling="avgmax", stage4_stride=(2, 2, 2))
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        self.assertTrue(result.checkpoint_path.is_file())
        self.assertEqual(result.checkpoint_path.parent,
                         self.base / "avgmax_smoke" / "outer_fold_0")
        self.assertEqual(result.metrics_path.parent, result.checkpoint_path.parent)
        metrics = json.loads(result.metrics_path.read_text())
        self.assertEqual(metrics["pooling"], "avgmax")
        self.assertNotIn("fictional-", result.metrics_path.read_text())
        self.assertFalse({p.patient_id for p in plan.outer_test} & set(self.repository.calls))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(result.checkpoint_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(result.metrics_path.stat().st_mode), 0o600)

    def test_each_fold_creates_new_model_and_optimizer_without_test_access(self):
        model_factory = ncct_training.CompactResNet10
        optimizer_factory = ncct_training.torch.optim.AdamW
        with patch.object(ncct_training, "CompactResNet10", wraps=model_factory) as models, \
             patch.object(ncct_training.torch.optim, "AdamW", wraps=optimizer_factory) as optimizers:
            results = []
            for fold in (0, 1):
                self.repository.calls.clear()
                plan = plan_outer_fold(self.contract, fold)
                results.append(run_one_fold(
                    plan, self.samples, private_output_dir=self.base, run_id=f"synthetic_fold_{fold}",
                    config=TrainingConfig(max_epochs=1),
                ))
                self.assertFalse({p.patient_id for p in plan.outer_test} & set(self.repository.calls))
            self.assertEqual(models.call_count, 2)
            self.assertEqual(optimizers.call_count, 2)
            self.assertNotEqual(results[0].checkpoint_path, results[1].checkpoint_path)
            self.assertTrue(all(item.epochs[0].training_loss_kind == "unweighted_bce" for item in results))

    def test_no_output_without_explicit_safe_absolute_location(self):
        plan = plan_outer_fold(self.contract, 0)
        with self.assertRaisesRegex(PrivateOutputError, "absolute"):
            run_one_fold(plan, self.samples, private_output_dir="relative-output", run_id="bad")
        with self.assertRaisesRegex(PrivateOutputError, "approved"):
            run_one_fold(plan, self.samples,
                         private_output_dir=Path(__file__).resolve().parents[1] / "docs",
                         run_id="bad")
        self.assertEqual(self.repository.calls, [])

    def test_existing_run_fails_before_model_or_sample_access(self):
        plan = plan_outer_fold(self.contract, 0)
        paths = create_private_run(self.base, "existing_run", 0)
        with open_private_binary(paths.metrics_path) as stream:
            stream.write(b"prior result")
        with patch.object(ncct_training, "CompactResNet10") as model_factory:
            with self.assertRaisesRegex(PrivateOutputError, "already exists"):
                run_one_fold(plan, self.samples, private_output_dir=self.base,
                             run_id="existing_run", config=TrainingConfig(max_epochs=1))
            model_factory.assert_not_called()
        self.assertEqual(paths.metrics_path.read_bytes(), b"prior result")
        self.assertEqual(self.repository.calls, [])

    def test_nonfinite_gradient_stops_before_optimizer_step(self):
        plan = plan_outer_fold(self.contract, 0)
        factory = ncct_training.CompactResNet10

        def bad_gradient_model(*, width, pooling, stage4_stride):
            model = factory(width=width, pooling=pooling, stage4_stride=stage4_stride)
            model.classifier.weight.register_hook(
                lambda gradient: torch.full_like(gradient, float("inf")),
            )
            return model

        with patch.object(ncct_training, "CompactResNet10", side_effect=bad_gradient_model), \
             patch.object(ncct_training.torch.optim.AdamW, "step",
                          side_effect=AssertionError("optimizer step must not run")) as step:
            with self.assertRaisesRegex(ValueError, "nonfinite parameter gradient"):
                run_one_fold(plan, self.samples, private_output_dir=self.base,
                             run_id="bad_gradient", config=TrainingConfig(max_epochs=1))
            step.assert_not_called()
        self.assertFalse(list((self.base / "bad_gradient").rglob("*.pt")))

    def test_nonfinite_parameter_after_step_stops(self):
        plan = plan_outer_fold(self.contract, 0)
        original_step = ncct_training.torch.optim.AdamW.step

        def corrupt_step(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            optimizer.param_groups[0]["params"][0].data.fill_(float("inf"))
            return result

        with patch.object(ncct_training.torch.optim.AdamW, "step", corrupt_step):
            with self.assertRaisesRegex(ValueError, "nonfinite model parameter"):
                run_one_fold(plan, self.samples, private_output_dir=self.base,
                             run_id="bad_parameter", config=TrainingConfig(max_epochs=1))
        self.assertFalse(list((self.base / "bad_parameter").rglob("*.pt")))

    def test_weighted_training_and_unweighted_validation_are_named_and_calculated_separately(self):
        contract = fictional_contract(per_fold=3)  # 2 good and 1 poor per outer fold.
        plan = plan_outer_fold(contract, 0)
        self.assertEqual(inner_train_positive_weight(plan), 2.0)
        samples = LazyNCCTSamples(contract, FakeNCCTRepository(contract.patients),
                                  lower=-20, upper=120)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor([0.0]))

            def forward(self, image):
                return self.bias.expand(image.shape[0])

            def architecture_record(self):
                return {"synthetic_fixture": "bias-only"}

        with patch.object(ncct_training, "CompactResNet10",
                          side_effect=lambda width, pooling, stage4_stride: TinyModel()):
            result = run_one_fold(plan, samples, private_output_dir=self.base,
                                  run_id="weighted_report", config=TrainingConfig(
                                      max_epochs=1, positive_weighting=True,
                                  ))
        epoch = result.epochs[0]
        self.assertEqual(epoch.training_loss_kind, "positive_weighted_bce")
        self.assertEqual(result.positive_weight, 2.0)
        state = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
        bias = state["bias"]
        expected_validation = np.mean([
            float(torch.nn.functional.binary_cross_entropy_with_logits(
                bias, torch.tensor([float(p.outcome == "poor")]),
            ))
            for p in plan.inner_validation
        ])
        self.assertAlmostEqual(epoch.validation_bce, expected_validation, places=6)
        logged = json.loads(result.metrics_path.read_text())
        self.assertEqual(logged["epochs"][0]["training_loss_kind"], "positive_weighted_bce")
        self.assertNotIn("train_bce", logged["epochs"][0])

    def test_loader_exception_does_not_expose_private_details(self):
        plan = plan_outer_fold(self.contract, 0)

        class FailingSamples:
            def load(self, patient):
                raise RuntimeError(f"{patient.patient_id} /secret/patient/path")

        with self.assertRaises(RuntimeError) as caught:
            run_one_fold(plan, FailingSamples(), private_output_dir=self.base,
                         run_id="safe_error", config=TrainingConfig(max_epochs=1))
        self.assertEqual(str(caught.exception), "NCCT sample loading failed")
        self.assertIsNone(caught.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
