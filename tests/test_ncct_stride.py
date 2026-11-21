"""NCCT stride, semantic checkpoints and guarded development: synthetic only."""

from __future__ import annotations

import copy
import hashlib
import io
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))

from ele630.ncct_model_config import NCCTModelConfig, record_signature

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from ele630.ncct_checkpoint import (
        NCCTCheckpointError, load_ncct_checkpoint, manifest_path, save_ncct_checkpoint,
    )
    from ele630.ncct_development import DevelopmentNCCTSamples, evaluate_selected_development
    from ele630.ncct_samples import LazyNCCTSamples
    from ele630.ncct_training import TrainingConfig, run_one_fold
    from ele630.ncct_splits import plan_outer_fold
    from ele630.resnet3d import CompactResNet10
    from ele630 import ncct_private_output
    from test_ncct_stage4a import fictional_contract, FakeNCCTRepository


class ModelConfigTests(unittest.TestCase):
    def test_defaults_and_distinct_semantic_signatures(self):
        original = NCCTModelConfig()
        self.assertEqual(original.stage4_stride, (2, 2, 2))
        self.assertEqual(original.as_record()["parameter_count"], 897489)
        self.assertNotEqual(record_signature(original.as_record()),
                            record_signature(NCCTModelConfig(stage4_stride=(1, 1, 1)).as_record()))
        self.assertNotEqual(record_signature(original.as_record()),
                            record_signature(NCCTModelConfig(pooling="avgmax").as_record()))

    def test_rejects_ambiguous_or_unsupported_stride(self):
        for stride in (None, 1, [1, 1, 1], (1, 1), (True, True, True),
                       (1.0, 1.0, 1.0), (1, 2, 2), (0, 0, 0), (3, 3, 3)):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                NCCTModelConfig(stage4_stride=stride)

    def test_geometry_with_ceiling_and_nominal_fields(self):
        for shape in ((1, 1, 1), (9, 17, 15), (59, 603, 512)):
            for stride in (1, 2):
                config = NCCTModelConfig(stage4_stride=(stride,) * 3)
                result = config.feature_geometry(shape)
                jump = (2 * stride, 8 * stride, 8 * stride)
                self.assertEqual(result["shape_dhw"], [(n + j - 1) // j for n, j in zip(shape, jump)])
        self.assertEqual(NCCTModelConfig().feature_geometry((59, 603, 512))["nominal_receptive_field_dhw"], [29, 95, 95])
        self.assertEqual(NCCTModelConfig(stage4_stride=(1, 1, 1)).feature_geometry((59, 603, 512))["nominal_receptive_field_dhw"], [25, 79, 79])

    def test_bad_geometry_and_nonfinite_signature(self):
        for shape in ((0, 1, 1), (True, 1, 1), (1, 2), [1, 2, 3], (1.0, 2, 3)):
            with self.assertRaises(ValueError):
                NCCTModelConfig().feature_geometry(shape)
        with self.assertRaises(ValueError):
            record_signature({"bad": float("nan")})


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: not a runtime pass")
class StrideArchitectureTests(unittest.TestCase):
    def test_original_default_matches_explicit_stride_and_initialized_weights(self):
        torch.manual_seed(630)
        default = CompactResNet10().eval()
        torch.manual_seed(630)
        explicit = CompactResNet10(stage4_stride=(2, 2, 2)).eval()
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        inputs = torch.rand(1, 1, 9, 17, 15)
        with torch.no_grad():
            torch.testing.assert_close(default(inputs), explicit(inputs), rtol=0, atol=0)

    def test_only_matching_terminal_strides_change_with_identical_initial_parameters(self):
        torch.manual_seed(630)
        original = CompactResNet10()
        torch.manual_seed(630)
        challenger = CompactResNet10(stage4_stride=(1, 1, 1))
        self.assertEqual(tuple(original.state_dict()), tuple(challenger.state_dict()))
        self.assertEqual(sum(p.numel() for p in challenger.parameters()), 897489)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, challenger.state_dict()[name], rtol=0, atol=0)
        for name, module in original.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                candidate = dict(challenger.named_modules())[name]
                self.assertEqual(module.kernel_size, candidate.kernel_size)
                self.assertEqual(module.padding, candidate.padding)
                self.assertEqual(module.stride != candidate.stride,
                                 name in {"stage4.conv1", "stage4.shortcut.0"})

    def test_variable_shapes_pooling_and_backward_both_strides(self):
        for stride in ((1, 1, 1), (2, 2, 2)):
            for pooling in ("avg", "avgmax"):
                model = CompactResNet10(width=4, pooling=pooling, stage4_stride=stride)
                for shape in ((1, 1, 1), (9, 17, 15), (8, 24, 20)):
                    shapes = []
                    hook = model.stage4.register_forward_hook(lambda _m, _i, out: shapes.append(tuple(out.shape[2:])))
                    try:
                        logits = model(torch.rand(1, 1, *shape))
                        expected = NCCTModelConfig(width=4, pooling=pooling, stage4_stride=stride).feature_geometry(shape)
                        self.assertEqual(shapes, [tuple(expected["shape_dhw"])])
                        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1)).backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                        self.assertEqual(model.forward_features(torch.rand(2, 1, *shape)).shape, (2, 32))
                    finally:
                        hook.remove()
                        model.zero_grad(set_to_none=True)

    def test_architecture_record_rejects_mutated_strides_and_pooling(self):
        model = CompactResNet10()
        model.stage4.shortcut[0].stride = (1, 1, 1)
        with self.assertRaises(ValueError):
            model.architecture_record()
        model = CompactResNet10()
        model.pool = torch.nn.AdaptiveMaxPool3d(1)
        with self.assertRaises(ValueError):
            model.architecture_record()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: not a runtime pass")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.pt"
        self.model = CompactResNet10(width=4)
        self.record = {"architecture": self.model.architecture_record(),
                       "input": {"lower": -20, "upper": 120}, "seed": 630,
                       "membership_signature": "synthetic-only"}
        save_ncct_checkpoint(self.model, self.path, run_record=self.record, epoch=1)

    def load(self, model=None, record=None, epoch=1):
        return load_ncct_checkpoint(model or self.model, self.path,
                                    expected_run_record=record or self.record, expected_epoch=epoch)

    def test_roundtrip_and_owner_only_artifacts(self):
        restored = CompactResNet10(width=4)
        with patch("torch.load", wraps=torch.load) as deserialize:
            self.load(restored)
        self.assertTrue(deserialize.call_args.kwargs["weights_only"])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(manifest_path(self.path).stat().st_mode & 0o777, 0o600)

    def test_rejects_stride_and_pooling_even_with_compatible_weight_keys(self):
        for model in (CompactResNet10(width=4, stage4_stride=(1, 1, 1)),
                      CompactResNet10(width=4, pooling="avgmax")):
            with self.assertRaises(NCCTCheckpointError), patch("torch.load") as deserialize:
                self.load(model)
            deserialize.assert_not_called()

    def test_rejects_input_membership_epoch_and_seed_changes_before_deserialization(self):
        for key, value in (("input", {"lower": 0, "upper": 80}),
                           ("seed", 631), ("membership_signature", "other")):
            changed = copy.deepcopy(self.record)
            changed[key] = value
            with self.assertRaises(NCCTCheckpointError), patch("torch.load") as deserialize:
                self.load(record=changed)
            deserialize.assert_not_called()
        with self.assertRaises(NCCTCheckpointError):
            self.load(epoch=2)

    def test_rejects_modified_bytes_before_deserialization(self):
        with self.path.open("ab") as stream:
            stream.write(b"altered synthetic bytes")
        with self.assertRaises(NCCTCheckpointError), patch("torch.load") as deserialize:
            self.load()
        deserialize.assert_not_called()

    def test_rejects_missing_or_malformed_manifest_and_legacy_without_manifest(self):
        sidecar = manifest_path(self.path)
        sidecar.write_text("{}", encoding="utf-8")
        with self.assertRaises(NCCTCheckpointError):
            self.load()
        sidecar.unlink()  # Synthetic fixture only.
        with self.assertRaises(NCCTCheckpointError):
            self.load()

    def test_rejects_nonfinite_weights_before_model_mutation(self):
        state = self.model.state_dict()
        state["classifier.weight"] = torch.full_like(state["classifier.weight"], float("nan"))
        torch.save(state, self.path)
        sidecar = manifest_path(self.path)
        saved = json.loads(sidecar.read_text())
        saved["sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(saved))
        snapshot = {k: v.clone() for k, v in self.model.state_dict().items()}
        with self.assertRaises(NCCTCheckpointError):
            self.load()
        for name, value in snapshot.items():
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)

    def test_rejects_weight_keys_shape_and_dtype_before_loading_state(self):
        for kind in ("keys", "shape", "dtype"):
            state = self.model.state_dict()
            if kind == "keys":
                state.pop("classifier.bias")
            elif kind == "shape":
                state["classifier.weight"] = torch.zeros(1, 1)
            else:
                state["classifier.weight"] = state["classifier.weight"].double()
            torch.save(state, self.path)
            sidecar = manifest_path(self.path)
            saved = json.loads(sidecar.read_text())
            saved["sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
            sidecar.write_text(json.dumps(saved))
            with self.assertRaises(NCCTCheckpointError), patch.object(self.model, "load_state_dict") as update:
                self.load()
            update.assert_not_called()

    def test_exclusive_publication_does_not_replace_existing_checkpoint(self):
        before = self.path.read_bytes()
        with self.assertRaises(NCCTCheckpointError):
            save_ncct_checkpoint(self.model, self.path, run_record=self.record, epoch=1)
        self.assertEqual(before, self.path.read_bytes())

    def test_manifest_signature_and_filename_tampering_rejected(self):
        sidecar = manifest_path(self.path)
        saved = json.loads(sidecar.read_text())
        for key, value in (("checkpoint_name", "different.pt"),
                           ("run_signature", "0" * 64), ("schema_version", True)):
            changed = dict(saved)
            changed[key] = value
            sidecar.write_text(json.dumps(changed))
            with self.assertRaises(NCCTCheckpointError):
                self.load()
        sidecar.write_text(json.dumps(saved))

    def test_symlink_leaf_is_rejected(self):
        alias = self.path.parent / "alias.pt"
        try:
            alias.symlink_to(self.path)
        except OSError:
            self.skipTest("symlink privileges unavailable")
        with self.assertRaises(NCCTCheckpointError):
            load_ncct_checkpoint(self.model, alias, expected_run_record=self.record, expected_epoch=1)

    def test_nonprivate_posix_mode_rejected(self):
        if os.name != "posix":
            self.skipTest("POSIX ownership/modes need UiS")
        os.chmod(self.path, 0o644)
        with self.assertRaises(NCCTCheckpointError):
            self.load()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: not a runtime pass")
class StrideDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "approved"
        self.base = self.root / "training"
        self.base.mkdir(parents=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.base, 0o700)
        for name in ("_APPROVED_PRIVATE_ROOT", "_TRUSTED_CANONICAL_ROOT"):
            mock = patch.object(ncct_private_output, name, self.root)
            mock.start()
            self.addCleanup(mock.stop)
        self.contract = fictional_contract(per_fold=2)
        self.plan = plan_outer_fold(self.contract, 0)
        self.repo = FakeNCCTRepository(self.contract.patients)
        self.samples = LazyNCCTSamples(self.contract, self.repo, lower=-20, upper=120)
        self.guarded = DevelopmentNCCTSamples(self.plan, self.samples)

    def test_outer_and_forged_patient_rejected_before_loading(self):
        for patient in (self.plan.outer_test[0], replace(self.plan.inner_train[0], cohort="forged")):
            with self.assertRaises(ValueError):
                self.guarded.load(patient)
        self.assertEqual(self.repo.calls, [])

    def test_changed_input_is_rejected_across_paired_loads(self):
        import numpy as np
        from ele630.ncct_samples import NCCTSample
        patient = self.plan.inner_train[0]
        first = self.guarded.load(patient)
        signature = self.guarded.input_consistency_signature
        self.guarded.load(patient)
        self.assertEqual(signature, self.guarded.input_consistency_signature)
        changed = NCCTSample(image=np.zeros_like(first.image), target=first.target)
        with patch.object(self.samples, "load", return_value=changed):
            with self.assertRaisesRegex(ValueError, "input changed"):
                self.guarded.load(patient)

    def test_training_diagnostics_provenance_and_selected_evaluation_both_strides(self):
        results = []
        for stride in ((2, 2, 2), (1, 1, 1)):
            result = run_one_fold(
                self.plan, self.guarded, private_output_dir=self.base,
                run_id=f"synthetic_stride_{stride[0]}",
                config=TrainingConfig(max_epochs=3, diagnostic_epoch=3, stage4_stride=stride),
                source_revision="a" * 40, protocol_sha256="b" * 64,
                require_complete_provenance=True,
            )
            results.append(result)
            self.assertEqual(result.diagnostics[0]["count"], len(self.plan.inner_train))
            self.assertEqual(result.diagnostics[0]["epoch"], 3)
            self.assertEqual(result.run_record["architecture"]["stage4_stride"], list(stride))
            self.assertEqual(result.run_record["input"]["lower"], -20)
            self.assertTrue(manifest_path(result.checkpoint_path).is_file())
            metrics = evaluate_selected_development(self.plan, self.guarded, result)
            self.assertFalse(metrics["outer_test_evaluated"])
            self.assertAlmostEqual(metrics["model"]["bce"], result.best_validation_bce, places=7)
            self.assertGreaterEqual(metrics["logit_sd"], 0)
            serialized = json.dumps(result.run_record)
            self.assertTrue(all(p.patient_id not in serialized for p in self.contract.patients))
        self.assertEqual(results[0].run_record["initial_parameters_sha256"],
                         results[1].run_record["initial_parameters_sha256"])
        self.assertFalse({p.patient_id for p in self.plan.outer_test} & set(self.repo.calls))

    def test_provenance_validation_precedes_output_and_data_access(self):
        with self.assertRaises(ValueError):
            run_one_fold(self.plan, self.guarded, private_output_dir=self.base, run_id="bad",
                         require_complete_provenance=True)
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertEqual(self.repo.calls, [])

    def test_initialization_mismatch_stops_before_optimizer_or_image_access(self):
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer constructed")):
            with self.assertRaisesRegex(ValueError, "initialization"):
                run_one_fold(self.plan, self.guarded, private_output_dir=self.base,
                             run_id="mismatch", expected_initial_parameters_sha256="0" * 64)
        self.assertEqual(self.repo.calls, [])

    def test_optional_diagnostic_does_not_change_optimizer_trajectory(self):
        results = [run_one_fold(
            self.plan, self.guarded, private_output_dir=self.base, run_id=f"diagnostic_{n}",
            config=TrainingConfig(max_epochs=1, diagnostic_epoch=epoch),
        ) for n, epoch in enumerate((None, 1))]
        self.assertEqual(results[0].best_validation_bce, results[1].best_validation_bce)
        states = [torch.load(r.checkpoint_path, weights_only=True) for r in results]
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_config_validation_and_original_diagnostics_default(self):
        self.assertIsNone(TrainingConfig().diagnostic_epoch)
        for value in (0, True, 11, 1.5):
            with self.assertRaises(ValueError):
                TrainingConfig(diagnostic_epoch=value)
        with self.assertRaises(ValueError):
            TrainingConfig(stage4_stride=(1, 2, 2))

    def test_changed_evaluation_plan_rejected_before_image_access(self):
        result = run_one_fold(self.plan, self.guarded, private_output_dir=self.base,
                              run_id="synthetic_eval", config=TrainingConfig(max_epochs=1))
        changed = replace(self.plan, seed=631)
        self.repo.calls.clear()
        with self.assertRaises(ValueError):
            evaluate_selected_development(changed, self.guarded, result)
        self.assertEqual(self.repo.calls, [])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable: not a runtime pass")
class SyntheticAndExecutionBoundaryTests(unittest.TestCase):
    @contextmanager
    def mocked_cuda_initialization(self, pilot):
        """Exercise initialization on a CPU-only host, including probe release."""
        events = []

        class Probe:
            def __del__(self):
                events.append("release")

        def allocate(*args, **kwargs):
            self.assertEqual(args, (1,))
            self.assertEqual(kwargs, {"dtype": torch.uint8, "device": torch.device("cuda:0")})
            events.append("allocate")
            return Probe()

        with ExitStack() as stack:
            stack.enter_context(patch.object(pilot, "_seed_everything", side_effect=lambda seed: events.append(("seed", seed))))
            stack.enter_context(patch.object(torch, "empty", side_effect=allocate))
            for name, event in (("synchronize", "synchronize"), ("empty_cache", "empty_cache"),
                                ("reset_peak_memory_stats", "reset")):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *args, event=event: events.append(event)))
            def properties(device):
                events.append("properties")
                return SimpleNamespace(name="synthetic GPU", total_memory=40 * 1024**3)
            stack.enter_context(patch.object(torch.cuda, "get_device_properties", side_effect=properties))
            yield events

    def test_cuda_probe_released_and_cache_cleared_before_peak_reset_and_model(self):
        import synthetic_ncct_gpu_memory as pilot
        output = io.StringIO()
        def model(**kwargs):
            events.append("model")
            raise RuntimeError("synthetic setup stop")
        with self.mocked_cuda_initialization(pilot) as events, patch.object(pilot, "CompactResNet10", side_effect=model), redirect_stdout(output):
            self.assertFalse(pilot._run_case(pilot.SHAPES[0], (2, 2, 2), torch.device("cuda:0"), 3, 630))
        self.assertEqual(events, [("seed", 630), "allocate", "release", "synchronize", "empty_cache",
                                  "reset", "properties", "model", "empty_cache"])
        record = json.loads(output.getvalue())
        self.assertEqual(record["error_stage"], "model_setup")

    def test_allocator_failure_reports_sanitized_message_and_nonzero_exit(self):
        import synthetic_ncct_gpu_memory as pilot
        output = io.StringIO()
        with self.mocked_cuda_initialization(pilot), patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch, "empty", side_effect=RuntimeError("Invalid device argument\nC:\\private\\trace.txt /private/trace.txt\x00")), \
                patch.object(torch.cuda, "reset_peak_memory_stats") as reset, \
                patch.object(pilot, "CompactResNet10") as model, redirect_stdout(output):
            self.assertEqual(pilot.main([]), 1)
        reset.assert_not_called()
        model.assert_not_called()
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["error_stage"], "cuda_allocator_initialization")
        self.assertEqual(records[0]["error_type"], "RuntimeError")
        self.assertEqual(records[0]["error_message"], "Invalid device argument <path> <path>")

    def test_peak_reset_failure_is_distinguished_from_allocator_initialization(self):
        import synthetic_ncct_gpu_memory as pilot
        output = io.StringIO()
        with self.mocked_cuda_initialization(pilot), \
                patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=RuntimeError("Invalid device argument")), \
                patch.object(pilot, "CompactResNet10") as model, redirect_stdout(output):
            self.assertFalse(pilot._run_case(pilot.SHAPES[0], (1, 1, 1), torch.device("cuda:0"), 3, 630))
        model.assert_not_called()
        record = json.loads(output.getvalue())
        self.assertEqual(record["error_stage"], "cuda_peak_reset")
        self.assertEqual(record["error_message"], "Invalid device argument")

    def test_cleanup_failure_does_not_mask_original_oom(self):
        import synthetic_ncct_gpu_memory as pilot
        output = io.StringIO()
        with self.mocked_cuda_initialization(pilot), \
                patch.object(torch.cuda, "empty_cache", side_effect=[None, RuntimeError("synthetic cleanup failure")]), \
                patch.object(pilot, "CompactResNet10", side_effect=MemoryError("synthetic OOM")), redirect_stdout(output):
            self.assertFalse(pilot._run_case(pilot.SHAPES[0], (2, 2, 2), torch.device("cuda:0"), 3, 630))
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "oom")
        self.assertEqual(record["error_stage"], "model_setup")
        self.assertEqual(record["error_message"], "synthetic OOM")
        self.assertEqual(record["cleanup_error_type"], "RuntimeError")
        self.assertEqual(record["cleanup_error_message"], "synthetic cleanup failure")

    def test_error_message_is_bounded_and_normalized(self):
        import synthetic_ncct_gpu_memory as pilot
        message = pilot._sanitized_error(RuntimeError("\x1b[31m\t" + "x" * 600))
        self.assertEqual(len(message), 500)
        self.assertFalse(any(ord(char) < 32 for char in message))

    def test_default_cuda_cases_preserve_shapes_strides_steps_and_seed(self):
        import synthetic_ncct_gpu_memory as pilot
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(pilot, "_run_case", return_value=True) as run:
            self.assertEqual(pilot.main([]), 0)
        self.assertEqual([call.args for call in run.call_args_list], [
            (shape, stride, torch.device("cuda:0"), 3, 630)
            for shape in ((8, 32, 32), (30, 512, 512), (59, 603, 512))
            for stride in ((2, 2, 2), (1, 1, 1))
        ])

    def test_cpu_full_step_profiles_both_strides_with_same_initial_parameters(self):
        import synthetic_ncct_gpu_memory as pilot
        output = io.StringIO()
        with patch.object(torch.cuda, "reset_peak_memory_stats") as reset, redirect_stdout(output):
            self.assertEqual(pilot.main(["--device", "cpu", "--steps", "2"]), 0)
        reset.assert_not_called()
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["stage4_stride"] for r in records], [[2, 2, 2], [1, 1, 1]])
        self.assertEqual(records[0]["initial_parameters_sha256"], records[1]["initial_parameters_sha256"])
        for record in records:
            self.assertEqual(record["status"], "pass")
            self.assertEqual(record["warmup_steps"], 1)
            self.assertEqual(record["measured_steps"], 2)
            self.assertIsNone(record["peak_allocated_mib"])
            self.assertIsNone(record["peak_reserved_mib"])
            self.assertGreater(record["seconds_per_full_step"], 0)
            self.assertGreater(record["seconds_per_validation_forward"], 0)

    def test_default_development_runner_is_plan_only_and_reads_no_files(self):
        import run_ncct_stride_comparison as runner
        output = io.StringIO()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), redirect_stdout(output):
            self.assertEqual(runner.main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertFalse(plan["patient_data_access"])
        self.assertFalse(plan["outer_test_evaluation"])
        self.assertEqual(len(plan["arms"]), 2)

    def test_execution_requires_slurm_and_explicit_provenance_before_file_reads(self):
        import run_ncct_stride_comparison as runner
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(["--execute-development"]), 2)

    def test_analytical_profiling_matches_parameter_and_map_storage_without_model(self):
        import synthetic_ncct_gpu_memory as pilot
        output = io.StringIO()
        with patch.object(pilot, "CompactResNet10", side_effect=AssertionError("model allocation")), redirect_stdout(output):
            self.assertEqual(pilot.main(["--analytical-only", "--shape", "59", "603", "512"]), 0)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["parameter_count"] for r in records], [897489, 897489])
        self.assertEqual(records[1]["one_stage4_feature_map_mib"], 8 * records[0]["one_stage4_feature_map_mib"])

    def test_cuda_missing_does_not_fall_back_to_cpu(self):
        import synthetic_ncct_gpu_memory as pilot
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(pilot, "_run_case") as run, redirect_stderr(io.StringIO()):
            self.assertEqual(pilot.main([]), 2)
        run.assert_not_called()

    def test_profiling_failure_never_retries_smaller_shape(self):
        import synthetic_ncct_gpu_memory as pilot
        with patch.object(pilot, "_run_case", return_value=False) as run:
            self.assertEqual(pilot.main(["--device", "cpu", "--shape", "9", "48", "48"]), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], (9, 48, 48))


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        torch.set_num_threads(1)
    unittest.main()
