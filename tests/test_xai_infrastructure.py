"""Fictional private-store/guard fixtures; no UiS connections or real data."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ele630.xai_schema import METRIC_NAMES, aggregate_cases

HAS_TORCH = importlib.util.find_spec("torch") is not None


def fictional_case(value=1.):
    return {"values": {k: float(value) for k in METRIC_NAMES}, "reasons": dict.fromkeys(METRIC_NAMES),
            "audit": {"feature_shape": [2, 2, 2], "native_shape": [8, 32, 32],
                      "spacing_mm_zyx": [3., 1., 1.], "mask_reason": None, "interventions": {},
                      "frame_count": 1, "blur_sigma_voxels_zyx": [2., 2., 2.], "blur_radius_voxels": 6,
                      "blur_sigma_mm_zyx": [6., 2., 2.], "patch_shape_zyx": [4, 16, 16]}}


def fictional_public():
    from ele630.clinical_runner import _APPROVED_VERSIONS
    return {"schema": 1, "status": "XAI_STAGE7C_AGGREGATE_PASS", "mode": "ctp", "source_revision": "a"*40,
            "protocol_sha256": "b"*64, "validation_count": 23, "patient_data_access": True,
            "outer_test_evaluated": False, "between_seed_stability_assessed": False,
            "summary": aggregate_cases([fictional_case(i*.1) for i in range(23)]),
            "dependencies": {"python": "3.10.11", "numpy": "1.26.4", "torch": "2.3.1+cu118",
                             "nibabel": "5.4.2", "clinical": _APPROVED_VERSIONS}}


def fictional_historical(modality="ncct"):
    from test_fusion import fake_result
    from ele630 import fusion_runtime as old
    from ele630.xai_runtime import BASE_REVISION
    baseline = fake_result()
    baseline.update(training_count=92, validation_count=23)
    for record in baseline["metrics"].values():
        record.update(validation_count=23, validation_good_count=16, validation_poor_count=7)
    references = {"clinical_only": (.2313358984, .0663344280, .9821428571, .9285714286),
                  "fusion": (.2452544303, .0745231080, .9732142857, .7857142857),
                  "imaging_only": (.6143876931, .2116712247, .6160714286, .5)}
    if modality == "ctp":
        references.update(fusion=(.1720028729, .0482148614, 1., .8571428571),
                          imaging_only=(.6143472110, .2116544701, .6785714286, .5))
    for name, numbers in references.items():
        for key, value in zip(("bce", "brier_score", "auroc", "balanced_accuracy_0_5"), numbers):
            baseline["metrics"][name][key] = value
    baseline["metrics"]["no_image"] = copy.deepcopy(baseline["metrics"]["clinical_only"])
    for name, differences in baseline["fusion_minus_reference"].items():
        for key in differences:
            differences[key] = baseline["metrics"]["fusion"][key]-baseline["metrics"][name][key]
    old.validate_result(baseline)
    provenance, membership = {"checkpoint_sha256": "c"*64, "selected_epoch": 7}, {"fictional_signature": "d"*64}
    record = {"status": "PASS", "mode": modality, "identity": {"source_revision": BASE_REVISION},
              "outer_test_evaluated": False, "patient_data_access": True,
              "provenance": {**provenance, "membership": membership}, "results": baseline}
    return {"metrics": copy.deepcopy(baseline["metrics"])}, record, provenance, membership


class InfrastructureTests(unittest.TestCase):
    def test_source_identity_binds_protocol_and_rejects_original_source_changes(self):
        from ele630 import xai_runtime as runtime
        required = {runtime.PROTOCOL, runtime.PLAN, "src/ele630/xai_runner.py", "src/ele630/xai_spatial.py",
                    "src/ele630/xai_clinical_worker.py", "tests/test_xai_foundation.py", "scripts/slurm/submit_stage7.sh"}
        changed = [False]
        def git(*args):
            if args[0] == "merge-base":
                return b""
            if args[0] == "ls-files":
                return "\n".join(required).encode()
            if args[0] == "ls-tree":
                return b"src/ele630/resnet3d.py\n"
            if args[0] == "show":
                return b"changed" if changed[0] and args[1].startswith("a"*40) else b"original"
            raise AssertionError("unexpected source query")
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project/"docs").mkdir()
            (project/runtime.PROTOCOL).write_text("fictional protocol")
            (project/runtime.PLAN).write_text("fictional plan")
            with patch.object(runtime.original, "source_identity", return_value={"source_revision": "a"*40}), \
                 patch.object(runtime.original, "git", side_effect=git), patch.object(runtime.original, "PROJECT", project):
                identity = runtime.source_identity("a"*40)
                self.assertEqual(len(identity["protocol_sha256"]), 64)
                self.assertEqual(identity["baseline_revision"], runtime.BASE_REVISION)
                changed[0] = True
                with self.assertRaisesRegex(ValueError, "original source contract changed"):
                    runtime.source_identity("a"*40)

    def test_real_flag_guard_precedes_server_access(self):
        from ele630 import xai_runner as runner
        with patch.object(runner.runtime, "source_identity") as source:
            for mode in ("ncct", "ctp"):
                with self.assertRaises(ValueError):
                    runner.execute(mode, "a"*40, f"stage7_{mode}_fictional")
            source.assert_not_called()

    def test_protocol_approval_and_mode_bound_run_id(self):
        from ele630 import xai_runtime as runtime
        for flag, approval in ((False, "b"*64), (True, "c"*64)):
            with self.assertRaises(ValueError):
                runtime.approved_protocol({"protocol_sha256": "b"*64}, approval, flag)
        runtime.approved_protocol({"protocol_sha256": "b"*64}, "b"*64, True)
        with self.assertRaises(ValueError):
            runtime.run_path("stage7_ctp_fictional", "ncct")
        with self.assertRaises(ValueError):
            runtime.run_path("../../fictional", "ctp")

    def test_historical_gate_all_references_and_frozen_tolerance(self):
        from ele630.xai_runtime import historical_gate
        for mode in ("ncct", "ctp"):
            result, historical, provenance, membership = fictional_historical(mode)
            historical_gate(result, mode, historical, provenance=provenance, membership=membership)
            for name in ("fusion", "imaging_only", "clinical_only"):
                for key in ("bce", "brier_score", "mean_predicted_probability", "auroc", "balanced_accuracy_0_5"):
                    wrong = copy.deepcopy(result)
                    wrong["metrics"][name][key] += 1e-5
                    with self.assertRaises(ValueError):
                        historical_gate(wrong, mode, historical, provenance=provenance, membership=membership)
            wrong = copy.deepcopy(historical)
            wrong["provenance"]["checkpoint_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                historical_gate(result, mode, wrong, provenance=provenance, membership=membership)
            wrong = copy.deepcopy(result)
            wrong["metrics"]["no_image"]["mean_predicted_probability"] += 2e-8
            with self.assertRaises(ValueError):
                historical_gate(wrong, mode, historical, provenance=provenance, membership=membership)

    def test_public_schema_rejects_private_unknown_small_and_nonfinite(self):
        from ele630.xai_runtime import validate_public_record
        record = fictional_public()
        validate_public_record(record)
        self.assertIsNone(record["summary"]["decomposition.intercept"]["mean"])
        wrong = copy.deepcopy(record)
        wrong["summary"]["decomposition.intercept"]["mean"] = .1
        with self.assertRaises(ValueError):
            validate_public_record(wrong)
        for field in ("patient_id", "embedding", "coefficients", "image", "probabilities", "checkpoint"):
            wrong = copy.deepcopy(record)
            wrong[field] = "fictional-sensitive"
            with self.assertRaises(ValueError):
                validate_public_record(wrong)
        for value in (float("nan"), float("inf"), [1., 2.]):
            wrong = copy.deepcopy(record)
            wrong["summary"][METRIC_NAMES[0]]["mean"] = value
            with self.assertRaises(ValueError):
                validate_public_record(wrong)
        small = aggregate_cases([fictional_case(i) for i in range(3)])
        self.assertIsNone(small[METRIC_NAMES[0]]["mean"])
        small[METRIC_NAMES[0]]["mean"] = 1.
        from ele630.xai_schema import validate_aggregate
        with self.assertRaises(ValueError):
            validate_aggregate(small, 3)

    def test_aggregation_patient_unit_missingness_pairing_and_determinism(self):
        first = [fictional_case(i) for i in range(6)]
        key = METRIC_NAMES[0]
        first[0]["values"][key] = None
        first[0]["reasons"][key] = "missing_mask"
        a, b = aggregate_cases(first), aggregate_cases(first)
        self.assertEqual(a, b)
        self.assertEqual(a[key]["n_valid"], 5)
        self.assertEqual(a[key]["missing"]["missing_mask"], 1)
        self.assertEqual(a[key]["mean"], 3.)
        self.assertTrue(a[key]["ci_low"] <= 3. <= a[key]["ci_high"])

    def test_validation_guard_before_images_and_masks(self):
        from ele630.xai_data import ValidationAccess
        from ele630.fusion_synthetic import fictional_plan
        plan, loader = fictional_plan(), Mock()
        access = ValidationAccess(plan, loader)
        for p in (plan.outer_test[0], plan.inner_train[0], replace(plan.inner_validation[0], cohort="forged")):
            with self.assertRaises(ValueError):
                access.load(p)
            with self.assertRaises(ValueError):
                access.labels(p, (2, 2, 2), np.eye(4))
        loader.assert_not_called()
        labels, reason = access.labels(plan.inner_validation[0], (2, 2, 2), np.eye(4))
        self.assertIsNone(labels)
        self.assertEqual(reason, "missing_mask")

    def test_private_ipc_errors_suppressed_and_gpu_environment_removed(self):
        from ele630 import xai_runtime as runtime
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=1, stdout="fictional-sensitive", stderr="fictional-sensitive")) as call:
            with self.assertRaisesRegex(ValueError, "private IPC failure") as error:
                runtime.bridge({"probe": True})
            self.assertNotIn("fictional-sensitive", str(error.exception))
            self.assertEqual(call.call_args.kwargs["env"]["PYTHONPATH"], str(runtime.original.PROJECT/"src"))

    def test_preflight_sanitizes_failure_and_has_no_image_access(self):
        from ele630 import xai_runner as runner
        with patch.object(runner.runtime, "source_identity", side_effect=ValueError("fictional-sensitive")):
            with self.assertRaisesRegex(runner.XAIFailure, "phase=source") as error:
                runner.preflight("a"*40)
            self.assertNotIn("fictional-sensitive", str(error.exception))

    def test_completed_case_resume_integrity_exclusive_and_pending_refusal(self):
        from ele630 import ncct_private_output as private
        from ele630.xai_store import write_case, read_case
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root/"fictional"
            directory.mkdir(mode=0o700)
            with patch.object(private, "_APPROVED_PRIVATE_ROOT", root), patch.object(private, "_TRUSTED_CANONICAL_ROOT", root):
                record = fictional_case()
                write_case(directory, 0, "a"*64, record)
                self.assertEqual(read_case(directory, 0, "a"*64), record)
                with self.assertRaises(ValueError):
                    write_case(directory, 0, "a"*64, record)
                with self.assertRaises(ValueError):
                    read_case(directory, 0, "b"*64)
                self.assertIsNone(read_case(directory, 1, "a"*64))
                (directory/"case_001.pending").write_text("fictional incomplete")
                with self.assertRaises(ValueError):
                    read_case(directory, 1, "a"*64)
                path = directory/"case_000.json"
                damaged = json.loads(path.read_text())
                damaged["record"]["values"][METRIC_NAMES[0]] = 99.
                path.write_text(json.dumps(damaged))
                with self.assertRaises(ValueError):
                    read_case(directory, 0, "a"*64)


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable in this interpreter")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_complete_ncct_pipeline_missingness_and_matched_differences(self):
        import torch
        from ele630.resnet3d import CompactResNet10
        from ele630.fusion_features import freeze
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        from ele630.xai_analysis import analyse_case
        from ele630.xai_store import validate_case
        torch.manual_seed(630)
        model = freeze(CompactResNet10())
        x = torch.rand(1, 1, 8, 32, 32)
        state = torch_state(fictional_state(), "cpu")
        labels = np.ones((32, 32, 8), dtype=np.uint8)
        labels[:5] = 0
        labels[16:24, 16:24, 4:] = 2
        labels[20:24, 20:24, 4:] = 3
        result = analyse_case(model, x, torch.zeros(1, 11), state, "ncct", np.eye(4), labels_xyz=labels)
        validate_case(result)
        self.assertEqual(result["audit"]["blur_sigma_mm_zyx"], [2., 2., 2.])
        self.assertEqual(result["reasons"]["temporal.frame_00"], "not_applicable")
        for key in ("fidelity.target_drop", "roi.lesion_union.mass"):
            a, b = (result["values"][f"{m}.{key}"] for m in ("gradcam", "hirescam"))
            expected = None if a is None or b is None else b-a
            self.assertEqual(result["values"][f"paired.hirescam_minus_gradcam.{key}"], expected)
        missing = analyse_case(model, x, torch.zeros(1, 11), state, "ncct", np.eye(4))
        self.assertIsNone(missing["values"]["gradcam.roi.core.mass"])
        self.assertEqual(missing["reasons"]["gradcam.roi.core.mass"], "missing_mask")

    def test_ctp_analysis_includes_all40_profiles_without_mask_substitution(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_synthetic import FictionalCTP
        from ele630.fusion_features import freeze
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        from ele630.xai_analysis import analyse_case
        from ele630.xai_store import validate_case
        torch.manual_seed(630)
        model = freeze(CompactCTPMean())
        result = analyse_case(model, FictionalCTP(630), torch.zeros(1, 11), torch_state(fictional_state(), "cpu"),
                              "ctp", np.eye(4))
        validate_case(result)
        self.assertTrue(all(result["values"][f"temporal.frame_{i:02d}"] is not None for i in range(40)))
        self.assertEqual(result["reasons"]["gradcam.fidelity.target_drop"], "missing_mask")
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_nonfinite_map_is_counted_without_noise_normalization_or_intervention(self):
        import torch
        from ele630.resnet3d import CompactResNet10
        from ele630.fusion_features import freeze
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        from ele630 import xai_analysis as analysis
        model = freeze(CompactResNet10())
        maps = {m: {"positive": np.full((2, 2, 2), np.nan)} for m in ("gradcam", "hirescam")}
        labels = np.ones((32, 32, 8), dtype=np.uint8)
        with patch.object(analysis, "ncct_cam", return_value={"logit": 1., "sign": 1, "maps": maps}):
            result = analysis.analyse_case(model, torch.rand(1, 1, 8, 32, 32), torch.zeros(1, 11),
                                           torch_state(fictional_state(), "cpu"), "ncct", np.eye(4), labels_xyz=labels)
        self.assertEqual(result["values"]["gradcam.map_nonfinite"], 1.)
        self.assertIsNone(result["values"]["gradcam.fidelity.target_drop"])
        self.assertEqual(result["reasons"]["gradcam.fidelity.target_drop"], "nonfinite_map")

    def test_runner_historical_failure_precedes_cam_or_torch_patient_gate(self):
        import torch
        from contextlib import nullcontext
        from types import SimpleNamespace
        from ele630 import xai_runner as runner
        from ele630.fusion_synthetic import fictional_plan
        plan = fictional_plan()
        from ele630.resnet3d import CompactResNet10
        model = CompactResNet10()
        with patch.dict(os.environ, {}), tempfile.TemporaryDirectory() as temporary:
            paths = SimpleNamespace(fold_dir=Path(temporary))
            with patch.object(runner.runtime, "source_identity", return_value={"protocol_sha256": "b"*64}), \
                 patch.object(runner.original, "environment", return_value={}), \
                 patch.object(runner.runtime, "bridge", return_value={"dependencies": {}, "metrics": {}}), \
                 patch.object(runner, "verify_smoke"), \
                 patch.object(runner.runtime, "run_path", return_value=paths), \
                 patch("ele630.xai_store.run_lock", return_value=nullcontext()), \
                 patch("ele630.fusion_runner.load_plan", return_value=(Mock(), plan)), \
                 patch("ele630.fusion_runner.check_clinical_path"), \
                 patch("ele630.xai_masks.real_mask_preflight", return_value={"path_contract_verified": True}), \
                 patch("ele630.xai_resources.ResourceProfile", return_value=Mock(attach=lambda m: nullcontext(), phase=lambda p: nullcontext())), \
                 patch("ele630.fusion_checkpoints.load_ncct", return_value=(model, {})), \
                 patch("ele630.fusion_runner.real_features", return_value=(np.zeros((18, 128)), np.zeros((6, 128)), np.ones(6)*.3)), \
                 patch.object(runner.original, "read_private_json", return_value={}), \
                 patch.object(runner, "_check_torch_equivalence") as equivalent, \
                 patch("ele630.xai_analysis.analyse_case") as analyse:
                with self.assertRaisesRegex(runner.XAIFailure, "phase=strict_historical_gate"):
                    runner.execute("ncct", "a"*40, "stage7_ncct_fictional", execute_real=True,
                                   protocol_approval="b"*64, smoke_run="stage7_smoke_fictional")
                equivalent.assert_not_called()
                analyse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
