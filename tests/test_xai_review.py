"""Targeted independent-review regressions; all inputs are fictional."""
from __future__ import annotations

import importlib.util
import time
import tracemalloc
import hashlib
import unittest
import gzip
import json
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


def original_patches(brain):
    result = []
    for z in range(0, brain.shape[0], 4):
        for y in range(0, brain.shape[1], 16):
            for x in range(0, brain.shape[2], 16):
                mask = np.zeros_like(brain)
                mask[z:z+4, y:y+16, x:x+16] = True
                indices = np.flatnonzero(mask & brain)
                if indices.size:
                    result.append(indices)
    return result


class PatchReviewTests(unittest.TestCase):
    def test_exact_patch_indices_masks_ties_and_controls_match_original(self):
        from ele630 import xai_perturbation as module
        rng = np.random.default_rng(630)
        for shape in ((1, 1, 1), (5, 17, 19), (9, 34, 35)):
            brain = rng.random(shape) > .45
            brain.flat[0] = True
            for b in (brain, brain[:, ::-1]):
                for a, old in zip(module.patches(b), original_patches(b), strict=True):
                    np.testing.assert_array_equal(a, old)
                for h in (np.ones(shape), rng.integers(0, 4, shape).astype(float)):
                    new = module.intervention_masks(h, b)
                    with patch.object(module, "patches", side_effect=original_patches):
                        old = module.intervention_masks(h, b)
                    for first, second in zip([new[0]]+new[1], [old[0]]+old[1]):
                        np.testing.assert_array_equal(first, second)

    def test_native_patch_enumeration_allocation_and_time_bound(self):
        from ele630.xai_perturbation import patches
        brain = np.ones((72, 512, 512), dtype=bool)
        brain[:, ::3, ::3] = False
        tracemalloc.start()
        start = time.perf_counter()
        try:
            with patch("ele630.xai_perturbation.np.zeros_like", side_effect=AssertionError("full-volume patch allocation")):
                regions = patches(brain)
            elapsed = time.perf_counter()-start
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        retained = sum(r.nbytes for r in regions)
        self.assertEqual(sum(len(r) for r in regions), brain.sum())
        self.assertLess(peak-retained, 4*1024*1024)
        self.assertLess(elapsed, 15., "native synthetic patch enumeration exceeded regression ceiling")

    def test_mask_score_cache_preserves_results_and_has_fixed_storage(self):
        from ele630.xai_perturbation import MemoizedMaskScore
        calls = []
        def score(mask):
            calls.append(mask.sum())
            return float(mask.sum()), int(mask.sum())
        cache = MemoizedMaskScore(score)
        mask = np.zeros((3, 4, 5), dtype=bool)
        for i in range(13):
            mask.flat[i] = True
            self.assertEqual(cache(mask), score(mask))
            self.assertEqual(cache(mask.copy()), (float(i+1), i+1))
        self.assertEqual(cache.hits, 13)
        self.assertEqual(len(cache.cache), 13)
        mask.flat[13] = True
        with self.assertRaises(ValueError):
            cache(mask)


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable")
class ResourceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_counter_includes_copied_models_and_removes_original_hook(self):
        import torch
        import copy
        from ele630.resnet3d import CompactResNet10
        from ele630.fusion_features import freeze
        from ele630.xai_resources import ResourceProfile, validate_profile, ctp_workload
        model = freeze(CompactResNet10())
        profile = ResourceProfile()
        before = len(model.stem[0]._forward_pre_hooks)
        with profile.attach(model), profile.phase("case_total"), torch.no_grad():
            model.forward_features(torch.ones(1, 1, 4, 16, 16))
            other = copy.deepcopy(model)
            other.forward_features(torch.ones(1, 1, 4, 16, 16))
        record = validate_profile(profile.record())
        self.assertEqual(record["encoder_forwards"], 2)
        self.assertEqual(record["phases"]["case_total"]["encoder_forwards"], 2)
        self.assertEqual(len(model.stem[0]._forward_pre_hooks), before)
        self.assertEqual(ctp_workload()["optimized_total_forwards"], 23345)
        with self.assertRaises(ValueError):
            profile.call("fictional-patient-id", lambda: None)
        with self.assertRaises(ValueError):
            validate_profile({**record, "patient_id": "fictional"})

    def test_cached_ctp_temporal_and_head_cam_match_recomputation(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import freeze
        from ele630.fusion_synthetic import FictionalCTP
        from ele630.xai_fused_score import torch_state, fused_logit_torch
        from ele630.xai_synthetic import fictional_state
        from ele630.xai_spatial import ctp_cam
        from ele630.xai_temporal import temporal_occlusion
        from ele630.xai_resources import ResourceProfile
        torch.manual_seed(630)
        model = freeze(CompactCTPMean())
        state, c = torch_state(fictional_state(), "cpu"), torch.zeros(1, 11)
        series = FictionalCTP(630)
        maps = ctp_cam(model, series, c, state)
        with torch.no_grad():
            altered = {k: v.clone() for k, v in state.items()}
            altered["coef"][:8] *= -2
        fresh = ctp_cam(model, series, c, altered, sign=maps["sign"])
        profile = ResourceProfile()
        with profile.attach(model):
            cached = ctp_cam(model, series, c, altered, sign=maps["sign"], baseline_embedding=maps["embedding"])
        self.assertEqual(profile.forwards, 40)
        for method in ("gradcam", "hirescam"):
            np.testing.assert_array_equal(fresh["maps"][method]["signed"], cached["maps"][method]["signed"])
        score = lambda e: fused_logit_torch(e, c, state)
        support = maps["sign"]*maps["logit"]
        fresh_temporal = temporal_occlusion(model, series, score, support, sign=maps["sign"])
        profile = ResourceProfile()
        with profile.attach(model):
            cached_temporal = temporal_occlusion(model, series, score, support, sign=maps["sign"],
                                                 baseline_embeddings=maps["frame_embeddings"], baseline_mean=maps["embedding"])
        self.assertEqual(profile.forwards, 95)
        np.testing.assert_allclose(fresh_temporal["window_drops"], cached_temporal["window_drops"], atol=1e-6, rtol=1e-6)

    def test_optimized_case_matches_review_commit_and_reduces_encoder_calls(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import freeze
        from ele630.fusion_synthetic import FictionalCTP
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        from ele630.xai_analysis import analyse_case
        from ele630.xai_resources import ResourceProfile
        # Independent reviewed orchestration, with current unchanged numerical
        # primitives: catches accidental changes to targets, masks or outputs.
        reference = Path(__file__).with_name("fixtures") / "stage7_review_xai_analysis.py"
        source = reference.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         "d836c4f0460dca4a1b0892b923aa39cb76796bc24813fd36f576622bb8d0165f")
        namespace = {"__package__": "ele630", "__name__": "ele630._review_reference"}
        exec(compile(source, "review_commit_xai_analysis", "exec"), namespace)
        torch.manual_seed(630)
        model = freeze(CompactCTPMean())
        state = torch_state(fictional_state(), "cpu")
        labels = np.ones((32, 32, 8), dtype=np.uint8)
        labels[:4] = 0
        labels[16:24, 16:24, 4:] = 2
        labels[20:24, 20:24, 4:] = 3
        args = (model, FictionalCTP(630, (8, 32, 32)), torch.zeros(1, 11), state, "ctp", np.eye(4))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        legacy_profile, new_profile = ResourceProfile(), ResourceProfile()
        with legacy_profile.attach(model):
            legacy = namespace["analyse_case"](*args, labels_xyz=labels)
        with new_profile.attach(model):
            new = analyse_case(*args, labels_xyz=labels, resources=new_profile)
        self.assertEqual(legacy["reasons"], new["reasons"])
        self.assertEqual(legacy["audit"], new["audit"])
        for key, value in legacy["values"].items():
            if value is None:
                self.assertIsNone(new["values"][key])
            else:
                self.assertAlmostEqual(value, new["values"][key], delta=1e-6, msg=key)
        self.assertLessEqual(new_profile.forwards, 815)
        self.assertGreaterEqual(legacy_profile.forwards-new_profile.forwards, 120)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], atol=0, rtol=0)

    def test_bounded_small_cpu_calibration_cannot_satisfy_native_cuda_gate(self):
        from ele630.xai_resource_check import synthetic_resource_check
        from ele630.xai_resources import validate_profile
        from ele630 import xai_runtime as runtime
        record = synthetic_resource_check((4, 16, 16), device="cpu")
        self.assertEqual(record["io_backend"], "in_memory_fictional")
        self.assertEqual(validate_profile(record["profile"])["encoder_forwards"], 50)
        identity, dependencies = {"fictional": True}, {}
        with patch.object(runtime, "source_identity", return_value=identity), \
             patch.object(runtime, "run_path", return_value=SimpleNamespace(fold_dir=Path("fictional"))), \
             patch.object(runtime.original, "read_private_json", return_value={"identity": identity, "dependencies": dependencies, "check": record}):
            with self.assertRaises(ValueError):
                runtime.verify_resource("stage7_resource_fictional", "a"*40, dependencies, [4, 16, 16])
            # Simulate ONLY metadata from a genuine future CUDA/gzip check to
            # exercise strict gate arithmetic; no CUDA execution is claimed.
            record.update(device="cuda", io_backend="nifti_gzip_proxy")
            record["cache"]["backend"] = "sequential_nifti_proxy"
            record["profile"].update(peak_cuda_allocated_bytes=1024, peak_cuda_reserved_bytes=2048)
            record["profile"]["phases"]["resource_fixture"]["calls"] = 1
            runtime.verify_resource("stage7_resource_fictional", "a"*40, dependencies, [4, 16, 16])
            with self.assertRaises(ValueError):
                runtime.verify_resource("stage7_resource_fictional", "a"*40, dependencies, [5, 16, 16])
            record["projected_23_case_seconds_with_2x_margin"] += 1
            with self.assertRaises(ValueError):
                runtime.verify_resource("stage7_resource_fictional", "a"*40, dependencies, [4, 16, 16])

    def test_ctp_resource_identity_required_before_any_server_access(self):
        from ele630 import xai_runner as runner
        with patch.object(runner.runtime, "source_identity") as source:
            with self.assertRaisesRegex(ValueError, "resource run"):
                runner.execute("ctp", "a"*40, "stage7_ctp_fictional", execute_real=True,
                               protocol_approval="b"*64, smoke_run="stage7_smoke_fictional")
            source.assert_not_called()

    def test_fixture_failure_cleans_up_only_after_our_exclusive_creation(self):
        from ele630.xai_resource_check import synthetic_resource_check
        for ours in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/"fictional.nii.gz"
                def failed_fixture(path, shape, profile, on_created):
                    path.write_bytes(b"fictional concurrent or owned fixture")
                    if ours:
                        on_created()
                    raise ValueError("fictional fixture creation failure")
                with patch("ele630.xai_resource_check._nifti_fixture", side_effect=failed_fixture):
                    with self.assertRaises(ValueError):
                        synthetic_resource_check((4, 16, 16), device="cpu", fixture_path=path)
                self.assertEqual(path.exists(), not ours)

    def test_nifti_fixture_reads_indexed_gzip_capability_from_compression(self):
        import io
        from types import ModuleType
        from unittest.mock import Mock, MagicMock
        from ele630.xai_resource_check import _nifti_fixture
        for available in (True, False):
            with self.subTest(indexed_gzip_available=available):
                nib = ModuleType("nibabel")
                compression = ModuleType("nibabel._compression")
                compression.HAVE_INDEXED_GZIP = available
                nib.openers = SimpleNamespace()  # No capability attribute in nibabel 5.4.2.
                header = MagicMock(binaryblock=b"\0"*348)
                nib.Nifti1Header = Mock(return_value=header)
                nib.load = Mock(return_value=object())
                profile, created = Mock(), Mock()
                with patch.dict(sys.modules, {"nibabel": nib, "nibabel._compression": compression}), \
                     patch("ele630.ncct_private_output._validate_base"), \
                     patch("ele630.ncct_private_output.open_private_binary", return_value=io.BytesIO()):
                    proxy, indexed = _nifti_fixture(Path("fictional.nii.gz"), (4, 4, 4), profile, created)
                self.assertIs(indexed, available)
                self.assertEqual(proxy.metadata.frame_count, 40)
                self.assertEqual(profile._check_deadline.call_count, 40)
                created.assert_called_once_with()
                nib.load.assert_called_once_with("fictional.nii.gz")

    @unittest.skipUnless(importlib.util.find_spec("nibabel") is not None,
                         "nibabel absent locally: real NIfTI header/gzip-proxy integration requires UiS")
    def test_real_nifti_header_and_gzip_resource_fixture(self):
        import nibabel as nib
        from ele630 import ncct_private_output as private
        from ele630.xai_masks import read_geometry_header
        from ele630.xai_resource_check import synthetic_resource_check
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            folder = root/"fictional"
            folder.mkdir(mode=0o700)
            affine = np.diag([-1.2, -1.3, 3., 1.])
            header_path = folder/"fictional_header.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((16, 16, 4), dtype=np.float32), affine), str(header_path))
            shape, parsed = read_geometry_header(header_path)
            self.assertEqual(shape, (16, 16, 4))
            np.testing.assert_allclose(parsed, affine, atol=1e-5, rtol=0)
            fixture_path = folder/"fictional_resource.nii.gz"
            with patch.object(private, "_APPROVED_PRIVATE_ROOT", root), patch.object(private, "_TRUSTED_CANONICAL_ROOT", root):
                record = synthetic_resource_check((4, 16, 16), device="cpu", fixture_path=fixture_path)
            self.assertEqual(record["io_backend"], "nifti_gzip_proxy")
            self.assertEqual(record["profile"]["encoder_forwards"], 50)
            self.assertFalse(fixture_path.exists())


class MaskPreflightReviewTests(unittest.TestCase):
    def test_standalone_mask_mode_loads_neither_checkpoints_nor_clinical_rows(self):
        from ele630 import xai_runner as runner
        plan = self.plan()
        with patch.object(runner.runtime, "source_identity", return_value={"source_revision": "a"*40}), \
             patch.object(runner.original, "environment"), \
             patch("ele630.fusion_runner.load_plan", return_value=(object(), plan)), \
             patch("ele630.xai_masks.real_mask_preflight", return_value={"path_contract_verified": False}) as headers, \
             patch("ele630.fusion_checkpoints.load_ncct") as checkpoint, \
             patch.object(runner.runtime, "bridge") as clinical:
            result = runner.mask_preflight("a"*40)
        self.assertEqual(result["status"], "MASK_HEADER_PREFLIGHT_BLOCKED")
        self.assertFalse(result["checkpoints_loaded"])
        self.assertFalse(result["image_or_clinical_values_accessed"])
        headers.assert_called_once()
        checkpoint.assert_not_called()
        clinical.assert_not_called()

    def plan(self):
        from ele630.fusion_synthetic import fictional_plan
        plan = fictional_plan()
        return replace(plan, inner_validation=tuple(replace(p, has_gt=i != 0) for i, p in enumerate(plan.inner_validation)))

    def paths(self, root, plan):
        paths = {}
        for p in plan.inner_validation:
            if p.has_gt:
                directory = root/"derivatives"/p.patient_id
                directory.mkdir(parents=True)
                mask = directory/"fictional_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz"
                mask.touch()
                paths[p.patient_id] = mask
        return paths

    def image_path(self, plan, calls):
        def path(p, modality):
            self.assertIn(p, plan.inner_validation)
            calls.append((p, modality))
            return Path(modality+".nii.gz")
        return path

    def header(self, path):
        return ((16, 16, 4, 40) if path.name == "ctp.nii.gz" else (16, 16, 4)), np.eye(4)

    def test_metadata_only_counts_distinguish_absence_registration_and_path_failure(self):
        from ele630.xai_masks import mask_header_preflight
        plan, calls = self.plan(), []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self.paths(root, plan)
            invalid = paths[plan.inner_validation[1].patient_id]
            def header(path):
                shape, affine = self.header(path)
                if path == invalid:
                    affine[0, 3] = 2
                return shape, affine
            record = mask_header_preflight(plan, root, self.image_path(plan, calls), header_reader=header)
            self.assertEqual(record["counts"]["manifest_absent"], 1)
            self.assertEqual(record["counts"]["invalid_geometry"], 1)
            self.assertEqual(record["counts"]["valid_header"], 4)
            self.assertTrue(record["path_contract_verified"])
            self.assertFalse(record["voxel_values_read"])
            self.assertEqual(len(calls), 12)
            text = json.dumps(record)
            self.assertFalse(any(p.patient_id in text for p in plan.inner_train+plan.inner_validation+plan.outer_test))

    def test_systematic_wrong_layout_detected_without_loading_or_substitution(self):
        from ele630.xai_masks import mask_header_preflight
        plan, calls = self.plan(), []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for p in plan.inner_validation:
                if p.has_gt:
                    directory = root/"derivatives"/"collection"/p.patient_id
                    directory.mkdir(parents=True)
                    (directory/"fictional_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz").touch()
            # A forbidden outer directory must NEVER be traversed.
            outer = root/"derivatives"/plan.outer_test[0].patient_id
            outer.mkdir()
            original = Path.iterdir
            def iterdir(path):
                if path == outer:
                    raise AssertionError("outer directory traversal")
                return original(path)
            def header(path):
                self.assertIn(path.name, ("ncct.nii.gz", "ctp.nii.gz"))
                return self.header(path)
            with patch.object(Path, "iterdir", iterdir):
                record = mask_header_preflight(plan, root, self.image_path(plan, calls), header_reader=header)
            self.assertTrue(record["systematic_path_mismatch_suspected"])
            self.assertFalse(record["path_contract_verified"])
            self.assertEqual(record["alternate_name_or_layout_candidates"], 5)
            self.assertEqual(record["resolved_expected_paths"], 0)
            self.assertFalse(record["alternatives_substituted"])

    def test_wrong_filename_ambiguity_and_missing_are_separate(self):
        from ele630.xai_masks import mask_header_preflight, resolve_mask, MaskPathError
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self.paths(root, plan)
            p = plan.inner_validation[1]
            paths[p.patient_id].rename(paths[p.patient_id].with_name("fictional_other_mask.nii.gz"))
            record = mask_header_preflight(plan, root, self.image_path(plan, []), header_reader=self.header)
            self.assertEqual(record["counts"]["missing_expected_name"], 1)
            self.assertEqual(record["alternate_name_or_layout_candidates"], 1)
            p = plan.inner_validation[2]
            paths[p.patient_id].with_name("other_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz").touch()
            with self.assertRaises(MaskPathError) as caught:
                resolve_mask(root, p)
            self.assertEqual(caught.exception.kind, "ambiguous_path")

    def test_header_reader_cannot_access_voxels_even_with_a_bad_backend(self):
        from ele630.xai_masks import read_geometry_header
        seen = []
        class Header:
            def get_data_shape(self):
                return (16, 16, 4)
            def get_best_affine(self):
                return np.eye(4)
            def __getitem__(self, key):
                return 1
        def parse(stream, check):
            data = stream.read()
            seen.append(len(data))
            self.assertEqual(data, b"h"*348)
            self.assertEqual(stream.read(), b"")
            return Header()
        backend = SimpleNamespace(Nifti1Header=SimpleNamespace(from_fileobj=parse))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"fictional.nii.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(b"h"*348+b"fictional voxel bytes must not be passed to parser")
            with patch.dict(sys.modules, {"nibabel": backend}):
                shape, affine = read_geometry_header(path)
        self.assertEqual(seen, [348])
        self.assertEqual(shape, (16, 16, 4))

    def test_resource_envelope_includes_training_headers_but_never_outer(self):
        from ele630.xai_masks import mask_header_preflight
        plan, reads = self.plan(), []
        def image_path(p, modality):
            self.assertIn(p, plan.inner_train+plan.inner_validation)
            reads.append(p)
            return Path("training" if p in plan.inner_train else "validation")/(modality+".nii.gz")
        def header(path):
            if path.parent.name == "training":
                return ((32, 40, 12, 40) if path.name == "ctp.nii.gz" else (32, 40, 12)), np.eye(4)
            return self.header(path)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.paths(root, plan)
            record = mask_header_preflight(plan, root, image_path, header_reader=header, include_training_geometry=True)
        self.assertEqual(record["training_image_header_cases"], 18)
        self.assertEqual(record["maximum_shape_zyx"]["ctp"], [12, 40, 32])
        self.assertEqual(len(reads), 48)


if __name__ == "__main__":
    unittest.main()
