"""Fictional-only Stage7 CPU cache, equivalence, lifetime and planning tests."""
from __future__ import annotations

import importlib.util
import pickle
import tracemalloc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ele630.fusion_synthetic import FictionalCTP, fictional_plan
from ele630.xai_ctp_cache import CachedCTPFrames, cached_series, prepared_bytes, MAX_CACHE_BYTES


class CacheTests(unittest.TestCase):
    def test_exact_prepared_frames_normalization_readonly_and_release(self):
        from ele630.ctp_intensity import scale_ctp_input
        source = FictionalCTP(630)
        source.metadata.affine = np.array([[0., -1.3, .1, 12.], [1.2, 0., 0., -9.], [0., 0., 3., 7.], [0., 0., 0., 1.]])
        with cached_series(source) as cache:
            self.assertEqual(source.reads, list(range(40)))
            refs = []
            for i in range(40):
                a, b = cache.load_frame(i).input, source.load_frame(i).input
                np.testing.assert_array_equal(a.tensor, b.tensor)
                np.testing.assert_array_equal(a.native_affine, b.native_affine)
                np.testing.assert_array_equal(scale_ctp_input(a.tensor), scale_ctp_input(b.tensor))
                self.assertEqual(a.frame_index, i)
                self.assertFalse(a.tensor.flags.writeable)
                refs.append(weakref.ref(a.tensor))
            del a, b
            self.assertEqual(cache.record()["frame_hits"], 40)
            self.assertEqual(cache.record()["prepared_bytes"], 40*4*4*16*16)
            with self.assertRaises(TypeError):
                pickle.dumps(cache)
        self.assertTrue(all(ref() is None for ref in refs))
        with self.assertRaises(ValueError):
            cache.load_frame(0)

    def test_bounds_before_decode_invalid_frames_and_exception_cleanup(self):
        source = FictionalCTP(630)
        with self.assertRaises(ValueError):
            CachedCTPFrames(source, max_bytes=1)
        self.assertEqual(source.reads, [])
        for index in (-1, 40, True):
            with cached_series(source) as cache:
                with self.assertRaises(ValueError):
                    cache.load_frame(index)
        cache = None
        with self.assertRaisesRegex(RuntimeError, "fictional analysis failure"):
            with cached_series(source) as cache:
                raise RuntimeError("fictional analysis failure")
        self.assertTrue(cache._closed)
        self.assertEqual(cache._frames, [])
        class Bad(FictionalCTP):
            def load_frame(self, i):
                frame = super().load_frame(i)
                if i == 7:
                    frame.input.tensor.flat[0] = np.nan
                return frame
        with self.assertRaises(ValueError):
            CachedCTPFrames(Bad(630))
        self.assertEqual(prepared_bytes((512, 603, 59)), 2914467840)
        self.assertLess(prepared_bytes((512, 603, 59)), MAX_CACHE_BYTES)

    def test_planning_charges_all138_caches_and_unchanged_forward_workload(self):
        from ele630.xai_resources import ResourceProfile, planning_seconds, ctp_workload, validate_profile
        record = ResourceProfile().record()
        times = {"resource_cache_creation": 7., "resource_baseline": 40*.1,
                 "resource_cam": .3, "resource_upper_cam": .2,
                 "resource_spatial": .4, "resource_temporal": 5*.5}
        for name, seconds in times.items():
            record["phases"][name]["seconds"] = seconds
        expected = 2*(138*7+4600*.1+23*(80*.1+120*.3+520*.4+95*.5))
        self.assertAlmostEqual(planning_seconds(record), expected)
        record["phases"]["resource_cache_creation"]["seconds"] += 10
        self.assertAlmostEqual(planning_seconds(record)-expected, 2*138*10)
        self.assertEqual(ctp_workload()["optimized_total_forwards"], 23345)
        self.assertEqual(ctp_workload()["total_cache_builds"], 138)
        with self.assertRaises(ValueError):
            validate_profile({**record, "schema": 1})

    def test_construction_peak_is_retained_frames_plus_bounded_frame_staging(self):
        shape = (8, 64, 64)
        retained = prepared_bytes(tuple(reversed(shape)))
        tracemalloc.start()
        try:
            with cached_series(FictionalCTP(630, shape)) as cache:
                _, peak = tracemalloc.get_traced_memory()
                self.assertEqual(cache.record()["prepared_bytes"], retained)
                # No additional whole-series array: allow64 bytes/voxel staging
                # and1MiB for Python/NumPy bookkeeping on this fictional path.
                self.assertLess(peak, retained+64*np.prod(shape)+1024**2)
        finally:
            tracemalloc.stop()

    @unittest.skipUnless(importlib.util.find_spec("nibabel") is not None, "nibabel absent: real gzip proxy equivalence requires UiS")
    def test_nifti_single_stream_exact_scaled_frames_affine_and_no_rereads(self):
        import nibabel as nib
        from nibabel.arrayproxy import ArrayProxy
        from nibabel.openers import ImageOpener
        from ele630.ctp import RegisteredCTPSeries
        # Nontrivial affine, float32 boundaries and NIfTI slope/intercept.
        affine = np.array([[0., -1.3, .1, 12.], [1.2, 0., 0., -9.], [0., 0., 3., 7.], [0., 0., 0., 1.]])
        data = np.random.default_rng(630).uniform(0, 100, (16, 17, 4, 40)).astype(np.float32)
        data[0, 0, 0, 0], data[-1, -1, -1, -1] = 0, 100
        with tempfile.TemporaryDirectory() as temp:
            for suffix, small in ((".nii", False), (".nii.gz", False), ("_tiny.nii.gz", True)):
                native = data[:4, :4, :4, :] if small else data
                path = Path(temp)/("fictional"+suffix)
                image = nib.Nifti1Image(native, affine)
                image.header.set_slope_inter(2., 7.)
                nib.save(image, path)
                image = nib.load(path)
                raw = RegisteredCTPSeries(SimpleNamespace(spatial_shape=native.shape[:3], frame_count=40, affine=image.affine), image.dataobj)
                opens, seeks, position = [], [], [0]
                # Instrument the actual nibabel stream, retaining read semantics.
                class ObservedStream:
                    def __init__(self):
                        self.stream = ImageOpener(image.dataobj.file_like, "rb", keep_open=True)
                    def __enter__(self):
                        opens.append(1)
                        return self
                    def __exit__(self, *args):
                        return self.stream.__exit__(*args)
                    def read(self, n=-1):
                        value = self.stream.read(n)
                        position[0] = self.stream.tell()
                        return value
                    def readinto(self, buffer):
                        result = self.stream.readinto(buffer)
                        position[0] = self.stream.tell()
                        return result
                    def seek(self, offset, whence=0):
                        self.assert_forward(offset, whence)
                        result = self.stream.seek(offset, whence)
                        position[0] = self.stream.tell()
                        return result
                    def tell(self):
                        return self.stream.tell()
                    @property
                    def closed(self):
                        return self.stream.closed
                    @property
                    def mode(self):
                        return self.stream.mode
                    def fileno(self):
                        return self.stream.fileno()
                    def write(self, value):
                        raise OSError("read-only fictional stream")
                    def assert_forward(self, offset, whence):
                        if whence != 0 or offset < position[0]:
                            raise AssertionError("gzip stream rewound")
                        seeks.append(offset)
                # Expected arrays come from the ORIGINAL slice-based loader.
                expected = [raw.load_frame(i).input for i in range(40)]
                def opener(file_like, *args, **kwargs):
                    return ImageOpener(file_like, *args, **kwargs) if hasattr(file_like, "read") else ObservedStream()
                with patch("nibabel.openers.ImageOpener", side_effect=opener):
                    with cached_series(raw) as cache:
                        self.assertEqual(len(opens), 1)
                        for i, reference in enumerate(expected):
                            a = cache.load_frame(i).input
                            np.testing.assert_array_equal(a.tensor, reference.tensor)
                            np.testing.assert_array_equal(a.native_affine, reference.native_affine)
                        # Cached reads cannot invoke either nibabel slice API.
                        with patch.object(ArrayProxy, "__getitem__", side_effect=AssertionError("proxy reread")):
                            for i in range(40):
                                cache.load_frame(i)
                self.assertTrue(seeks)
                self.assertEqual(position[0], image.dataobj.offset+native.size*4)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class CacheTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_all_development_features_match_with_membership_and_lifetime_guards(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import DevelopmentFeatures
        from ele630.xai_data import cached_development_features
        from ele630.xai_resources import ResourceProfile
        torch.manual_seed(630)
        model = CompactCTPMean()
        plan = fictional_plan()
        patients = plan.inner_train+plan.inner_validation
        cache_refs = []
        def loader(patient):
            # Every previous cache is closed before opening the next subject.
            self.assertTrue(all(ref() is None for ref in cache_refs))
            return FictionalCTP(630+patients.index(patient))
        original = DevelopmentFeatures(plan, loader, model, "ctp")
        profile = ResourceProfile()
        cached = cached_development_features(plan, loader, model, resources=profile)
        original_init = CachedCTPFrames.__init__
        def tracked_init(cache, *args, **kwargs):
            original_init(cache, *args, **kwargs)
            cache_refs.extend(weakref.ref(f.input.tensor) for f in cache._frames)
        with patch.object(CachedCTPFrames, "__init__", tracked_init):
            for partition in (plan.inner_train, plan.inner_validation):
                expected, p = original.partition(partition)
                actual, q = cached.partition(partition)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(p, q)
        self.assertTrue(all(ref() is None for ref in cache_refs))
        self.assertEqual(profile.cache["builds"], len(patients))
        for patient in (plan.outer_test[0], SimpleNamespace(patient_id=patients[0].patient_id)):
            with self.assertRaises(ValueError):
                cached.load(patient)

    def test_full_case_outputs_targets_controls_and_weights_are_identical(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import freeze
        from ele630.xai_analysis import analyse_case
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        from ele630.xai_resources import ResourceProfile
        from ele630 import xai_spatial, xai_analysis
        torch.manual_seed(630)
        model = freeze(CompactCTPMean())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        state, clinical = torch_state(fictional_state(), "cpu"), torch.zeros(1, 11)
        labels = np.ones((32, 32, 8), dtype=np.uint8)
        labels[:4] = 0
        labels[16:24, 16:24, 4:] = 2
        labels[20:24, 20:24, 4:] = 3
        gradients, masks = [], []
        gradient, blur = xai_spatial.image_gradient, xai_analysis.replace_blur
        def collect_gradient(*a, **k):
            result = gradient(*a, **k)
            gradients.append(result.clone())
            return result
        def collect_blur(frame, mask):
            masks.append(np.packbits(mask).tobytes())
            return blur(frame, mask)
        profiles = [ResourceProfile(), ResourceProfile()]
        source = FictionalCTP(630, (8, 32, 32))
        with patch.object(xai_spatial, "image_gradient", side_effect=collect_gradient), \
             patch.object(xai_analysis, "replace_blur", side_effect=collect_blur):
            with profiles[0].attach(model):
                expected = analyse_case(model, source, clinical, state, "ctp", np.eye(4), labels_xyz=labels)
            n_grad, n_masks = len(gradients), len(masks)
            source.reads.clear()
            with cached_series(source) as cache, profiles[1].attach(model):
                actual = analyse_case(model, cache, clinical, state, "ctp", np.eye(4), labels_xyz=labels)
                self.assertEqual(source.reads, list(range(40)))
        self.assertEqual(actual, expected)  # all metrics, NA reasons AND audit fields
        self.assertEqual(profiles[0].forwards, profiles[1].forwards)
        self.assertLessEqual(profiles[1].forwards, 815)
        self.assertEqual(masks[:n_masks], masks[n_masks:])
        self.assertEqual(len(gradients), 2*n_grad)
        for a, b in zip(gradients[:n_grad], gradients[n_grad:], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
