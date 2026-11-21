"""Fictional anatomical, perturbation, temporal and sanity-control oracles."""
import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


class QuantitativeNumpyTests(unittest.TestCase):
    def test_roi_mass_enrichment_overlap_missing_and_degenerate(self):
        from ele630.xai_metrics import roi_metrics, diagnostic
        labels = np.array([[[1, 2, 3, 0]]])
        x = np.array([[[1., 2., 3., 4.]]])
        result = roi_metrics(x, labels)
        self.assertAlmostEqual(result["regions"]["core"]["mass"], .5)
        self.assertAlmostEqual(result["regions"]["core"]["enrichment"], 1.5)
        self.assertAlmostEqual(result["regions"]["lesion_union"]["mass"], 5/6)
        self.assertAlmostEqual(result["nuisance"]["nonbrain"], .4)
        for value, reason in ((np.zeros_like(x), "near_zero_map"), (np.ones_like(x), "constant_map"),
                              (x*np.nan, "nonfinite_map")):
            self.assertEqual(diagnostic(value), reason)
            self.assertIsNone(roi_metrics(value, labels)["regions"]["core"]["mass"])
        self.assertEqual(roi_metrics(x)["reason"], "missing_mask")
        self.assertEqual(roi_metrics(x, np.zeros_like(labels))["reason"], "empty_brain")
        self.assertEqual(roi_metrics(x, np.ones_like(labels))["regions"]["core"]["reason"], "empty_region")
        self.assertEqual(roi_metrics(x, labels[..., :2])["reason"], "invalid_mask")

    def test_geometry_and_native_feature_centres(self):
        from ele630.xai_geometry import validate_labels, feature_labels, project_native
        labels = np.zeros((32, 32, 8), dtype=np.uint8)
        labels[16, 16, 4] = 3
        affine = np.diag([-1.2, -1.3, 3., 1.])
        accepted = validate_labels(labels, affine, labels.shape, affine)
        coarse = feature_labels(accepted, affine, (2, 2, 2))
        self.assertEqual(coarse[1, 1, 1], 3)
        h = np.arange(8.).reshape(2, 2, 2)
        native = project_native(h, (8, 32, 32))
        self.assertEqual(native[4, 16, 16], h[1, 1, 1])
        self.assertEqual(native[2, 8, 8], h.mean())
        self.assertEqual(native[-1, -1, -1], h[-1, -1, -1])
        wrong = affine.copy()
        wrong[0, 3] += 2
        for args in ((labels, affine, labels.shape, wrong), (labels+5, affine, labels.shape, affine)):
            with self.assertRaises(ValueError):
                validate_labels(*args)
        with self.assertRaises(ValueError):
            feature_labels(labels, affine, (2, 3, 2))

    def test_patch_budget_seed_ties_and_control_matching(self):
        from ele630.xai_perturbation import intervention_masks
        brain = np.ones((9, 34, 35), dtype=bool)
        brain[:, :2] = False
        h = np.arange(brain.size).reshape(brain.shape).astype(float)
        a, controls = intervention_masks(h, brain)
        a2, controls2 = intervention_masks(h, brain)
        self.assertEqual(len(controls), 10)
        for first, second in zip([a]+controls, [a2]+controls2):
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.sum(), int(np.ceil(.1*brain.sum())))
            self.assertFalse((first & ~brain).any())
        _, controls_other_map = intervention_masks(h[::-1], brain)
        for first, second in zip(controls, controls_other_map):
            np.testing.assert_array_equal(first, second)
        tied, _ = intervention_masks(np.ones_like(h), brain)
        self.assertTrue(tied.ravel()[np.flatnonzero(brain)[0]])

    def test_fidelity_fixed_score_and_negative_findings(self):
        from ele630.xai_perturbation import fidelity
        x = np.arange(2*16*32, dtype=float).reshape(2, 16, 32)
        result = fidelity(x, np.ones_like(x, dtype=bool), lambda m: (5.-x[m].sum()/1e5, int(m.sum())), 5.)
        self.assertAlmostEqual(result["excess_drop"], result["target_drop"]-result["random_median"])
        self.assertEqual(len(result["random_drops"]), 10)
        self.assertEqual(fidelity(np.zeros_like(x), np.ones_like(x, dtype=bool), None, 5.)["reason"], "near_zero_map")

    def test_rank_ties_and_degenerate_comparisons(self):
        from ele630.xai_metrics import ranks, compare_maps, boundary
        np.testing.assert_allclose(ranks([3., 1., 1., 4.]), [2, .5, .5, 3])
        h = np.arange(100.).reshape(1, 10, 10)
        self.assertAlmostEqual(compare_maps(h, h)["rank_correlation"], 1.)
        self.assertEqual(compare_maps(h, h)["top_dice"], 1.)
        self.assertAlmostEqual(compare_maps(h, h.max()-h)["rank_correlation"], -1.)
        self.assertIsNone(compare_maps(h, np.ones_like(h))["rank_correlation"])
        brain = np.ones((3, 3, 3), dtype=bool)
        self.assertEqual(boundary(brain).sum(), 26)

    def test_blur_oracle_constant_and_impulse(self):
        from ele630.xai_perturbation import blur_numpy, gaussian_kernel
        np.testing.assert_allclose(blur_numpy(np.ones((3, 4, 5))), 1.)
        impulse = np.zeros((17, 17, 17))
        impulse[8, 8, 8] = 1
        blurred = blur_numpy(impulse)
        self.assertAlmostEqual(blurred.sum(), 1.)
        self.assertAlmostEqual(blurred[8, 8, 8], gaussian_kernel()[6]**3)


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable in this interpreter")
class QuantitativeTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_torch_blur_matches_independent_oracle_and_unchanged_support(self):
        import torch
        from ele630.xai_perturbation import blur_numpy, replace_blur
        rng = np.random.default_rng(630)
        x = rng.random((4, 12, 13), dtype=np.float32)
        mask = rng.random(x.shape) > .5
        tensor = torch.tensor(x).reshape(1, 1, *x.shape)
        result, changed = replace_blur(tensor, mask)
        expected = np.where(mask, blur_numpy(x), x)
        np.testing.assert_allclose(result.numpy().reshape(x.shape), expected, atol=2e-7)
        np.testing.assert_array_equal(result.numpy().reshape(x.shape)[~mask], x[~mask])
        self.assertEqual(changed, mask.sum())
        self.assertFalse(result.requires_grad)

    def test_temporal_all_windows_agree_with_full_40_recomputation(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import freeze
        from ele630.fusion_synthetic import FictionalCTP
        from ele630.xai_temporal import temporal_occlusion, windows, replacement_frames
        from ele630.xai_fused_score import fused_logit_torch, torch_state
        from ele630.xai_synthetic import fictional_state
        torch.manual_seed(630)
        model = freeze(CompactCTPMean())
        series, c = FictionalCTP(630), torch.zeros(1, 11)
        state = torch_state(fictional_state(), "cpu")
        score = lambda e: fused_logit_torch(e, c, state)
        with torch.no_grad():
            baseline = -score(model._mean_embedding(series)).item()
        result = temporal_occlusion(model, series, score, baseline, sign=-1)
        self.assertEqual(result["window_starts"][-1], 35)
        self.assertTrue(all(n > 0 for n in result["coverage"]))
        frame_at = lambda i: model._frame_tensor(series, i)
        with torch.no_grad():
            for index, start in enumerate(windows()):
                replacement = dict(replacement_frames(frame_at, start))
                # Independent complete counterfactual, not an embedding delta.
                total = torch.zeros(1, 128)
                for i in range(40):
                    total += model.encoder.forward_features(replacement[i] if i in replacement else frame_at(i))
                oracle = baseline+score(total/40).item()
                self.assertAlmostEqual(result["window_drops"][index], oracle, delta=1e-6)
        # Explicit edge policy and interior weights, independent scalar series.
        left = dict(replacement_frames(lambda i: torch.tensor(float(i*i)), 0))
        self.assertTrue(all(v.item() == 25 for v in left.values()))
        right = dict(replacement_frames(lambda i: torch.tensor(float(i*i)), 35))
        self.assertTrue(all(v.item() == 34**2 for v in right.values()))
        middle = dict(replacement_frames(lambda i: torch.tensor(float(i*i)), 2))
        self.assertAlmostEqual(middle[2].item(), 9.)

    def test_randomization_separate_copies_reproducible_fixed_clinical_and_original(self):
        import torch
        from ele630.resnet3d import CompactResNet10
        from ele630.fusion_features import freeze
        from ele630.xai_randomization import randomized_copy
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        model = freeze(CompactResNet10())
        state = torch_state(fictional_state(), "cpu")
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        copy, randomized = randomized_copy(model, state, upper_block=True)
        repeat, repeated = randomized_copy(model, state, upper_block=True)
        torch.testing.assert_close(rng_before, torch.get_rng_state())
        torch.testing.assert_close(randomized["coef"][8:], state["coef"][8:])
        self.assertFalse(torch.equal(randomized["coef"][:8], state["coef"][:8]))
        self.assertFalse(torch.equal(copy.stage4.conv2.weight, model.stage4.conv2.weight))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            torch.testing.assert_close(copy.state_dict()[k], repeat.state_dict()[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in copy.parameters()))
        zero = {k: v.clone() for k, v in state.items()}
        zero["coef"][:8] = 0
        with self.assertRaisesRegex(ValueError, "zero_head"):
            randomized_copy(model, zero)


if __name__ == "__main__":
    unittest.main()
