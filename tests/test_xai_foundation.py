"""Synthetic numerical/immutability oracles. No patient-file access."""
import importlib.util
import unittest
from dataclasses import replace

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn unavailable in this interpreter")
class ReconstructionTests(unittest.TestCase):
    def test_reconstruction_matches_original_aggregates_and_train_only_fit(self):
        from ele630.fusion_synthetic import fictional_plan, fictional_rows
        from ele630.fusion import run_fusion
        from ele630.xai_reconstruction import reconstruct
        plan = fictional_plan()
        rng = np.random.default_rng(630)
        train, val = rng.normal(size=(18, 128)), rng.normal(size=(6, 128))
        rows = fictional_rows(plan)
        result = reconstruct(plan, rows, train, val, [.3] * 6)
        original = run_fusion(plan, rows, train, val, [.3] * 6)
        self.assertEqual(result["metrics"], {k: original["metrics"][k] for k in result["metrics"]})
        # Fit must not request validation clinical data until all fits finish.
        from unittest.mock import patch
        from ele630 import xai_reconstruction as module
        fitted = []
        original_fit = module._fit
        def fit(x, y):
            fitted.append(x.copy())
            self.assertEqual(len(x), 18)
            return original_fit(x, y)
        class Rows(dict):
            def __getitem__(self, key):
                if key in {p.patient_id for p in plan.inner_validation}:
                    if len(fitted) != 3:
                        raise AssertionError("validation requested before fits")
                return super().__getitem__(key)
        with patch.object(module, "_fit", side_effect=fit):
            repeated = reconstruct(plan, Rows(rows), train, val * 1000, [.3] * 6)
        self.assertEqual(result["state"], repeated["state"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch unavailable in this interpreter")
    def test_sklearn_logit_and_probability_oracle(self):
        import torch
        from ele630.fusion_synthetic import fictional_plan, fictional_rows
        from ele630.xai_reconstruction import reconstruct
        from ele630.xai_fused_score import torch_state, fused_logit_torch
        plan = fictional_plan()
        rng = np.random.default_rng(631)
        train, val = rng.normal(size=(18, 128)), rng.normal(size=(6, 128))
        result = reconstruct(plan, fictional_rows(plan), train, val, [.3] * 6)
        state = torch_state(result["state"], "cpu")
        for i in range(6):
            z = fused_logit_torch(torch.tensor(val[i:i+1]),
                                  torch.tensor(result["clinical_validation"][i:i+1]), state)
            self.assertAlmostEqual(z.item(), result["logits"][i], delta=1e-10)
            self.assertAlmostEqual(z.sigmoid().item(), result["probabilities"][i], delta=1e-12)


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable in this interpreter")
class TorchFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_fused_gradient_finite_difference_and_additive_clinical_independence(self):
        import torch
        from ele630.xai_fused_score import torch_state, fused_logit_torch, image_gradient, logit_components, predicted_sign
        from ele630.xai_synthetic import fictional_state
        state = torch_state(fictional_state(), "cpu")
        e = torch.linspace(-1, 1, 128, dtype=torch.float64).reshape(1, 128).requires_grad_()
        c = torch.zeros(1, 11)
        z = fused_logit_torch(e, c, state)
        g, = torch.autograd.grad(z, e)
        torch.testing.assert_close(g[0], image_gradient(state), rtol=1e-12, atol=1e-12)
        for j in (0, 70, 127):
            step = torch.zeros_like(e)
            step[0, j] = 1e-5
            fd = (fused_logit_torch(e+step, c, state)-fused_logit_torch(e-step, c, state))/2e-5
            self.assertAlmostEqual(fd.item(), g[0, j].item(), delta=1e-9)
        g2, = torch.autograd.grad(fused_logit_torch(e, c+10, state), e)
        torch.testing.assert_close(g, g2)
        self.assertAlmostEqual(sum(v.item() for v in logit_components(e, c, state).values()), z.item())
        self.assertEqual(predicted_sign(torch.tensor([0.])), 1)

    def test_precision_device_state_and_nonfinite_input_contract(self):
        import torch
        from ele630.xai_fused_score import torch_state, fused_logit_torch
        from ele630.xai_synthetic import fictional_state
        state = torch_state(fictional_state(), "cpu")
        z = fused_logit_torch(torch.ones(1, 128, dtype=torch.float32), torch.zeros(1, 11), state)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(z.device.type, "cpu")
        with self.assertRaises(ValueError):
            fused_logit_torch(torch.ones(1, 128), torch.zeros(1, 11), {k: v.float() for k, v in state.items()})
        with self.assertRaises(ValueError):
            fused_logit_torch(torch.ones(1, 128)*float("nan"), torch.zeros(1, 11), state)
        invalid = fictional_state()
        invalid["scale"][0] = 0
        with self.assertRaises(ValueError):
            torch_state(invalid, "cpu")

    def test_cam_independent_small_tensor_oracle_signed_and_positive(self):
        import torch
        from ele630.xai_spatial import cam_definitions
        a = torch.tensor([[[[[1., -2.], [3., 4.]]], [[[2., 1.], [-1., 3.]]]]])
        g = torch.tensor([[[[[1., 2.], [-3., 1.]]], [[[-2., 4.], [2., -1.]]]]])
        result = cam_definitions(a, g)
        an, gn = a.numpy()[0], g.numpy()[0]
        gc = sum(an[k]*gn[k].mean() for k in range(2))
        hr = sum(an[k]*gn[k] for k in range(2))
        for name, expected in (("gradcam", gc), ("hirescam", hr)):
            np.testing.assert_allclose(result[name]["signed"], expected)
            np.testing.assert_allclose(result[name]["positive"], np.maximum(expected, 0))
        with self.assertRaises(ValueError):
            cam_definitions(a*float("nan"), g)

    def test_real_ncct_topology_inplace_hooks_and_immutability(self):
        import torch
        from ele630.resnet3d import CompactResNet10
        from ele630.fusion_features import freeze
        from ele630.xai_fused_score import torch_state
        from ele630.xai_synthetic import fictional_state
        from ele630.xai_spatial import ncct_cam
        torch.manual_seed(630)
        model = freeze(CompactResNet10())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        frame = torch.rand(1, 1, 8, 32, 32)
        result = ncct_cam(model, frame, torch.zeros(1, 11), torch_state(fictional_state(), "cpu"))
        self.assertEqual(result["maps"]["gradcam"]["positive"].shape, (2, 2, 2))
        np.testing.assert_array_equal(result["maps"]["gradcam"]["poor_signed"],
                                      result["sign"]*result["maps"]["gradcam"]["signed"])
        self.assertEqual(len(model.xai_feature_layer._forward_hooks), 0)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):
            ncct_cam(model.train(), frame, torch.zeros(1, 11), torch_state(fictional_state(), "cpu"))

    def test_ctp_40_frame_chain_rule_matches_full_graph_oracle(self):
        import torch
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import freeze
        from ele630.fusion_synthetic import FictionalCTP
        from ele630.xai_spatial import ctp_cam, frame_cam
        from ele630.xai_fused_score import torch_state, fused_logit_torch, image_gradient
        from ele630.xai_synthetic import fictional_state
        torch.manual_seed(630)
        model = freeze(CompactCTPMean())
        state, clinical = torch_state(fictional_state(), "cpu"), torch.zeros(1, 11)
        series = FictionalCTP(630, (4, 16, 16))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = ctp_cam(model, series, clinical, state, sign=-1)
        self.assertEqual(series.reads, list(range(40))*2)
        self.assertEqual(len(result["frames"]), 40)
        # Independent all-40-graph oracle is used on tiny fictional tensors only.
        captured = []
        handle = model.encoder.xai_feature_layer.register_forward_hook(lambda m, a, o: captured.append(o))
        try:
            embeddings = [model.encoder.forward_features(model._frame_tensor(FictionalCTP(630), i).requires_grad_())
                          for i in range(40)]
            logit = -fused_logit_torch(torch.stack(embeddings).mean(0), clinical, state)
            gradients = torch.autograd.grad(logit, captured)
        finally:
            handle.remove()
        from ele630.xai_spatial import cam_definitions
        for i in range(40):
            oracle = cam_definitions(captured[i], gradients[i])
            for name in ("gradcam", "hirescam"):
                np.testing.assert_allclose(result["frames"][i][name]["signed"], oracle[name]["signed"], atol=1e-8, rtol=1e-5)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_partition_guard_rejects_outer_before_any_load(self):
        from unittest.mock import Mock
        from ele630.fusion_synthetic import fictional_plan
        from ele630.fusion_features import DevelopmentFeatures
        from ele630.resnet3d import CompactResNet10
        plan, loader = fictional_plan(), Mock()
        source = DevelopmentFeatures(plan, loader, CompactResNet10(), "ncct")
        for patient in (plan.outer_test[0], replace(plan.inner_validation[0], outcome="poor")):
            if patient == plan.inner_validation[0]:
                patient = replace(patient, cohort="forged")
            with self.assertRaises(ValueError):
                source.load(patient)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
