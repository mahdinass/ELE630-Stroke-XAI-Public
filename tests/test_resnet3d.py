"""Small synthetic CPU tensors only; full native-size tensors are pilot-only."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from ele630.resnet3d import CompactResNet10


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CompactResNet10Tests(unittest.TestCase):
    def test_feature_interface_bypasses_classifier_and_preserves_pooled_values(self):
        torch.manual_seed(636)
        for pooling in ("avg", "avgmax"):
            model = CompactResNet10(width=4, pooling=pooling).eval()
            source = torch.randn(1, 1, 9, 17, 15)
            keys = tuple(model.state_dict())
            with torch.no_grad():
                maps = model.stage4(model.stage3(model.stage2(model.stage1(model.stem(source)))))
                pooled = model.pool(maps)
                if pooling == "avgmax":
                    pooled = 0.5 * (pooled + model.max_pool(maps))
                with patch.object(model.classifier, "forward", side_effect=AssertionError("classifier called")):
                    embedding = model.forward_features(source)
            self.assertEqual(tuple(embedding.shape), (1, 32))
            torch.testing.assert_close(embedding, pooled.flatten(1), rtol=0, atol=0)
            self.assertEqual(tuple(model.state_dict()), keys)

    def test_batch_one_and_two_logits_with_variable_native_like_shapes(self):
        torch.manual_seed(630)
        with torch.no_grad():
            for pooling in ("avg", "avgmax"):
                model = CompactResNet10(width=4, pooling=pooling).eval()
                for batch, spatial in ((1, (8, 24, 20)), (2, (11, 19, 17))):
                    with self.subTest(pooling=pooling, batch=batch, spatial=spatial):
                        logits = model(torch.randn(batch, 1, *spatial))
                        self.assertEqual(tuple(logits.shape), (batch,))
                        self.assertTrue(torch.isfinite(logits).all().item())

    def test_default_avg_matches_explicit_legacy_computation(self):
        torch.manual_seed(632)
        model = CompactResNet10(width=4).eval()
        self.assertEqual(model.pooling, "avg")
        source = torch.randn(1, 1, 9, 17, 15)
        with torch.no_grad():
            features = model.stage4(model.stage3(model.stage2(model.stage1(model.stem(source)))))
            expected = model.classifier(model.pool(features).flatten(1)).squeeze(1)
            actual = model(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_avgmax_matches_explicit_fixed_combination(self):
        torch.manual_seed(633)
        model = CompactResNet10(width=4, pooling="avgmax").eval()
        source = torch.randn(1, 1, 9, 17, 15)
        with torch.no_grad():
            features = model.stage4(model.stage3(model.stage2(model.stage1(model.stem(source)))))
            expected_pool = 0.5 * (model.pool(features) + model.max_pool(features))
            expected = model.classifier(expected_pool.flatten(1)).squeeze(1)
            actual = model(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_state_dict_keys_and_strict_loading_are_preserved(self):
        torch.manual_seed(634)
        legacy_shaped = CompactResNet10(width=4)
        keys = tuple(legacy_shaped.state_dict())
        avgmax = CompactResNet10(width=4, pooling="avgmax")
        self.assertEqual(tuple(avgmax.state_dict()), keys)
        self.assertIn("classifier.weight", keys)
        self.assertIn("stage4.conv2.weight", keys)
        self.assertFalse(any("max_pool" in key for key in keys))
        self.assertEqual(sum(p.numel() for p in avgmax.parameters()),
                         sum(p.numel() for p in legacy_shaped.parameters()))
        restored_default = CompactResNet10(width=4).eval()
        restored_ablation = CompactResNet10(width=4, pooling="avgmax").eval()
        for restored in (restored_default, restored_ablation):
            incompatible = restored.load_state_dict(legacy_shaped.state_dict(), strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
        source = torch.randn(1, 1, 8, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(restored_default(source), legacy_shaped.eval()(source),
                                       rtol=0, atol=0)

    def test_localized_activation_changes_pool_aggregate_without_logit_assumption(self):
        model = CompactResNet10(width=4, pooling="avgmax")
        features = torch.zeros(1, 1, 2, 2, 2)
        features[0, 0, 1, 0, 1] = 8.0
        average = model.pool(features)
        maximum = model.max_pool(features)
        combined = 0.5 * (average + maximum)
        self.assertEqual(float(average.item()), 1.0)
        self.assertEqual(float(maximum.item()), 8.0)
        self.assertEqual(float(combined.item()), 4.5)

    def test_avgmax_backward_has_finite_gradients(self):
        torch.manual_seed(635)
        model = CompactResNet10(width=4, pooling="avgmax")
        logits = model(torch.randn(1, 1, 9, 17, 15))
        nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0])).backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all().item() for g in gradients))
        self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0.0)

    def test_backward_has_finite_nonzero_gradients(self):
        torch.manual_seed(631)
        model = CompactResNet10(width=4)
        logits = model(torch.randn(1, 1, 8, 16, 16))
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0]))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all().item() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)
        self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0.0)

    def test_explicit_seed_reproduces_scratch_initialization(self):
        torch.manual_seed(1234)
        first = CompactResNet10(width=4)
        first_weights = {name: value.clone() for name, value in first.state_dict().items()}
        torch.manual_seed(1234)
        second = CompactResNet10(width=4)
        for name, value in second.state_dict().items():
            self.assertTrue(torch.equal(first_weights[name], value), name)

    def test_default_parameter_count_and_shape_independent_classifier(self):
        model = CompactResNet10()
        self.assertEqual(model.width, 16)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 897_489)
        self.assertEqual(model.classifier.in_features, 128)
        self.assertIsInstance(model.pool, nn.AdaptiveAvgPool3d)
        self.assertFalse(hasattr(model, "max_pool"))
        self.assertTrue(any(isinstance(layer, nn.GroupNorm) for layer in model.modules()))
        self.assertFalse(any(isinstance(layer, nn.BatchNorm3d) for layer in model.modules()))

    def test_xai_target_is_accessible_without_retaining_feature_maps(self):
        model = CompactResNet10(width=4).eval()
        self.assertIs(model.xai_feature_layer, model.stage4.conv2)
        ablation = CompactResNet10(width=4, pooling="avgmax")
        self.assertIs(ablation.xai_feature_layer, ablation.stage4.conv2)
        self.assertIsInstance(model.xai_feature_layer, nn.Conv3d)
        shapes = []
        handle = model.xai_feature_layer.register_forward_hook(
            lambda _layer, _inputs, output: shapes.append(tuple(output.shape)),
        )
        try:
            with torch.no_grad():
                model(torch.randn(1, 1, 12, 32, 32))
        finally:
            handle.remove()
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0][0:2], (1, 32))
        self.assertGreater(shapes[0][2], 1)

    def test_rejects_invalid_input_shape_channels_and_width(self):
        model = CompactResNet10(width=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.zeros(1, 8, 16, 16))
        with self.assertRaisesRegex(ValueError, "one NCCT channel"):
            model(torch.zeros(1, 2, 8, 16, 16))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model(torch.zeros(1, 1, 0, 16, 16))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            model(torch.zeros(1, 1, 8, 16, 16, dtype=torch.int16))
        with self.assertRaisesRegex(ValueError, "width"):
            CompactResNet10(width=0)
        for pooling in ("max", "AVG", "", None, 1):
            with self.subTest(pooling=pooling), self.assertRaisesRegex(ValueError, "pooling"):
                CompactResNet10(width=4, pooling=pooling)


if __name__ == "__main__":
    unittest.main()
