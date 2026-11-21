"""Synthetic-only TCN architecture, order and exact streamed-gradient checks."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_ctp_model import SyntheticSeries, synthetic_data

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from ele630.ctp_model import CompactCTPShallowTCN


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CompactCTPShallowTCNTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(630)

    def test_fixed_architecture_configuration_parameter_count_and_strict_loading(self):
        model = CompactCTPShallowTCN()
        first, relu1, second, relu2 = model.temporal
        self.assertEqual((first.in_channels, first.out_channels, first.kernel_size,
                          first.dilation, first.padding, first.stride),
                         (128, 32, (3,), (1,), (1,), (1,)))
        self.assertEqual((second.in_channels, second.out_channels, second.kernel_size,
                          second.dilation, second.padding, second.stride),
                         (32, 32, (3,), (2,), (2,), (1,)))
        self.assertIsInstance(relu1, nn.ReLU)
        self.assertIsInstance(relu2, nn.ReLU)
        self.assertIsInstance(model.encoder.classifier, nn.Identity)
        self.assertEqual((model.head.in_features, model.head.out_features), (32, 1))
        config = model.configuration
        self.assertEqual(json.loads(json.dumps(config)), config)
        self.assertEqual({key: config[key] for key in (
            "model_name", "model_version", "width", "embedding_dim", "frame_count",
            "temporal_channels", "kernel_sizes", "dilations", "paddings",
            "temporal_pooling", "order_sensitive", "intensity_divisor")}, {
                "model_name": "CompactCTPShallowTCN", "model_version": 1,
                "width": 16, "embedding_dim": 128, "frame_count": 40,
                "temporal_channels": [32, 32], "kernel_sizes": [3, 3],
                "dilations": [1, 2], "paddings": [1, 2], "temporal_pooling": "mean",
                "order_sensitive": True, "intensity_divisor": 255.0,
            })
        self.assertEqual(sum(p.numel() for p in model.encoder.parameters()), 897360)
        self.assertEqual(sum(p.numel() for p in model.temporal.parameters()), 15424)
        self.assertEqual(sum(p.numel() for p in model.head.parameters()), 33)
        clone = CompactCTPShallowTCN()
        clone.load_state_dict(model.state_dict(), strict=True)
        for width in (2, 32, True):
            with self.assertRaisesRegex(ValueError, "fixed width 16"):
                CompactCTPShallowTCN(width=width)

    def test_forward_reads_original_frame_order_axes_scale_and_variable_shapes(self):
        model = CompactCTPShallowTCN().eval()
        original = model.encoder.forward_features
        for shape in ((5, 7, 5), (7, 5, 7)):
            series = SyntheticSeries(synthetic_data(shape))
            seen = []

            def checked(frame):
                index = series.calls[-1]
                expected = series.data[..., index].transpose(2, 1, 0).copy() / np.float32(255)
                np.testing.assert_allclose(frame.numpy()[0, 0], expected, rtol=1e-6, atol=1e-7)
                seen.append(index)
                return original(frame)

            with self.subTest(shape=shape), patch.object(model.encoder, "forward_features", side_effect=checked), torch.no_grad():
                logit = model(series)
            self.assertEqual(tuple(logit.shape), (1,))
            self.assertTrue(torch.isfinite(logit).all().item())
            self.assertEqual(series.calls, list(range(40)))
            self.assertEqual(seen, list(range(40)))

    def test_controlled_weights_prove_temporal_order_changes_logit(self):
        model = CompactCTPShallowTCN().eval()
        with torch.no_grad():
            for parameter in model.temporal.parameters():
                parameter.zero_()
            model.temporal[0].weight[0, 0, 0] = 1  # Read the previous frame.
            model.temporal[2].weight[0, 0, 1] = 1  # Center tap, dilation 2.
            model.head.weight.zero_()
            model.head.weight[0, 0] = 1
            model.head.bias.zero_()
        data = np.zeros((3, 5, 7, 40), dtype=np.float32)
        data[..., 39] = 255

        def controlled_embedding(frame):
            features = torch.zeros((1, 128))
            features[:, 0] = frame.mean()
            return features

        original, reversed_series = SyntheticSeries(data), SyntheticSeries(data[..., ::-1].copy())
        with patch.object(model.encoder, "forward_features", side_effect=controlled_embedding), torch.no_grad():
            forward_logit = model(original)
            reverse_logit = model(reversed_series)
        # Padding drops the last-frame impulse for this deliberately directed filter.
        torch.testing.assert_close(forward_logit, torch.tensor([0.0]), rtol=0, atol=0)
        torch.testing.assert_close(reverse_logit, torch.tensor([1 / 40]), rtol=0, atol=1e-8)
        self.assertEqual(original.calls, list(range(40)))
        self.assertEqual(reversed_series.calls, list(range(40)))

    def test_streamed_all_parameter_gradients_match_full_graph_and_release_each_frame(self):
        model, reference = CompactCTPShallowTCN(), CompactCTPShallowTCN()
        reference.load_state_dict(model.state_dict(), strict=True)
        data = synthetic_data((5, 7, 5))
        reference_series = SyntheticSeries(data)
        target = torch.tensor([1.0])
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        # Full native-frame graphs are permitted only in this tiny synthetic oracle.
        features = [reference.encoder.forward_features(reference._frame_tensor(reference_series, index))
                    for index in range(40)]
        sequence = torch.stack(features, dim=2)
        reference_logit = reference.head(reference.temporal(sequence).mean(dim=2)).squeeze(1)
        reference_loss = nn.BCEWithLogitsLoss()(reference_logit, target)
        reference_loss.backward()
        series = SyntheticSeries(data)
        events = []
        previous_embedding = None
        encode = model.encoder.forward_features

        def observed(frame):
            nonlocal previous_embedding
            if torch.is_grad_enabled() and previous_embedding is not None:
                self.assertIsNone(previous_embedding(), "previous frame graph output was retained")
            embedding = encode(frame)
            if torch.is_grad_enabled():
                events.append("forward")
                embedding.register_hook(lambda gradient: events.append("backward"))
                previous_embedding = weakref.ref(embedding)
            return embedding

        with patch.object(model.encoder, "forward_features", side_effect=observed):
            result = model.backward_series(series, target)
        self.assertEqual(events, ["forward", "backward"] * 40)
        self.assertIsNone(previous_embedding())
        self.assertEqual(series.calls, list(range(40)) * 2)
        self.assertFalse(result.loss.requires_grad)
        self.assertFalse(result.logit.requires_grad)
        self.assertTrue(torch.isfinite(result.loss).item())
        torch.testing.assert_close(result.loss, reference_loss.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result.logit, reference_logit.detach(), rtol=1e-5, atol=1e-6)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad,
                                       rtol=5e-4, atol=3e-6)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        for parameter in (model.encoder.stem[0].weight, model.temporal[0].weight,
                          model.temporal[2].weight, model.head.weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_invalid_series_target_and_failed_recomputation_clear_partial_gradients(self):
        model = CompactCTPShallowTCN()
        series = SyntheticSeries(synthetic_data((5, 7, 5)))
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            model(series)
        self.assertEqual(series.calls, [])
        for target in (torch.tensor([0.5]), torch.tensor([float("nan")]), torch.tensor([1, 0])):
            with self.assertRaisesRegex(ValueError, "target"):
                model.backward_series(series, target)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "40 frames"):
            model(SyntheticSeries(series.data[..., :39]))
        load = series.load_frame

        def fail_second_pass(index):
            if len(series.calls) == 40:
                raise OSError("fictional interrupted frame")
            return load(index)

        with patch.object(series, "load_frame", side_effect=fail_second_pass):
            with self.assertRaises(OSError):
                model.backward_series(series, torch.tensor([1.0]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
