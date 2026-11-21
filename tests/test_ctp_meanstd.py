"""Synthetic-only checks for the prespecified 40-frame mean+population-std model."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_ctp_model import SyntheticSeries, synthetic_data

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from ele630.ctp_intensity import scale_ctp_input
    from ele630.ctp_model import CompactCTPMean, CompactCTPMeanStd


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CompactCTPMeanStdTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(630)

    def test_known_128_dim_population_statistics_all_frames_and_permutation(self):
        model = CompactCTPMeanStd(width=16).eval()
        data = np.broadcast_to(np.arange(40, dtype=np.float32), (3, 5, 7, 40)).copy()
        series = SyntheticSeries(data)

        def known_embedding(_):
            index = series.calls[-1]
            result = torch.arange(128, dtype=torch.float32)
            result[0] = 5
            result[1:] += 4 * (index % 4)
            return result.unsqueeze(0)

        with patch.object(model.encoder, "forward_features", side_effect=known_embedding), torch.no_grad():
            stacked, mean, std = model._temporal_statistics(series)
            logit = model(series)
        expected_mean = torch.arange(128, dtype=torch.float32).unsqueeze(0) + 6
        expected_mean[0, 0] = 5
        expected_std = torch.full((1, 128), 20 ** 0.5)
        expected_std[0, 0] = 0
        self.assertEqual(tuple(stacked.shape), (40, 1, 128))
        self.assertEqual(tuple(torch.cat((mean, std), dim=1).shape), (1, 256))
        torch.testing.assert_close(mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(std, expected_std, rtol=1e-6, atol=1e-6)
        self.assertEqual(series.calls, list(range(40)) * 2)
        self.assertTrue(torch.isfinite(logit).all().item())
        self.assertLess(float(std[0, 1]), float(stacked[:, 0, 1].std(unbiased=True)))

        reversed_series = SyntheticSeries(data[..., ::-1].copy())
        with torch.no_grad():
            original = model(SyntheticSeries(data))
            reversed_logit = model(reversed_series)
        torch.testing.assert_close(original, reversed_logit, rtol=1e-5, atol=1e-6)
        self.assertEqual(reversed_series.calls, list(range(40)))

    def test_frame_axis_scaling_variable_shapes_configuration_and_strict_state(self):
        model = CompactCTPMeanStd(width=4).eval()
        self.assertEqual(model.configuration["temporal_pooling"], "mean_std_population")
        self.assertEqual(model.configuration["embedding_dim"], 32)
        self.assertEqual(model.configuration["head_input_dim"], 64)
        self.assertEqual(model.configuration["std_correction"], 0)
        self.assertEqual(model.configuration["intensity_divisor"], 255.0)
        self.assertEqual(model.head.in_features, 64)
        clone = CompactCTPMeanStd(width=4).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        mean_model = CompactCTPMean(width=4)
        self.assertEqual(mean_model.configuration["temporal_pooling"], "mean")
        self.assertEqual(mean_model.head.in_features, 32)
        with self.assertRaises(RuntimeError):
            clone.load_state_dict(mean_model.state_dict(), strict=True)
        mean_clone = CompactCTPMean(width=4)
        mean_clone.load_state_dict(mean_model.state_dict(), strict=True)
        self.assertEqual(set(mean_clone.state_dict()), set(mean_model.state_dict()))

        for shape in ((9, 7, 5), (7, 9, 7)):
            series = SyntheticSeries(synthetic_data(shape))
            seen = []
            original = model.encoder.forward_features

            def checked_features(frame):
                index = series.calls[-1]
                expected = series.data[..., index].transpose(2, 1, 0).copy() / np.float32(255)
                np.testing.assert_allclose(frame.detach().cpu().numpy()[0, 0], expected,
                                           rtol=1e-6, atol=1e-7)
                seen.append(index)
                return original(frame)

            with self.subTest(shape=shape), patch.object(model.encoder, "forward_features", side_effect=checked_features), torch.no_grad():
                logit = model(series)
                self.assertEqual(tuple(logit.shape), (1,))
                self.assertTrue(torch.isfinite(logit).all().item())
                self.assertEqual(seen, list(range(40)))

    def test_streamed_gradients_match_full_graph_reference_one_graph_at_a_time(self):
        source = synthetic_data()
        model = CompactCTPMeanStd(width=2)
        reference = CompactCTPMeanStd(width=2)
        reference.load_state_dict(model.state_dict(), strict=True)
        target = torch.tensor([1.0])
        reference_series = SyntheticSeries(source)
        embeddings = []
        for index in range(40):
            native = reference_series.load_frame(index).input.tensor
            frame = torch.from_numpy(scale_ctp_input(native)).unsqueeze(0)
            embeddings.append(reference.encoder.forward_features(frame))
        stacked = torch.stack(embeddings)
        pooled = torch.cat((stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)), dim=1)
        reference_logit = reference.head(pooled).squeeze(1)
        reference_loss = nn.BCEWithLogitsLoss()(reference_logit, target)
        reference_loss.backward()

        series = SyntheticSeries(source)
        events = []
        original = model.encoder.forward_features

        def observed(frame):
            features = original(frame)
            if torch.is_grad_enabled():
                events.append("forward")
                features.register_hook(lambda gradient: events.append("backward"))
            return features

        with patch.object(model.encoder, "forward_features", side_effect=observed):
            result = model.backward_series(series, target)
        self.assertEqual(events, ["forward", "backward"] * 40)
        self.assertEqual(series.calls, list(range(40)) * 2)
        torch.testing.assert_close(result.logit, reference_logit.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result.loss, reference_loss.detach(), rtol=1e-5, atol=1e-6)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad,
                                       rtol=5e-4, atol=3e-6)
        self.assertGreater(float(model.encoder.stem[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_constant_embeddings_have_zero_std_and_finite_reference_gradients(self):
        class ConstantEncoder(nn.Module):
            width = 2

            def __init__(self):
                super().__init__()
                self.embedding = nn.Parameter(torch.ones(16, dtype=torch.float32))

            def forward_features(self, frame):
                return self.embedding.unsqueeze(0)

        model = CompactCTPMeanStd(width=2)
        model.encoder = ConstantEncoder()
        reference = CompactCTPMeanStd(width=2)
        reference.encoder = ConstantEncoder()
        reference.load_state_dict(model.state_dict(), strict=True)
        series = SyntheticSeries(np.zeros((3, 5, 7, 40), dtype=np.float32))
        with torch.no_grad():
            _, mean, std = model._temporal_statistics(series)
        torch.testing.assert_close(mean, model.encoder.embedding.unsqueeze(0), rtol=0, atol=1e-7)
        torch.testing.assert_close(std, torch.zeros_like(std), rtol=0, atol=0)
        target = torch.tensor([0.0])
        embeddings = torch.stack([reference.encoder.forward_features(None) for _ in range(40)])
        pooled = torch.cat((embeddings.mean(0), embeddings.std(0, unbiased=False)), dim=1)
        reference_loss = nn.BCEWithLogitsLoss()(reference.head(pooled).squeeze(1), target)
        reference_loss.backward()
        result = model.backward_series(SyntheticSeries(series.data), target)
        self.assertTrue(torch.isfinite(result.loss).item())
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                       rtol=5e-4, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
