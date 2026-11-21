"""Tiny synthetic arrays only; full-graph training appears only as a test oracle."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.ctp_input import prepare_ctp_frame_input
from ele630.ctp_intensity import scale_ctp_input

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
NIBABEL_AVAILABLE = importlib.util.find_spec("nibabel") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from ele630.ctp_model import CompactCTPMean


AFFINE = np.diag([-1.0, -1.0, 3.0, 1.0])


class SyntheticSeries:
    """Same frame adapter as Stage 4O, with no file access or patient labels."""

    def __init__(self, data):
        self.data = data
        self.metadata = SimpleNamespace(frame_count=data.shape[3], spatial_shape=data.shape[:3], affine=AFFINE)
        self.calls = []

    def load_frame(self, index):
        self.calls.append(index)
        return SimpleNamespace(input=prepare_ctp_frame_input(self.data[..., index], AFFINE, index))


def synthetic_data(shape=(9, 7, 5)):
    return np.random.default_rng(630).uniform(0, 255, (*shape, 40)).astype(np.float32)


class CTPIntensityTests(unittest.TestCase):
    def test_explicit_division_boundaries_and_independent_storage(self):
        source = np.array([[[[0, 127.5, 255]]]], dtype=np.float32)
        scaled = scale_ctp_input(source)
        np.testing.assert_array_equal(scaled, [[[[0, 0.5, 1]]]])
        self.assertEqual(scaled.dtype, np.float32)
        self.assertTrue(scaled.flags.c_contiguous)
        self.assertFalse(np.shares_memory(source, scaled))
        scaled[:] = 0
        np.testing.assert_array_equal(source, [[[[0, 127.5, 255]]]])

    def test_adapter_axes_preserved_by_intensity_scaling(self):
        x, y, z = np.indices((3, 5, 7))
        source = (50 * x + 10 * y + z).astype(np.float32)
        adapted = prepare_ctp_frame_input(source, AFFINE, 19)
        scaled = scale_ctp_input(adapted.tensor)
        self.assertEqual(scaled.shape, (1, 7, 5, 3))
        self.assertEqual(scaled[0, 6, 4, 2], np.float32(146) / np.float32(255))
        np.testing.assert_array_equal(scaled[0].transpose(2, 1, 0), source / np.float32(255))

    def test_invalid_values_types_and_shapes_rejected_without_clipping(self):
        for value in (-0.01, 255.01, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scale_ctp_input(np.full((1, 2, 3, 4), value, dtype=np.float32))
        for source in (np.zeros((2, 3, 4)), np.zeros((2, 2, 3, 4), dtype=np.float32),
                       np.zeros((1, 0, 2, 3), dtype=np.float32),
                       np.zeros((1, 2, 3, 4), dtype=np.float64)):
            with self.subTest(shape=source.shape), self.assertRaises(ValueError):
                scale_ctp_input(source)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed locally")
class CompactCTPMeanTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(630)

    def test_all_40_frames_and_known_temporal_mean_use_correct_axes_and_scale(self):
        x, y, z, t = np.indices((3, 5, 7, 40))
        data = (40 * x + 7 * y + z + t).astype(np.float32)
        series = SyntheticSeries(data)
        model = CompactCTPMean(width=4)
        model.head.weight.data.fill_(1 / 32)
        model.head.bias.data.zero_()
        seen = []

        def controlled_features(frame):
            index = len(seen)
            expected = torch.from_numpy(data[..., index].transpose(2, 1, 0).copy()).unsqueeze(0).unsqueeze(0) / 255
            # NumPy and torch scalar division can round differently in float32.
            torch.testing.assert_close(frame, expected, rtol=1e-6, atol=1e-7)
            seen.append(index)
            return frame.mean().expand(1, 32)

        with patch.object(model.encoder, "forward_features", side_effect=controlled_features), torch.no_grad():
            logit = model(series)
        self.assertEqual(series.calls, list(range(40)))
        self.assertEqual(seen, list(range(40)))
        torch.testing.assert_close(logit, torch.tensor([float(data.mean() / 255)]), rtol=1e-6, atol=1e-7)

    def test_finite_logits_variable_native_shapes_and_recorded_configuration(self):
        model = CompactCTPMean(width=4).eval()
        self.assertIsInstance(model.encoder.classifier, nn.Identity)
        self.assertFalse(any(name.startswith("encoder.classifier.") for name in model.state_dict()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.configuration["intensity_divisor"], 255.0)
        self.assertEqual(model.configuration["frame_count"], 40)
        self.assertEqual(model.configuration["temporal_pooling"], "mean")
        self.assertEqual(model.configuration["embedding_dim"], 32)
        for shape in ((9, 7, 5), (7, 9, 7)):
            series = SyntheticSeries(synthetic_data(shape))
            with self.subTest(shape=shape), torch.no_grad():
                logit = model(series)
                self.assertEqual(tuple(logit.shape), (1,))
                self.assertTrue(torch.isfinite(logit).all().item())
                self.assertEqual(series.calls, list(range(40)))

    def test_real_encoder_mean_is_invariant_to_temporal_permutation(self):
        source = synthetic_data()
        model = CompactCTPMean(width=4).eval()
        with torch.no_grad():
            original = model(SyntheticSeries(source))
            permuted = model(SyntheticSeries(source[..., ::-1].copy()))
        torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)

    def test_streamed_gradients_match_tiny_full_graph_reference(self):
        source = synthetic_data()
        unchanged = source.copy()
        model = CompactCTPMean(width=4)
        reference = CompactCTPMean(width=4)
        reference.load_state_dict(model.state_dict(), strict=True)
        weights_before = {name: p.detach().clone() for name, p in model.named_parameters()}
        target = torch.tensor([1.0])
        reference_series = SyntheticSeries(source)
        # Deliberately tiny test oracle only: do not use this full-graph pattern
        # for native-size CTP. Production backward_series recomputes each frame.
        embeddings = []
        for index in range(40):
            native = reference_series.load_frame(index).input.tensor
            frame = torch.from_numpy(scale_ctp_input(native)).unsqueeze(0)
            embeddings.append(reference.encoder.forward_features(frame))
        reference_logit = reference.head(torch.stack(embeddings).mean(0)).squeeze(1)
        reference_loss = nn.BCEWithLogitsLoss()(reference_logit, target)
        reference_loss.backward()

        series = SyntheticSeries(source)
        events = []
        encoder_forward = model.encoder.forward_features

        def observe(frame):
            embedding = encoder_forward(frame)
            if torch.is_grad_enabled():
                events.append("forward")
                embedding.register_hook(lambda gradient: events.append("backward"))
            return embedding

        with patch.object(model.encoder, "forward_features", side_effect=observe):
            result = model.backward_series(series, target)
        self.assertEqual(events, ["forward", "backward"] * 40)
        self.assertEqual(series.calls, list(range(40)) * 2)
        self.assertFalse(result.logit.requires_grad)
        self.assertFalse(result.loss.requires_grad)
        torch.testing.assert_close(result.logit, reference_logit.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result.loss, reference_loss.detach(), rtol=1e-5, atol=1e-6)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, rtol=5e-4, atol=2e-6)
            torch.testing.assert_close(parameter, weights_before[name], rtol=0, atol=0)
        self.assertGreater(float(model.encoder.stem[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
        np.testing.assert_array_equal(source, unchanged)

    def test_unsafe_forward_and_invalid_series_target_or_frame_are_rejected(self):
        model = CompactCTPMean(width=4)
        series = SyntheticSeries(synthetic_data())
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            model(series)
        self.assertEqual(series.calls, [])
        for target in (torch.tensor([0.5]), torch.tensor([float("nan")]), torch.tensor([1, 0])):
            with self.assertRaisesRegex(ValueError, "target"):
                model.backward_series(series, target)
        wrong_count = SyntheticSeries(synthetic_data()[..., :39])
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "40 frames"):
            model(wrong_count)
        wrong_frame = series.load_frame(1)
        with patch.object(series, "load_frame", return_value=wrong_frame), torch.no_grad():
            with self.assertRaisesRegex(ValueError, "requested adapted"):
                model(series)


@unittest.skipUnless(TORCH_AVAILABLE and NIBABEL_AVAILABLE, "PyTorch and nibabel are required")
class CTPModelNiftiTests(unittest.TestCase):
    def test_opened_synthetic_series_matches_in_memory_adapter_input(self):
        import nibabel as nib
        from ele630.ctp import RegisteredCTPRepository
        from ele630.data_contract import DatasetContract, OutcomePatient

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(630)
        data = synthetic_data()
        patient = OutcomePatient("fictional-ctp-model", "01", "good", False, 0)
        contract = DatasetContract((patient,), (), 1, 0, 0, "synthetic", 630)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctp_path = root / "derivatives" / patient.patient_id / "ses-01" / "fictional_ses-01_space-ncct_ctp.nii.gz"
            ncct_path = root / "raw_data_nifti" / patient.patient_id / "ses-01" / "fictional_ses-01_ncct_skull_stripped.nii.gz"
            ctp_path.parent.mkdir(parents=True)
            ncct_path.parent.mkdir(parents=True)
            nib.save(nib.Nifti1Image(data, AFFINE), ctp_path)
            nib.save(nib.Nifti1Image(np.zeros(data.shape[:3], dtype=np.int16), AFFINE), ncct_path)
            series = RegisteredCTPRepository(contract, root).open_series(patient)
            model = CompactCTPMean(width=4).eval()
            with torch.no_grad(), patch.object(nib.arrayproxy.ArrayProxy, "__array__", side_effect=AssertionError("full series read")):
                loaded = model(series)
                expected = model(SyntheticSeries(data))
            torch.testing.assert_close(loaded, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
