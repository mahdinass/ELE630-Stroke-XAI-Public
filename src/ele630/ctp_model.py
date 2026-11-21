"""CTP temporal mean, mean+std and shallow-TCN representations with streamed backward.

No patient lookup, labels from files, optimizer, split, or artifact writing.
Use a Stage 4O opened series. Forward is inference-only; backward_series
computes one patient's BCE gradients with at most one encoder graph live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ctp_input import CTPFrameInput
from .ctp_intensity import CTP_INTENSITY_DIVISOR, scale_ctp_input
from .resnet3d import CompactResNet10


class _Metadata(Protocol):
    frame_count: int
    spatial_shape: tuple[int, int, int]
    affine: np.ndarray


class _Frame(Protocol):
    input: CTPFrameInput


class CTPSeriesSource(Protocol):
    metadata: _Metadata

    def load_frame(self, frame_index: int) -> _Frame: ...


@dataclass(frozen=True)
class CTPBackwardResult:
    loss: torch.Tensor  # Detached scalar unweighted BCE.
    logit: torch.Tensor  # Detached shape (1,); never logged or written here.


class CompactCTPMean(nn.Module):
    """Shared spatial encoder -> mean of all 40 embeddings -> one linear logit.

    Default width 16 gives 128 features per frame. The encoder is trainable
    from scratch. Its unused NCCT classifier is removed only on this instance.
    One patient is processed at a time, on the model's device, in float32.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        self.encoder = CompactResNet10(width=width, pooling="avg")
        self.encoder.classifier = nn.Identity()
        self.head = nn.Linear(8 * width, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    @property
    def configuration(self) -> dict[str, object]:
        """Data-independent settings for a future private run record."""
        return {
            "encoder": "CompactResNet10", "width": self.encoder.width,
            "embedding_dim": self.head.in_features, "spatial_pooling": "avg",
            "frame_count": 40, "temporal_pooling": "mean",
            "intensity_divisor": CTP_INTENSITY_DIVISOR,
            "intensity_transform": "float32(x) / 255; no clipping",
            "backward_method": "two_pass_frame_recomputation",
        }

    @staticmethod
    def _check_series(series: CTPSeriesSource) -> None:
        if not callable(getattr(series, "load_frame", None)):
            raise ValueError("an opened CTP series with load_frame(index) is required")
        if series.metadata.frame_count != 40:
            raise ValueError("CTP temporal mean requires exactly 40 frames")

    def _frame_tensor(self, series: CTPSeriesSource, index: int) -> torch.Tensor:
        frame = series.load_frame(index).input
        if not isinstance(frame, CTPFrameInput) or frame.frame_index != index:
            raise ValueError("series must return the requested adapted CTP frame")
        shape = tuple(series.metadata.spatial_shape)
        if (frame.native_shape != shape or len(shape) != 3
                or frame.tensor.shape != (1, *reversed(shape))
                or not np.array_equal(frame.native_affine, series.metadata.affine)):
            raise ValueError("adapted CTP frame disagrees with the opened series grid")
        normalized = scale_ctp_input(frame.tensor)
        return torch.from_numpy(normalized).unsqueeze(0).to(self.head.weight.device)

    def _mean_embedding(self, series: CTPSeriesSource) -> torch.Tensor:
        # This routine must never build a graph spanning all 40 encodings.
        if torch.is_grad_enabled():
            raise RuntimeError("temporal accumulation requires torch.no_grad()")
        self._check_series(series)
        total = torch.zeros(1, self.head.in_features, device=self.head.weight.device)
        for index in range(40):
            features = self.encoder.forward_features(self._frame_tensor(series, index))
            if not torch.isfinite(features).all().item():
                raise ValueError("nonfinite CTP frame embedding")
            total.add_(features)
            del features
        return total / 40

    def forward(self, series: CTPSeriesSource) -> torch.Tensor:
        """Return shape (1,) under torch.no_grad(); use backward_series to train."""
        mean = self._mean_embedding(series)
        logit = self.head(mean).squeeze(1)
        if not torch.isfinite(logit).all().item():
            raise ValueError("nonfinite CTP logit")
        return logit

    def backward_series(self, series: CTPSeriesSource, target: torch.Tensor) -> CTPBackwardResult:
        """Compute unweighted BCE gradients; no optimizer step is performed.

        Clears existing gradients. First collect the mean without graphs, then
        differentiate head(mean). Re-encode each frame and backpropagate
        (dL/dmean)/40 immediately, accumulating shared encoder gradients.
        GroupNorm has no running statistics and this encoder has no dropout.
        Inputs/weights must stay unchanged across the two passes. On failure,
        discard partial gradients. AMP, higher-order gradients and concurrent
        parameter updates are outside this primitive's contract.
        """
        if not torch.is_grad_enabled():
            raise RuntimeError("backward_series requires gradient mode")
        if (not isinstance(target, torch.Tensor) or tuple(target.shape) != (1,)
                or target.dtype != torch.float32 or not torch.isfinite(target).all().item()
                or not ((target == 0) | (target == 1)).all().item()):
            raise ValueError("target must be a finite binary float32 tensor of shape (1,)")
        if any(not p.requires_grad for p in self.parameters()):
            raise ValueError("CTP encoder and head must remain trainable")
        self.zero_grad(set_to_none=True)
        try:
            with torch.no_grad():
                mean = self._mean_embedding(series)
            mean.requires_grad_(True)
            logit = self.head(mean).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, target.to(logit.device))
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite CTP loss")
            loss.backward()
            per_frame_gradient = mean.grad.detach() / 40
            if not torch.isfinite(per_frame_gradient).all().item():
                raise ValueError("nonfinite CTP temporal gradient")
            for index in range(40):
                features = self.encoder.forward_features(self._frame_tensor(series, index))
                if not torch.isfinite(features).all().item():
                    raise ValueError("nonfinite CTP frame embedding")
                features.backward(per_frame_gradient)
                del features  # Free this graph before loading the next frame.
            if any(p.grad is None or not torch.isfinite(p.grad).all().item()
                   for p in self.parameters()):
                raise ValueError("missing or nonfinite CTP parameter gradient")
            return CTPBackwardResult(loss.detach(), logit.detach())
        except Exception:
            self.zero_grad(set_to_none=True)
            raise


class CompactCTPShallowTCN(nn.Module):
    """Fixed order-sensitive 40-frame spatial encoder + shallow temporal Conv1d.

    Width 16 and both 32-channel temporal layers are prespecified. Forward is
    inference-only; backward_series retains only detached embedding values and
    their gradients across frame passes, never 40 spatial computation graphs.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        if type(width) is not int or width != 16:
            raise ValueError("CompactCTPShallowTCN requires the fixed width 16")
        self.encoder = CompactResNet10(width=16, pooling="avg")
        self.encoder.classifier = nn.Identity()
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "model_name": "CompactCTPShallowTCN", "model_version": 1,
            "encoder": "CompactResNet10", "width": 16, "embedding_dim": 128,
            "spatial_pooling": "avg", "frame_count": 40,
            "temporal_representation": "shallow_temporal_convolution",
            "temporal_channels": [32, 32], "kernel_sizes": [3, 3],
            "dilations": [1, 2], "paddings": [1, 2], "strides": [1, 1],
            "temporal_bias": True, "padding_mode": "zeros", "activations": ["relu", "relu"],
            "temporal_pooling": "mean", "head_input_dim": 32,
            "order_sensitive": True, "frame_order": "original_indices_0_through_39",
            "temporal_normalization": "none", "temporal_dropout": 0,
            "temporal_initialization": "pytorch_conv1d_default",
            "head_initialization": "normal_std_0.01_zero_bias",
            "intensity_divisor": CTP_INTENSITY_DIVISOR,
            "intensity_transform": "float32(x) / 255; no clipping",
            "backward_method": "two_pass_embedding_chain_rule_frame_recomputation",
        }

    def _frame_tensor(self, series: CTPSeriesSource, index: int) -> torch.Tensor:
        frame = series.load_frame(index).input
        if not isinstance(frame, CTPFrameInput) or frame.frame_index != index:
            raise ValueError("series must return the requested adapted CTP frame")
        shape = tuple(series.metadata.spatial_shape)
        if (frame.native_shape != shape or len(shape) != 3
                or frame.tensor.shape != (1, *reversed(shape))
                or not np.array_equal(frame.native_affine, series.metadata.affine)):
            raise ValueError("adapted CTP frame disagrees with the opened series grid")
        normalized = scale_ctp_input(frame.tensor)
        return torch.from_numpy(normalized).unsqueeze(0).to(self.head.weight.device)

    def _sequence_embeddings(self, series: CTPSeriesSource) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("temporal accumulation requires torch.no_grad()")
        if not callable(getattr(series, "load_frame", None)):
            raise ValueError("an opened CTP series with load_frame(index) is required")
        if series.metadata.frame_count != 40:
            raise ValueError("CTP shallow TCN requires exactly 40 frames")
        embeddings = []
        for index in range(40):
            features = self.encoder.forward_features(self._frame_tensor(series, index))
            if tuple(features.shape) != (1, 128) or not torch.isfinite(features).all().item():
                raise ValueError("CTP shallow TCN requires finite 128-dimensional frame embeddings")
            embeddings.append(features.detach())
            del features
        return torch.stack(embeddings, dim=2)  # (1, 128, 40), original frame order.

    def _temporal_logit(self, embeddings: torch.Tensor) -> torch.Tensor:
        logit = self.head(self.temporal(embeddings).mean(dim=2)).squeeze(1)
        if not torch.isfinite(logit).all().item():
            raise ValueError("nonfinite CTP logit")
        return logit

    def forward(self, series: CTPSeriesSource) -> torch.Tensor:
        """Return shape (1,) under torch.no_grad(); use backward_series to train."""
        return self._temporal_logit(self._sequence_embeddings(series))

    def backward_series(self, series: CTPSeriesSource, target: torch.Tensor) -> CTPBackwardResult:
        """Exact chain-rule gradients; the caller performs the sole optimizer step.

        Differentiate the temporal network/head using a detached sequence leaf.
        Then re-encode frames 0..39 and immediately backpropagate the stored
        dLoss/dEmbedding for each frame. Inputs and weights must stay fixed
        between passes. The encoder has GroupNorm and no stochastic layers.
        AMP, higher-order gradients and concurrent updates are outside scope.
        On failure all partial model gradients are discarded.
        """
        if not torch.is_grad_enabled():
            raise RuntimeError("backward_series requires gradient mode")
        if (not isinstance(target, torch.Tensor) or tuple(target.shape) != (1,)
                or target.dtype != torch.float32 or not torch.isfinite(target).all().item()
                or not ((target == 0) | (target == 1)).all().item()):
            raise ValueError("target must be a finite binary float32 tensor of shape (1,)")
        if any(not parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("CTP encoder, temporal layers and head must remain trainable")
        self.zero_grad(set_to_none=True)
        try:
            with torch.no_grad():
                embeddings = self._sequence_embeddings(series)
            embeddings.requires_grad_(True)
            logit = self._temporal_logit(embeddings)
            loss = F.binary_cross_entropy_with_logits(logit, target.to(logit.device))
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite CTP loss")
            loss.backward()
            frame_gradients = embeddings.grad.detach()
            if not torch.isfinite(frame_gradients).all().item():
                raise ValueError("nonfinite CTP temporal gradient")
            for index in range(40):
                features = self.encoder.forward_features(self._frame_tensor(series, index))
                if tuple(features.shape) != (1, 128) or not torch.isfinite(features).all().item():
                    raise ValueError("invalid CTP frame embedding during recomputation")
                features.backward(frame_gradients[:, :, index])
                del features  # Release this spatial graph before the next frame.
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
                   for parameter in self.parameters()):
                raise ValueError("missing or nonfinite CTP parameter gradient")
            return CTPBackwardResult(loss.detach(), logit.detach())
        except Exception:
            self.zero_grad(set_to_none=True)
            raise


class CompactCTPMeanStd(nn.Module):
    """Shared encoder -> population mean and std of 40 embeddings -> logit.

    Only the 40 small, detached embeddings are retained across the two passes.
    Encoder computation graphs are released after each frame's backward call.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        self.encoder = CompactResNet10(width=width, pooling="avg")
        self.encoder.classifier = nn.Identity()
        self.head = nn.Linear(16 * width, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _check_series(series: CTPSeriesSource) -> None:
        if not callable(getattr(series, "load_frame", None)):
            raise ValueError("an opened CTP series with load_frame(index) is required")
        if series.metadata.frame_count != 40:
            raise ValueError("CTP temporal mean+std requires exactly 40 frames")

    def _frame_tensor(self, series: CTPSeriesSource, index: int) -> torch.Tensor:
        frame = series.load_frame(index).input
        if not isinstance(frame, CTPFrameInput) or frame.frame_index != index:
            raise ValueError("series must return the requested adapted CTP frame")
        shape = tuple(series.metadata.spatial_shape)
        if (frame.native_shape != shape or len(shape) != 3
                or frame.tensor.shape != (1, *reversed(shape))
                or not np.array_equal(frame.native_affine, series.metadata.affine)):
            raise ValueError("adapted CTP frame disagrees with the opened series grid")
        normalized = scale_ctp_input(frame.tensor)
        return torch.from_numpy(normalized).unsqueeze(0).to(self.head.weight.device)

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "encoder": "CompactResNet10", "width": self.encoder.width,
            "embedding_dim": 8 * self.encoder.width, "head_input_dim": self.head.in_features,
            "spatial_pooling": "avg", "frame_count": 40,
            "temporal_pooling": "mean_std_population", "std_correction": 0,
            "intensity_divisor": CTP_INTENSITY_DIVISOR,
            "intensity_transform": "float32(x) / 255; no clipping",
            "backward_method": "two_pass_frame_recomputation",
        }

    def _temporal_statistics(self, series: CTPSeriesSource) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError("temporal accumulation requires torch.no_grad()")
        self._check_series(series)
        embeddings = []
        for index in range(40):
            features = self.encoder.forward_features(self._frame_tensor(series, index))
            if not torch.isfinite(features).all().item():
                raise ValueError("nonfinite CTP frame embedding")
            embeddings.append(features.detach())
            del features
        stacked = torch.stack(embeddings)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item():
            raise ValueError("nonfinite CTP temporal statistics")
        return stacked, mean, std

    def forward(self, series: CTPSeriesSource) -> torch.Tensor:
        """Return one logit under torch.no_grad(); use backward_series to train."""
        _, mean, std = self._temporal_statistics(series)
        logit = self.head(torch.cat((mean, std), dim=1)).squeeze(1)
        if not torch.isfinite(logit).all().item():
            raise ValueError("nonfinite CTP logit")
        return logit

    def backward_series(self, series: CTPSeriesSource, target: torch.Tensor) -> CTPBackwardResult:
        """Differentiate exact population mean/std using one encoder graph at a time.

        For N=40, dL/de_t = g_mean/N + g_std*(e_t-mean)/(N*std) for std>0.
        PyTorch's population-std backward is zero at zero variance; those
        dimensions use zero std contribution, with no configurable epsilon.
        """
        if not torch.is_grad_enabled():
            raise RuntimeError("backward_series requires gradient mode")
        if (not isinstance(target, torch.Tensor) or tuple(target.shape) != (1,)
                or target.dtype != torch.float32 or not torch.isfinite(target).all().item()
                or not ((target == 0) | (target == 1)).all().item()):
            raise ValueError("target must be a finite binary float32 tensor of shape (1,)")
        if any(not p.requires_grad for p in self.parameters()):
            raise ValueError("CTP encoder and head must remain trainable")
        self.zero_grad(set_to_none=True)
        try:
            with torch.no_grad():
                embeddings, mean, std = self._temporal_statistics(series)
            pooled = torch.cat((mean, std), dim=1).detach().requires_grad_(True)
            logit = self.head(pooled).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, target.to(logit.device))
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite CTP loss")
            loss.backward()
            mean_gradient, std_gradient = pooled.grad.detach().split(8 * self.encoder.width, dim=1)
            positive = std > 0
            denominator = torch.where(positive, std, torch.ones_like(std))
            for index in range(40):
                frame_gradient = (mean_gradient / 40 +
                                  torch.where(positive,
                                              std_gradient * (embeddings[index] - mean) / (40 * denominator),
                                              torch.zeros_like(std_gradient)))
                if not torch.isfinite(frame_gradient).all().item():
                    raise ValueError("nonfinite CTP temporal gradient")
                features = self.encoder.forward_features(self._frame_tensor(series, index))
                if not torch.isfinite(features).all().item():
                    raise ValueError("nonfinite CTP frame embedding")
                features.backward(frame_gradient)
                del features  # Free this encoder graph before the next frame.
            if any(p.grad is None or not torch.isfinite(p.grad).all().item()
                   for p in self.parameters()):
                raise ValueError("missing or nonfinite CTP parameter gradient")
            return CTPBackwardResult(loss.detach(), logit.detach())
        except Exception:
            self.zero_grad(set_to_none=True)
            raise
