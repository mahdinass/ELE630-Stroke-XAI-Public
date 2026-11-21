"""Compact scratch-initialized 3D residual classifier for native-grid NCCT.

The default is a ResNet-10-style model: a 3D stem, one two-convolution
BasicBlock in each of four stages, global pooling, and one linear logit.
It is defined here explicitly rather than claimed identical to a published
implementation. Input is (N, 1, D, H, W); no image preprocessing occurs.
"""

from __future__ import annotations

from math import gcd

import torch
from torch import nn

from .ncct_model_config import NCCTModelConfig


def _norm(channels: int) -> nn.GroupNorm:
    groups = gcd(channels, 8)
    if groups == channels and groups > 1:
        groups //= 2  # At least two channels per group, even at a 1-voxel feature map.
    return nn.GroupNorm(groups, channels)


class BasicBlock3D(nn.Module):
    """Two padded 3×3×3 convolutions and a projected shortcut when needed."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.norm1 = _norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.norm2 = _norm(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == (1, 1, 1)
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class CompactResNet10(nn.Module):
    """One-channel native-shape classifier with one logit per volume.

    Default width 16 yields stage widths (16, 32, 64, 128). The stem and
    second stage halve H/W; stage 3 halves D/H/W. Stage 4 defaults to
    halving D/H/W; ``stage4_stride=(1,1,1)`` preserves stage-3 resolution.
    Stage 1 has unit
    stride. Kernel padding preserves each stage's intended resolution before
    striding; no input volume is padded or resized by this class. ``pooling``
    defaults to the original average pooling; ``avgmax`` averages global
    average and maximum features before the same classifier.
    """

    def __init__(self, width: int = 16, pooling: str = "avg", *,
                 stage4_stride: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.model_config = NCCTModelConfig(width, pooling, stage4_stride)
        self.width = width
        self.pooling = pooling
        self.stage4_stride = stage4_stride
        self.stem = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            _norm(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = BasicBlock3D(width, width, (1, 1, 1))
        self.stage2 = BasicBlock3D(width, 2 * width, (1, 2, 2))
        self.stage3 = BasicBlock3D(2 * width, 4 * width, (2, 2, 2))
        self.stage4 = BasicBlock3D(4 * width, 8 * width, stage4_stride)
        self.pool = nn.AdaptiveAvgPool3d(1)
        if pooling == "avgmax":
            self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.classifier = nn.Linear(8 * width, 1)
        self._initialize_from_scratch()

    def architecture_record(self) -> dict:
        """Bind semantic configuration to actual topology, including parameter-free choices."""
        config = NCCTModelConfig(self.width, self.pooling, self.stage4_stride)
        if config != self.model_config:
            raise ValueError("model configuration was changed after construction")
        expected_strides = [(1, 1, 1), (1, 2, 2), (2, 2, 2), self.stage4_stride]
        for stage, stride in zip((self.stage1, self.stage2, self.stage3, self.stage4), expected_strides):
            if stage.conv1.stride != stride or stage.conv2.stride != (1, 1, 1):
                raise ValueError("model stride disagrees with architecture configuration")
            if not isinstance(stage.shortcut, nn.Identity) and stage.shortcut[0].stride != stride:
                raise ValueError("projection stride disagrees with architecture configuration")
        if (self.stem[0].stride != (1, 2, 2)
                or not isinstance(self.pool, nn.AdaptiveAvgPool3d)
                or (self.pooling == "avgmax") != hasattr(self, "max_pool")
                or (self.pooling == "avgmax" and not isinstance(self.max_pool, nn.AdaptiveMaxPool3d))):
            raise ValueError("model pooling/stem disagrees with architecture configuration")
        record = config.as_record()
        # Actual topology also detects changed kernels, norms or head dimensions.
        record["modules"] = [
            {"name": name, "type": type(module).__name__, **(
                {"in": module.in_channels, "out": module.out_channels,
                 "kernel": list(module.kernel_size), "stride": list(module.stride),
                 "padding": list(module.padding), "dilation": list(module.dilation),
                 "groups": module.groups, "bias": module.bias is not None}
                if isinstance(module, nn.Conv3d) else
                {"groups": module.num_groups, "channels": module.num_channels,
                 "eps": module.eps, "affine": module.affine}
                if isinstance(module, nn.GroupNorm) else
                {"in": module.in_features, "out": module.out_features,
                 "bias": module.bias is not None}
                if isinstance(module, nn.Linear) else
                {"inplace": module.inplace}
                if isinstance(module, nn.ReLU) else
                {"output_size": module.output_size}
            )}
            for name, module in self.named_modules()
            if isinstance(module, (nn.Conv3d, nn.GroupNorm, nn.Linear, nn.ReLU,
                                   nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool3d))
        ]
        return record

    @property
    def xai_feature_layer(self) -> nn.Conv3d:
        """Final convolution before stage-4 normalization and residual addition."""
        return self.stage4.conv2

    def _initialize_from_scratch(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.GroupNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled (N, 8*width) embedding before the classifier."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim != 5:
            raise ValueError("input must have shape (N, 1, D, H, W)")
        if x.shape[1] != 1:
            raise ValueError("input must have exactly one NCCT channel")
        if x.shape[0] == 0 or any(size == 0 for size in x.shape[2:]):
            raise ValueError("batch and spatial dimensions must be nonempty")
        if not x.is_floating_point():
            raise TypeError("input voxels must have a floating-point dtype")
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if self.pooling == "avg":
            x = self.pool(x).flatten(1)
        else:
            x = (0.5 * (self.pool(x) + self.max_pool(x))).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)).squeeze(1)
