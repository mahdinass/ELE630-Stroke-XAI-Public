"""Pure NCCT architecture identity and analytical geometry; no image/file access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def record_signature(record: dict) -> str:
    """Canonical JSON SHA-256; this is consistency checking, not authentication."""
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def validate_stage4_stride(value: object) -> tuple[int, int, int]:
    if (type(value) is not tuple or len(value) != 3
            or any(type(item) is not int for item in value)
            or value not in {(1, 1, 1), (2, 2, 2)}):
        raise ValueError("stage4_stride must be (1,1,1) or (2,2,2)")
    return value


@dataclass(frozen=True)
class NCCTModelConfig:
    width: int = 16
    pooling: str = "avg"
    stage4_stride: tuple[int, int, int] = (2, 2, 2)

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 2:
            raise ValueError("width must be an integer of at least 2")
        if type(self.pooling) is not str or self.pooling not in {"avg", "avgmax"}:
            raise ValueError("pooling must be 'avg' or 'avgmax'")
        validate_stage4_stride(self.stage4_stride)

    def as_record(self) -> dict:
        return {
            "schema_version": 1,
            "implementation": "CompactResNet10-stage4-config-v1",
            "width": self.width,
            "pooling": self.pooling,
            "stage4_stride": list(self.stage4_stride),
            "stem_stride": [1, 2, 2],
            "stage_strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], list(self.stage4_stride)],
            "block_counts": [1, 1, 1, 1],
            "kernel_padding": {"main": [3, 1], "projection": [1, 0]},
            "normalization": "GroupNorm-gcd(channels,8)-min-two-channels-per-group",
            "initialization": "scratch-kaiming-fan_out-GN-one-zero-linear-normal-0.01",
            "input_axes": "N,C,D=z,H=y,W=x",
            "output": "one-raw-poor-outcome-logit",
            "parameter_count": 3498 * self.width**2 + 125 * self.width + 1,
        }

    def feature_geometry(self, shape_dhw: tuple[int, int, int]) -> dict:
        if (type(shape_dhw) is not tuple or len(shape_dhw) != 3
                or any(type(n) is not int or n < 1 for n in shape_dhw)):
            raise ValueError("shape_dhw must be three positive integers")
        stride = self.stage4_stride[0]
        jump = (2 * stride, 8 * stride, 8 * stride)
        return {
            "shape_dhw": [(n + step - 1) // step for n, step in zip(shape_dhw, jump)],
            "jump_dhw": list(jump),
            "nominal_receptive_field_dhw": [21 + 4 * stride, 63 + 16 * stride, 63 + 16 * stride],
            "note": "convolutional field only; GroupNorm and pooling have nonlocal dependencies",
        }
