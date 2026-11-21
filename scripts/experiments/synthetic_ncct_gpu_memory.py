#!/usr/bin/env python3
"""Synthetic-only paired full-step profiling, including AdamW state.

CUDA is the default, with no CPU fallback. Explicit CPU mode defaults to a
small tensor and reports no GPU/CPU peak-memory claim. No data/weight files.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ele630.ncct_model_config import NCCTModelConfig, record_signature
from ele630.ncct_training import _parameter_digest, _seed_everything
from ele630.resnet3d import CompactResNet10

MIB = 1024 * 1024
SHAPES = ((8, 32, 32), (30, 512, 512), (59, 603, 512))


def _sanitized_error(exc: Exception) -> str:
    """Bound diagnostics, remove control characters and redact absolute paths."""
    message = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(exc))
    message = re.sub(r"(?<!\w)(?:[A-Za-z]:[\\/]|/|\\\\)[^\s\"']+", "<path>", message)
    return " ".join(message.split())[:500]


def analytical_case(shape: tuple[int, int, int], stride: tuple[int, int, int]) -> dict:
    config = NCCTModelConfig(stage4_stride=stride)
    geometry = config.feature_geometry(shape)
    count = config.as_record()["parameter_count"]
    return {
        "shape_dhw": shape, "stage4_stride": stride, "parameter_count": count,
        "parameter_mib": count * 4 / MIB,
        "parameter_gradient_adam_moments_mib": count * 16 / MIB,
        "input_mib": math.prod(shape) * 4 / MIB,
        "one_stage4_feature_map_mib": math.prod(geometry["shape_dhw"]) * 128 * 4 / MIB,
        "feature_geometry": geometry,
        "note": "storage arithmetic, not peak runtime memory or GPU feasibility",
    }


def _run_case(shape, stride, device, steps, seed) -> bool:
    gpu = device.type == "cuda"
    model = optimizer = inputs = target = logits = loss = None
    record = analytical_case(shape, stride)
    record.update({"device": str(device), "status": "error", "seed": seed,
                   "torch": str(torch.__version__), "measured_steps": steps,
                   "warmup_steps": 1, "peak_allocated_mib": None, "peak_reserved_mib": None})
    started = time.perf_counter()
    stage = "seed"

    def synchronize():
        if gpu:
            torch.cuda.synchronize(device)

    try:
        _seed_everything(seed)
        if gpu:
            stage = "cuda_allocator_initialization"
            # reset_peak_memory_stats does not lazy-initialize CUDA in PyTorch
            # 2.3.1. A nonempty allocation initializes CUDA/the allocator without
            # consuming RNG state; release it before the measurement window.
            probe = torch.empty(1, dtype=torch.uint8, device=device)
            del probe
            synchronize()
            torch.cuda.empty_cache()
            stage = "cuda_peak_reset"
            torch.cuda.reset_peak_memory_stats(device)
            stage = "cuda_device_properties"
            properties = torch.cuda.get_device_properties(device)
            record.update(gpu_name=properties.name, gpu_total_mib=properties.total_memory / MIB)
        stage = "model_setup"
        model = CompactResNet10(stage4_stride=stride).float().to(device).train()
        record["architecture_signature"] = record_signature(model.architecture_record())
        record["initial_parameters_sha256"] = _parameter_digest(model)
        stage = "optimizer_setup"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        stage = "input_setup"
        inputs = torch.rand((1, 1, *shape), device=device, dtype=torch.float32)
        target = torch.tensor([1.0], device=device)

        def step():
            nonlocal logits, loss
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.BCEWithLogitsLoss()(logits, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError("nonfinite synthetic loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise RuntimeError("invalid synthetic gradients")
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise RuntimeError("nonfinite synthetic parameters")

        stage = "warmup_step"
        step()
        synchronize()
        timing = time.perf_counter()
        stage = "training_steps"
        for _ in range(steps):
            step()
        synchronize()
        record["seconds_per_full_step"] = (time.perf_counter() - timing) / steps
        stage = "validation_forwards"
        model.eval()
        timing = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                if not torch.isfinite(model(inputs)).all().item():
                    raise RuntimeError("nonfinite synthetic validation")
        synchronize()
        record["seconds_per_validation_forward"] = (time.perf_counter() - timing) / steps
        if gpu:
            stage = "cuda_memory_statistics"
            record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / MIB
            record["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / MIB
        record["status"] = "pass"
    except Exception as exc:
        record["status"] = "oom" if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) else "error"
        record["error_type"] = type(exc).__name__
        record["error_stage"] = stage
        record["error_message"] = _sanitized_error(exc)
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        loss = logits = target = inputs = optimizer = model = None
        gc.collect()
        if gpu:
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                if record["status"] == "pass":
                    record.update(status="error", error_stage="cuda_cleanup",
                                  error_type=type(exc).__name__, error_message=_sanitized_error(exc))
                else:
                    record.update(cleanup_error_type=type(exc).__name__,
                                  cleanup_error_message=_sanitized_error(exc))
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record["status"] == "pass"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--shape", nargs=3, type=int, action="append", metavar=("D", "H", "W"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=630)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--analytical-only", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.seed < 0 or args.threads < 1:
        parser.error("steps/threads must be positive and seed nonnegative")
    shapes = tuple(tuple(shape) for shape in args.shape) if args.shape else (
        SHAPES if args.device == "cuda" or args.analytical_only else (SHAPES[0],))
    for shape in shapes:
        NCCTModelConfig().feature_geometry(shape)
    if args.analytical_only:
        for shape in shapes:
            for stride in ((2, 2, 2), (1, 1, 1)):
                print(json.dumps(analytical_case(shape, stride), sort_keys=True))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable: synthetic pilot refused CPU fallback", file=sys.stderr)
        return 2
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    for shape in shapes:
        for stride in ((2, 2, 2), (1, 1, 1)):
            if not _run_case(shape, stride, device, args.steps, args.seed):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
