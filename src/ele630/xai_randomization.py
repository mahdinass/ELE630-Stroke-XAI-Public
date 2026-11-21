"""Controlled image-score randomization on separate copies, fixed targets."""
from __future__ import annotations

import copy
import numpy as np


def randomized_copy(model, state, *, upper_block=False, seed=630):
    import torch
    from torch import nn
    from .xai_spatial import require_frozen
    require_frozen(model)
    result = copy.deepcopy(model)
    result_state = {k: v.detach().clone() for k, v in state.items()}
    coef = result_state["coef"][:8]
    rms = float(coef.square().mean().sqrt().item())
    if rms <= 1e-12:
        raise ValueError("zero_head")
    values = np.random.default_rng(seed).normal(0, rms, 8)
    result_state["coef"][:8] = torch.as_tensor(values, device=coef.device, dtype=coef.dtype)
    if upper_block:
        encoder = getattr(result, "encoder", result)
        devices = [next(result.parameters()).device.index] if next(result.parameters()).is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for layer in encoder.stage4.modules():
                if isinstance(layer, nn.Conv3d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.GroupNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
    return result, result_state
