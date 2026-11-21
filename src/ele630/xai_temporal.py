"""CTP content sensitivity with detached mean-embedding delta updates."""
from __future__ import annotations

import numpy as np
import torch

from .xai_spatial import require_frozen


def windows():
    return tuple(range(0, 36, 2))+(35,)


def replacement_frames(frame_at, start):
    if start not in windows():
        raise ValueError("unregistered temporal window")
    left, right = start-1, start+5
    a = frame_at(left) if left >= 0 else frame_at(right)
    b = frame_at(right) if right < 40 else a
    for index in range(start, start+5):
        weight = (index-left)/(right-left) if left >= 0 and right < 40 else 0.
        yield index, (1-weight)*a+weight*b


def temporal_occlusion(model, series, score, baseline_support, *, sign=1, baseline_embeddings=None, baseline_mean=None):
    """score(mean) returns raw fused z; sign is the ORIGINAL prediction sign."""
    require_frozen(model)
    model._check_series(series)
    if sign not in (-1, 1):
        raise ValueError("fixed target sign required")
    frame_at = lambda i: model._frame_tensor(series, i)
    with torch.no_grad():
        if (baseline_embeddings is None) != (baseline_mean is None):
            raise ValueError("both private temporal cache values required together")
        if baseline_embeddings is None:
            embeddings = []
            total = torch.zeros(1, 128, device=next(model.parameters()).device)
            for i in range(40):
                e = model.encoder.forward_features(frame_at(i)).detach()
                embeddings.append(e)
                total.add_(e)
            mean = total/40
        else:
            embeddings, mean = baseline_embeddings.detach(), baseline_mean.detach()
            if (embeddings.shape != (40, 1, 128) or mean.shape != (1, 128)
                    or embeddings.device != next(model.parameters()).device or mean.device != embeddings.device
                    or embeddings.dtype != torch.float32 or mean.dtype != torch.float32
                    or not torch.isfinite(embeddings).all() or not torch.isfinite(mean).all()):
                raise ValueError("invalid detached temporal embedding cache")
            total = torch.zeros_like(mean)
            for e in embeddings:
                total.add_(e)
            torch.testing.assert_close(total/40, mean, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sign*score(mean), torch.as_tensor([baseline_support], device=mean.device, dtype=torch.float64),
                                   atol=1e-6, rtol=1e-6)
        drops = []
        profile, counts = np.zeros(40), np.zeros(40, dtype=int)
        for start in windows():
            delta = torch.zeros_like(mean)
            for i, replacement in replacement_frames(frame_at, start):
                delta.add_(model.encoder.forward_features(replacement)-embeddings[i])
            value = float((sign*score(mean+delta/40)).item())
            if not np.isfinite(value):
                raise ValueError("nonfinite temporal counterfactual")
            drop = baseline_support-value
            drops.append(drop)
            profile[start:start+5] += drop
            counts[start:start+5] += 1
        if np.any(counts == 0):
            raise ValueError("incomplete temporal coverage")
    return {"window_starts": list(windows()), "window_drops": drops,
            "profile": (profile/counts).tolist(), "coverage": counts.tolist()}
