"""Read-only CAMs using one spatial graph, frozen stage4.conv2, both methods.

No module backward hooks: tensor autograd avoids in-place ReLU view conflicts.
Only detached CPU feature-grid maps survive each frame recomputation.
"""
from __future__ import annotations

import numpy as np
import torch

from .xai_fused_score import fused_logit_torch, image_gradient, predicted_sign


def require_frozen(model):
    if (any(m.training for m in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise ValueError("XAI requires an entirely frozen eval model with no parameter gradients")


def cam_definitions(activation, gradient):
    if activation.ndim != 5 or activation.shape != gradient.shape or activation.shape[0] != 1:
        raise ValueError("matched one-patient 3D activations and gradients required")
    if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
        raise ValueError("nonfinite activation or gradient")
    signed = {"gradcam": (activation * gradient.mean((2, 3, 4), keepdim=True)).sum(1)[0],
              "hirescam": (activation * gradient).sum(1)[0]}
    return {name: {"signed": value.detach().cpu().numpy().copy(),
                   "positive": value.clamp_min(0).detach().cpu().numpy().copy()}
            for name, value in signed.items()}


def frame_cam(encoder, frame, embedding_gradient):
    """Autograd w.r.t. captured conv output; never accumulate parameter grads."""
    require_frozen(encoder)
    if frame.ndim != 5 or frame.shape[:2] != (1, 1) or not torch.isfinite(frame).all():
        raise ValueError("finite one-patient model tensor required")
    captured = []
    def capture(module, args, output):
        captured.append(output)
    handle = encoder.xai_feature_layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            x = frame.detach().clone().requires_grad_(True)
            embedding = encoder.forward_features(x)
            if len(captured) != 1:
                raise ValueError("selected layer must execute exactly once per spatial pass")
            activation = captured[0]
            if embedding_gradient.shape != (128,):
                raise ValueError("128D image-score gradient required")
            gradient, = torch.autograd.grad(embedding, activation,
                                           grad_outputs=embedding_gradient.to(embedding).reshape(1, 128))
            maps = cam_definitions(activation, gradient)
            return embedding.detach(), maps
    finally:
        handle.remove()
        captured.clear()


def ncct_cam(model, frame, clinical, state, *, sign=None):
    require_frozen(model)
    with torch.no_grad():
        baseline = model.forward_features(frame)
        logit = fused_logit_torch(baseline, clinical, state)
    sign = predicted_sign(logit) if sign is None else sign
    recomputed, maps = frame_cam(model, frame, image_gradient(state, sign))
    torch.testing.assert_close(recomputed, baseline, rtol=1e-6, atol=1e-6)
    for data in maps.values():
        data["poor_signed"] = sign*data["signed"]
    return {"logit": float(logit.item()), "sign": sign, "maps": maps, "embedding": baseline.detach()}


def ctp_cam(model, series, clinical, state, *, sign=None, baseline_embedding=None):
    require_frozen(model)
    with torch.no_grad():
        model._check_series(series)
        baseline = model._mean_embedding(series) if baseline_embedding is None else baseline_embedding.detach()
        if baseline.shape != (1, 128) or not torch.isfinite(baseline).all():
            raise ValueError("finite private baseline embedding required")
        logit = fused_logit_torch(baseline, clinical, state)
    sign = predicted_sign(logit) if sign is None else sign
    # Match float32 mean.backward: cast the score gradient, then divide by40.
    per_frame_gradient = image_gradient(state, sign).to(baseline) / 40
    frames = []
    total = torch.zeros_like(baseline)
    embeddings = []
    for index in range(40):
        embedding, maps = frame_cam(model.encoder, model._frame_tensor(series, index), per_frame_gradient)
        total.add_(embedding)
        embeddings.append(embedding)
        frames.append(maps)
        del embedding, maps
    torch.testing.assert_close(total / 40, baseline, rtol=1e-6, atol=1e-6)
    aggregate = {method: {kind: np.mean([m[method][kind] for m in frames], axis=0)
                          for kind in ("positive", "signed")}
                 for method in ("gradcam", "hirescam")}
    for data in aggregate.values():
        data["poor_signed"] = sign*data["signed"]
    for frame in frames:
        for data in frame.values():
            data["poor_signed"] = sign*data["signed"]
    return {"logit": float(logit.item()), "sign": sign, "maps": aggregate, "frames": frames,
            "embedding": baseline.detach(), "frame_embeddings": torch.stack(embeddings).detach()}
