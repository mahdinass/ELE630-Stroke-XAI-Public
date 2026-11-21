"""Float64 fixed scaler/PCA/logistic inference with gradients to float32 CNNs."""
from __future__ import annotations

import torch

from .xai_reconstruction import validate_state


def torch_state(state, device):
    return {k: torch.as_tensor(v, dtype=torch.float64, device=device)
            for k, v in validate_state(state).items()}


def logit_components(embedding, clinical, state):
    """Exact additive logit terms. No whitening or post-PCA scaling."""
    if embedding.shape != (1, 128) or clinical.shape != (1, 11):
        raise ValueError("one 128D embedding and 11D clinical vector required")
    if not torch.isfinite(embedding).all() or not torch.isfinite(clinical).all():
        raise ValueError("finite fused inputs required")
    if set(state) != {"mean", "scale", "pca_mean", "components", "coef", "intercept"}:
        raise ValueError("invalid fused state")
    if any(v.device != embedding.device or v.dtype != torch.float64 for v in state.values()):
        raise ValueError("fusion state must be float64 on the CNN device")
    u = (embedding.to(torch.float64) - state["mean"]) / state["scale"]
    pcs = (u - state["pca_mean"]) @ state["components"].T
    image = (pcs * state["coef"][:8]).sum(1)
    clinical_term = (clinical.to(device=embedding.device, dtype=torch.float64) * state["coef"][8:]).sum(1)
    return {"intercept": state["intercept"], "image": image, "clinical": clinical_term}


def fused_logit_torch(embedding, clinical, state):
    parts = logit_components(embedding, clinical, state)
    return parts["intercept"] + parts["image"] + parts["clinical"]


def predicted_sign(logit):
    """p >= .5, equivalently z >= 0. Hold this sign for ALL interventions."""
    if logit.numel() != 1 or not torch.isfinite(logit).all():
        raise ValueError("one finite baseline logit required")
    return 1 if logit.detach().item() >= 0 else -1


def image_gradient(state, sign=1):
    if sign not in (-1, 1):
        raise ValueError("target sign must be fixed at -1 or +1")
    return sign * (state["coef"][:8] @ state["components"]) / state["scale"]
