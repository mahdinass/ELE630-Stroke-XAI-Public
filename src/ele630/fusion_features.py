"""Frozen inference only. CTP reuses the established one-frame mean accumulator."""
from __future__ import annotations
import numpy as np
import torch


def freeze(model):
    model.eval()
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    return model


def extract(model, value, modality):
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("the entire imaging model must be frozen in evaluation mode")
    with torch.inference_mode():
        if modality == "ncct":
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.ndim != 4 or value.shape[0] != 1
                    or not value.flags.c_contiguous or not np.isfinite(value).all()
                    or np.min(value) < 0 or np.max(value) > 1):
                raise ValueError("invalid native windowed NCCT input")
            x = torch.from_numpy(value).unsqueeze(0).to(next(model.parameters()).device)
            embedding = model.forward_features(x)
            logit = model.classifier(embedding).reshape(1)
        elif modality == "ctp":
            embedding = model._mean_embedding(value)
            logit = model.head(embedding).reshape(1)
        else:
            raise ValueError("unknown imaging modality")
        if embedding.shape != (1, 128) or not torch.isfinite(embedding).all() or not torch.isfinite(logit).all():
            raise ValueError("invalid frozen imaging representation")
        # Compute the matched imaging score from the very same representation.
        return embedding.cpu().numpy().reshape(128).copy(), float(logit.sigmoid().item())


class DevelopmentFeatures:
    def __init__(self, plan, loader, model, modality):
        from .clinical_baseline import _validate_development_plan
        _validate_development_plan(plan)
        self.allowed = {p.patient_id: p for p in plan.inner_train + plan.inner_validation}
        self.loader, self.model, self.modality = loader, freeze(model), modality

    def load(self, patient):
        if self.allowed.get(getattr(patient, "patient_id", None)) != patient:
            raise ValueError("image request is outside authorized development")
        return extract(self.model, self.loader(patient), self.modality)

    def partition(self, patients):
        values = [self.load(p) for p in patients]
        return np.stack([v[0] for v in values]), np.array([v[1] for v in values])
