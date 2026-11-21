"""Development-only access and selected-checkpoint metrics; never outer scoring."""

from __future__ import annotations

import math
import hashlib

import numpy as np
import torch

from .ctp_metrics import aggregate_ctp_metrics
from .ncct_checkpoint import load_ncct_checkpoint
from .ncct_model_config import record_signature
from .ncct_splits import validate_fold_plan
from .ncct_training import _tensor_pair
from .resnet3d import CompactResNet10


class DevelopmentNCCTSamples:
    """Authorize membership before delegating any image access."""

    def __init__(self, plan, samples):
        validate_fold_plan(plan)
        if not callable(getattr(samples, "load", None)):
            raise TypeError("samples must provide load(patient)")
        self._allowed = {p.patient_id: p for p in plan.inner_train + plan.inner_validation}
        self._samples = samples
        self._input_digests: dict[str, str] = {}

    @property
    def input_record(self) -> dict:
        return dict(getattr(self._samples, "input_record", {"status": "not_recorded"}))

    def load(self, patient):
        if self._allowed.get(getattr(patient, "patient_id", None)) != patient:
            raise ValueError("NCCT sample request is outside authorized development")
        sample = self._samples.load(patient)
        image = getattr(sample, "image", None)
        if not isinstance(image, np.ndarray) or not image.flags.c_contiguous:
            raise ValueError("NCCT development sample must have a contiguous image")
        digest = hashlib.sha256(str((image.shape, str(image.dtype))).encode("ascii"))
        digest.update(memoryview(image))
        current = digest.hexdigest()
        previous = self._input_digests.get(patient.patient_id)
        if previous is not None and current != previous:
            raise ValueError("NCCT development input changed between registered loads")
        self._input_digests[patient.patient_id] = current
        return sample

    @property
    def input_consistency_signature(self) -> str:
        """Private aggregate identity of the checked inputs, never raw case records."""
        return record_signature(self._input_digests)


def evaluate_selected_development(plan, samples, result, *, device: str = "cpu") -> dict:
    """Score inner validation once, returning aggregates and paired prior controls.

    Run metadata must be the registered training result, not inferred from an
    arbitrary checkpoint. No outer images or labels are scored. Individual
    predictions are transient; optional uncertainty analysis is a separate
    registered development analysis, not supplied by this aggregate function.
    """
    from .ncct_training import _membership_record

    validate_fold_plan(plan)
    record = result.run_record
    if (record.get("outer_fold") != plan.outer_fold
            or record.get("membership") != _membership_record(plan)
            or record.get("split_seed") != plan.seed
            or record.get("validation_fraction") != plan.validation_fraction):
        raise ValueError("NCCT evaluation membership disagrees with registered training")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    target_device = torch.device(device)
    architecture = record["architecture"]
    model = CompactResNet10(
        width=architecture["width"], pooling=architecture["pooling"],
        stage4_stride=tuple(architecture["stage4_stride"]),
    ).to(target_device)
    load_ncct_checkpoint(model, result.checkpoint_path, expected_run_record=record,
                         expected_epoch=result.best_epoch, map_location=target_device)
    model.eval()
    guarded = DevelopmentNCCTSamples(plan, samples)
    labels, scores, losses, prior_losses, logits = [], [], [], [], []
    prior = sum(p.outcome == "poor" for p in plan.inner_train) / len(plan.inner_train)
    with torch.no_grad():
        for patient in plan.inner_validation:
            try:
                sample = guarded.load(patient)
            except Exception:
                raise RuntimeError("NCCT validation sample loading failed") from None
            x, y = _tensor_pair(sample, patient, target_device)
            logit = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            if logit.shape != (1,) or not torch.isfinite(logit).all().item() or not torch.isfinite(loss).item():
                raise ValueError("NCCT validation produced nonfinite or invalid predictions")
            labels.append(int(y.item()))
            logits.append(float(logit.item()))
            scores.append(float(logit.sigmoid().item()))
            losses.append(float(loss.item()))
            # Float64 prior BCE, independent of validation prevalence.
            prior_losses.append(-math.log(prior if labels[-1] else 1 - prior))
    return {
        "partition": "inner_validation", "outer_test_evaluated": False,
        "model": aggregate_ctp_metrics(labels, scores, losses).as_dict(),
        "training_prior": aggregate_ctp_metrics(labels, [prior] * len(labels), prior_losses).as_dict(),
        "selected_epoch": result.best_epoch,
        "probability_sd": float(np.std(scores)),
        "probability_range": [min(scores), max(scores)],
        "logit_sd": float(np.std(logits)),
        "logit_range": [min(logits), max(logits)],
    }
