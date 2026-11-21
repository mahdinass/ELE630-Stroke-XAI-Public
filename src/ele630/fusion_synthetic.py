"""Fictional inputs for source tests and the allocated-GPU smoke job."""
from __future__ import annotations
import numpy as np
from .data_contract import OutcomePatient
from .ncct_splits import FoldPlan
from .clinical_preprocessing import FEATURE_NAMES


def fictional_plan():
    def patients(part, count, fold):
        return tuple(OutcomePatient(f"fictional-{part}-{i:03d}", "fictional",
                                    "poor" if i % 3 == 0 else "good", False, fold) for i in range(count))
    return FoldPlan(0, 630, .20, patients("train", 18, 1), patients("validation", 6, 2),
                    patients("outer", 3, 0), ())


def fictional_rows(plan):
    rows = {}
    for i, p in enumerate(plan.inner_train + plan.inner_validation):
        values = (30 + i, "F" if i % 2 else "M", i % 35, i % 4,
                  f"{i % 5}:30", i % 2, (i // 2) % 2, i % 2, i % 3, (i // 3) % 2, i % 2)
        rows[p.patient_id] = dict(zip(FEATURE_NAMES, values))
    return rows


class FictionalCTP:
    """One synthetic frame at a time, using the real CTP input adapter."""
    def __init__(self, seed, shape=(4, 16, 16)):
        from types import SimpleNamespace
        self.seed, self.shape, self.reads = seed, shape, []
        self.metadata = SimpleNamespace(frame_count=40, spatial_shape=tuple(reversed(shape)), affine=np.eye(4))

    def load_frame(self, index):
        from types import SimpleNamespace
        from .ctp_input import prepare_ctp_frame_input
        if index not in range(40):
            raise ValueError("invalid fictional frame")
        self.reads.append(index)
        frame = np.random.default_rng(self.seed + index).uniform(0, 255, tuple(reversed(self.shape))).astype(np.float32)
        return SimpleNamespace(input=prepare_ctp_frame_input(frame, self.metadata.affine, index))


def synthetic_embeddings(modality, device):
    import torch
    from .fusion_features import DevelopmentFeatures
    from .resnet3d import CompactResNet10
    from .ctp_model import CompactCTPMean
    from .ncct_training import _seed_everything, _parameter_digest
    plan = fictional_plan()
    _seed_everything(630)
    model = (CompactResNet10(width=16) if modality == "ncct" else CompactCTPMean(width=16)).to(device)
    before = _parameter_digest(model)
    patients = plan.inner_train + plan.inner_validation
    indices = {p.patient_id: i for i, p in enumerate(patients)}
    reads = []
    def loader(p):
        i = indices[p.patient_id]
        if modality == "ncct":
            return np.random.default_rng(630 + i).uniform(0, 1, (1, 4, 16, 16)).astype(np.float32)
        series = FictionalCTP(630 + i * 100)
        reads.append(series)
        return series
    source = DevelopmentFeatures(plan, loader, model, modality)
    train, _ = source.partition(plan.inner_train)
    validation, probabilities = source.partition(plan.inner_validation)
    first = source.load(plan.inner_validation[0])
    if not np.array_equal(first[0], validation[0]) or first[1] != probabilities[0]:
        raise ValueError("nondeterministic frozen inference")
    if before != _parameter_digest(model) or any(p.grad is not None for p in model.parameters()):
        raise ValueError("frozen model changed")
    if modality == "ctp" and any(series.reads != list(range(40)) for series in reads):
        raise ValueError("CTP streaming order violated")
    try:
        source.load(plan.outer_test[0])
    except ValueError:
        pass
    else:
        raise ValueError("outer access guard failed")
    return plan, train, validation, probabilities
