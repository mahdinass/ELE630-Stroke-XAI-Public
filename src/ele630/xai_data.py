"""Restricted original-development adapters. Real access only from guarded runner."""
from __future__ import annotations

import numpy as np

from . import fusion_runtime as original
from .xai_geometry import validate_labels
from .xai_masks import MASK_PATTERN, resolve_mask, MaskPathError


class ValidationAccess:
    """Restrict cases and mask paths before any loader invocation."""
    def __init__(self, plan, loader):
        from .clinical_baseline import _validate_development_plan
        _validate_development_plan(plan)
        self.allowed = {p.patient_id: p for p in plan.inner_validation}
        self.loader = loader

    def check(self, patient):
        if self.allowed.get(getattr(patient, "patient_id", None)) != patient:
            raise ValueError("XAI case is outside original development validation")

    def load(self, patient):
        self.check(patient)
        return self.loader(patient)

    def labels(self, patient, shape_xyz, affine):
        self.check(patient)
        if not patient.has_gt:
            return None, "missing_mask"
        import nibabel as nib
        try:
            path = resolve_mask(original.DATASET, patient)
        except MaskPathError as error:
            if error.kind == "unsafe_path":
                raise
            return None, "missing_mask" if error.kind in ("missing_directory", "missing_expected_name") else "invalid_mask"
        try:
            image = nib.load(str(path))
            if (len(image.shape) != 3 or tuple(image.shape) != tuple(shape_xyz)
                    or not np.allclose(image.affine, affine, rtol=0, atol=1e-5)):
                return None, "invalid_mask"
            labels = np.asarray(image.dataobj)
            return validate_labels(labels, image.affine, shape_xyz, affine), None
        except (OSError, ValueError, nib.filebasedimages.ImageFileError):
            return None, "invalid_mask"


def real_validation_access(contract, plan, modality, device):
    import torch
    if modality == "ncct":
        from .ncct import NCCTRepository
        from .native_ncct_input import prepare_native_ncct_input
        from .ncct_intensity import clip_scale_ncct
        repository = NCCTRepository(contract, original.DATASET)
        def loader(patient):
            volume = repository.load(patient)
            native = prepare_native_ncct_input(volume.data, volume.affine)
            image = clip_scale_ncct(native.tensor, lower=-20, upper=120)
            return torch.from_numpy(image).unsqueeze(0).to(device), volume.affine, volume.shape
    elif modality == "ctp":
        from .ctp import RegisteredCTPRepository
        repository = RegisteredCTPRepository(contract, original.DATASET)
        def loader(patient):
            value = repository.open_series(patient)
            return value, value.metadata.affine, value.metadata.spatial_shape
    else:
        raise ValueError("fixed modality required")
    return ValidationAccess(plan, loader)


def cached_development_features(plan, loader, model, *, resources=None):
    """Original DevelopmentFeatures membership/extract contract, one cache/load."""
    from .fusion_features import DevelopmentFeatures, extract
    from .xai_ctp_cache import cached_series
    class CachedDevelopmentFeatures(DevelopmentFeatures):
        def load(self, patient):
            if self.allowed.get(getattr(patient, "patient_id", None)) != patient:
                raise ValueError("image request is outside authorized development")
            with cached_series(self.loader(patient), resources=resources) as value:
                return extract(self.model, value, "ctp")
    return CachedDevelopmentFeatures(plan, loader, model, "ctp")


def real_ctp_features(contract, plan, model, *, resources=None):
    from .ctp import RegisteredCTPRepository
    repository = RegisteredCTPRepository(contract, original.DATASET)
    source = cached_development_features(plan, repository.open_series, model.to("cuda"), resources=resources)
    train, _ = source.partition(plan.inner_train)
    validation, probabilities = source.partition(plan.inner_validation)
    return train, validation, probabilities
