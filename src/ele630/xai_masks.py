"""Read-only validation-only anatomical path/header preflight. Never loads voxels."""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np

from .geometry import _affine

MASK_PATTERN = "*_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz"
MASK_STATUSES = ("manifest_absent", "valid_header", "invalid_geometry", "unreadable_header",
                 "missing_directory", "missing_expected_name", "ambiguous_path", "unsafe_path", "invalid_image_header")


class MaskPathError(ValueError):
    def __init__(self, kind):
        super().__init__("anatomical reference path check failed")
        self.kind = kind


def _safe_path(path, parent):
    if not path.resolve().is_relative_to(parent.resolve()):
        raise MaskPathError("unsafe_path")
    current = parent
    if current.is_symlink():
        raise MaskPathError("unsafe_path")
    for part in path.relative_to(parent).parts:
        current /= part
        if current.is_symlink():
            raise MaskPathError("unsafe_path")


def resolve_mask(dataset, patient):
    """Caller must authorize canonical validation membership before this call."""
    if not patient.has_gt:
        return None
    pid = patient.patient_id
    if not pid or pid in (".", "..") or "/" in pid or "\\" in pid:
        raise MaskPathError("unsafe_path")
    dataset = Path(dataset)
    parent = dataset/"derivatives"/pid
    _safe_path(parent, dataset)
    if not parent.is_dir():
        raise MaskPathError("missing_directory")
    paths = sorted(parent.rglob(MASK_PATTERN))
    if not paths:
        raise MaskPathError("missing_expected_name")
    if len(paths) != 1:
        raise MaskPathError("ambiguous_path")
    _safe_path(paths[0], dataset)
    if not paths[0].is_file():
        raise MaskPathError("unsafe_path")
    return paths[0]


def read_geometry_header(path):
    """At most348 UNCOMPRESSED NIfTI-1 bytes; no array proxy/dataobj exists.

    nibabel supplies the same sform/qform interpretation as real loaders. Its
    parser sees only an in-memory header, making any voxel access impossible.
    NIfTI-2/unsupported headers fail explicitly, never infer a geometry.
    """
    import nibabel as nib
    import warnings
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(348)
    if len(raw) != 348:
        raise ValueError("incomplete anatomical header")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw), check=True)
        shape = tuple(int(n) for n in header.get_data_shape())
        affine = np.asarray(header.get_best_affine(), dtype=np.float64)
    if caught or not shape or any(n <= 0 for n in shape) or (int(header["sform_code"]) <= 0 and int(header["qform_code"]) <= 0):
        raise ValueError("unsupported anatomical header geometry")
    _affine(affine)
    return shape, affine


def _alternative_candidate(dataset, patient, plan):
    """Bounded NAME-ONLY diagnosis; never return or load an alternative mask.

    Inspect only this validation patient's expected/known raw directory and
    derivatives/<collection>/<validation-id>. Never enter another subject.
    Unknown deeper layouts remain unresolved, not claimed to be true absence.
    """
    roots = [dataset/"derivatives"/patient.patient_id, dataset/"raw_data_nifti"/patient.patient_id]
    derivatives = dataset/"derivatives"
    all_subjects = {p.patient_id for p in plan.inner_train+plan.inner_validation+plan.outer_test}
    limited = False
    if derivatives.is_dir() and not derivatives.is_symlink():
        for i, item in enumerate(derivatives.iterdir()):
            if i >= 256:
                limited = True
                break
            if item.name not in all_subjects and not item.name.startswith("sub-") and item.is_dir() and not item.is_symlink():
                roots.append(item/patient.patient_id)
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        try:
            _safe_path(root, dataset)
        except MaskPathError:
            continue
        # Bounded DFS does not follow symlinks or inspect other patient folders.
        pending, seen = [(root, 0)], 0
        while pending:
            directory, depth = pending.pop()
            for item in directory.iterdir():
                seen += 1
                if seen > 2048:
                    limited = True
                    pending.clear()
                    break
                if item.is_symlink():
                    continue
                if item.is_dir() and depth < 4 and item.name not in all_subjects and not item.name.startswith("sub-"):
                    pending.append((item, depth+1))
                elif item.is_file() and item.name.endswith((".nii", ".nii.gz")) and any(word in item.name.lower() for word in ("mask", "msk", "lesion")):
                    return True, limited
    return False, limited


def mask_header_preflight(plan, dataset, image_path, *, header_reader=read_geometry_header, include_training_geometry=False):
    """image_path(patient, modality) resolves ONLY headers after membership check."""
    from .xai_data import ValidationAccess
    access = ValidationAccess(plan, lambda p: (_ for _ in ()).throw(AssertionError("no image loader")))
    counts = dict.fromkeys(MASK_STATUSES, 0)
    alternatives, limited, resolved = 0, 0, 0
    maximum = {m: [0, 0, 0] for m in ("ncct", "ctp")}
    training_checked, invalid_training = 0, 0
    if include_training_geometry:
        # Resource sizing covers reconstruction on train92 as well as val23.
        # No training mask is sought and no outer patient is sent to a resolver.
        for patient in plan.inner_train:
            try:
                for modality in ("ncct", "ctp"):
                    shape, _ = header_reader(image_path(patient, modality))
                    if len(shape) != (3 if modality == "ncct" else 4) or (modality == "ctp" and shape[-1] != 40):
                        raise ValueError("invalid training imaging dimensions")
                    maximum[modality] = [max(a, int(b)) for a, b in zip(maximum[modality], reversed(shape[:3]))]
                training_checked += 1
            except (OSError, ValueError, EOFError):
                invalid_training += 1
    for patient in plan.inner_validation:
        access.check(patient)
        geometry = {}
        try:
            for modality in ("ncct", "ctp"):
                shape, affine = header_reader(image_path(patient, modality))
                if len(shape) != (3 if modality == "ncct" else 4) or (modality == "ctp" and shape[-1] != 40):
                    raise ValueError("invalid imaging dimensions")
                geometry[modality] = (tuple(shape[:3]), affine)
                maximum[modality] = [max(a, int(b)) for a, b in zip(maximum[modality], reversed(shape[:3]))]
        except (OSError, ValueError, EOFError):
            counts["invalid_image_header"] += 1
            continue
        try:
            path = resolve_mask(dataset, patient)
        except MaskPathError as error:
            counts[error.kind] += 1
            if error.kind in ("missing_directory", "missing_expected_name"):
                found, truncated = _alternative_candidate(Path(dataset), patient, plan)
                alternatives += int(found)
                limited += int(truncated)
            continue
        if path is None:
            counts["manifest_absent"] += 1
            continue
        resolved += 1
        try:
            shape, affine = header_reader(path)
        except (OSError, ValueError, EOFError):
            counts["unreadable_header"] += 1
            continue
        if len(shape) != 3 or any(shape != expected_shape or not np.allclose(affine, expected_affine, atol=1e-5, rtol=0)
                                  for expected_shape, expected_affine in geometry.values()):
            counts["invalid_geometry"] += 1
        else:
            counts["valid_header"] += 1
    unresolved = sum(counts[k] for k in ("missing_directory", "missing_expected_name", "ambiguous_path", "unsafe_path", "invalid_image_header"))
    expected = sum(p.has_gt for p in plan.inner_validation)
    return {"validation_count": len(plan.inner_validation), "expected_masks": expected, "resolved_expected_paths": resolved,
            "counts": counts, "alternate_name_or_layout_candidates": alternatives, "diagnostic_search_limited_cases": limited,
            "systematic_path_mismatch_suspected": expected > 0 and resolved == 0,
            "path_contract_verified": unresolved == 0 and invalid_training == 0, "maximum_shape_zyx": maximum,
            "training_image_header_cases": training_checked, "invalid_training_image_headers": invalid_training,
            "voxel_values_read": False, "outer_test_accessed": False, "alternatives_substituted": False}


def real_mask_preflight(contract, plan):
    from . import fusion_runtime as original
    from .ncct_paths import NCCTResolver
    from .ctp_paths import CTPResolver
    resolvers = {"ncct": NCCTResolver(contract, original.DATASET), "ctp": CTPResolver(contract, original.DATASET)}
    return mask_header_preflight(plan, original.DATASET, lambda p, m: resolvers[m].resolve(p), include_training_geometry=True)
