"""Stage 5C2D: training-only native NCCT QC, never anatomical certification.

No model, labels, GT, resampling or repair is used in measurement/selection.
NIfTI loading is delegated to NCCTRepository by the execution CLI.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import warnings
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr

import numpy as np

from .data_contract import ACTIVE_PROFILE, DatasetContract, OutcomePatient
from .native_ncct_input import prepare_native_ncct_input
from .ncct_intensity import clip_scale_ncct
from .ncct_splits import plan_outer_fold


class AnatomyQCError(ValueError):
    """Fixed, non-identifying QC failure category."""

    def __init__(self, kind: str):
        super().__init__("NCCT training QC failed")
        self.kind = kind


SELECTORS = (
    ("minimum_native_depth", "native_depth", False),
    ("maximum_native_depth", "native_depth", True),
    ("minimum_mean_inplane_spacing", "mean_inplane_spacing_mm", False),
    ("maximum_mean_inplane_spacing", "mean_inplane_spacing_mm", True),
    ("maximum_inplane_voxels", "inplane_voxel_count", True),
    ("maximum_axis_obliquity", "maximum_axis_obliquity_degrees", True),
    ("minimum_transformed_zero_fraction", "transformed_zero_fraction", False),
    ("maximum_transformed_zero_fraction", "transformed_zero_fraction", True),
)


def validate_qc_plan(contract: DatasetContract, supplied_plan=None):
    """Reconstruct the exact registered split before any repository access."""
    if (type(contract) is not DatasetContract or len(contract.patients) != 143
            or (contract.good, contract.poor, contract.with_gt) != (98, 45, 142)
            or contract.fingerprint != ACTIVE_PROFILE.fingerprint or contract.seed != 630):
        raise AnatomyQCError("contract")
    try:
        canonical = plan_outer_fold(contract, 0, seed=630, validation_fraction=.20)
    except ValueError:
        raise AnatomyQCError("contract") from None
    if (tuple(map(len, (canonical.inner_train, canonical.inner_validation, canonical.outer_test)))
            != (92, 23, 28) or (supplied_plan is not None and supplied_plan != canonical)):
        raise AnatomyQCError("partition")
    return canonical


class TrainingOnlyNCCTQC:
    """Authorize original training objects BEFORE delegating path/image I/O."""

    def __init__(self, contract, repository, *, plan=None):
        self.plan = validate_qc_plan(contract, plan)
        self._allowed = {p.patient_id: p for p in self.plan.inner_train}
        if not callable(getattr(repository, "load", None)):
            raise AnatomyQCError("repository")
        self._repository = repository

    def load(self, patient):
        if (type(patient) is not OutcomePatient
                or self._allowed.get(patient.patient_id) is not patient):
            # Identity also rejects reconstructed/forged membership objects.
            raise AnatomyQCError("unauthorized")
        return self._repository.load(patient)


def measure_volume(volume) -> dict:
    """Measure one already NIfTI-scaled native array without changing it."""
    native = prepare_native_ncct_input(volume.data, volume.affine)
    if tuple(volume.shape) != native.native_shape:
        raise AnatomyQCError("dimension")
    shape = np.asarray(native.native_shape, dtype=np.int64)
    affine = np.asarray(native.native_affine)
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    header_spacing = np.asarray(volume.spacing_mm, dtype=np.float64)
    if (header_spacing.shape != (3,) or not np.isfinite(header_spacing).all()
            or np.any(header_spacing <= 0)):
        raise AnatomyQCError("header_spacing")
    units = volume.header.get_xyzt_units()[0]
    # Unit conversion applies only to QC measurements, never input voxels/affine.
    factor = {"mm": 1.0, "meter": 1000.0, "micron": .001}.get(units)
    if factor is None:
        raise AnatomyQCError("unknown_spatial_units")
    slope, intercept = float(volume.slope), float(volume.intercept)
    if not np.isfinite((slope, intercept)).all() or slope == 0:
        raise AnatomyQCError("scaling")
    directions = axes / spacing
    obliquity = np.degrees(np.arccos(np.clip(np.max(np.abs(directions), axis=0), 0, 1)))
    corners = np.asarray(list(itertools.product(*[(-.5, float(n) - .5) for n in shape])))
    world = (corners @ axes.T + affine[:3, 3]) * factor
    flags = []
    # Arithmetic/header comparison tolerances, not anatomical acceptance limits.
    if not np.allclose(header_spacing, spacing, rtol=1e-5, atol=1e-4 / factor):
        flags.append("header_affine_spacing_disagreement")
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        flags.append("nonorthogonal_affine_axes")
    if not volume.sform_code and not volume.qform_code:
        flags.append("fallback_affine")
    if volume.sform_code and volume.qform_code:
        if not np.allclose(volume.header.get_sform(), volume.header.get_qform(), rtol=1e-5, atol=1e-4 / factor):
            flags.append("sform_qform_disagreement")
    transformed = clip_scale_ncct(native.tensor, lower=-20, upper=120)
    raw = volume.data
    if np.min(raw) == np.max(raw):
        flags.append("constant_source_intensity")
    if np.min(transformed) == np.max(transformed):
        flags.append("constant_transformed_intensity")
    quantiles = np.percentile(raw, (0, 1, 25, 50, 75, 99, 100))
    record = {
        "native_shape_xyz": shape.tolist(), "model_shape_czyx": list(native.tensor.shape),
        "native_depth": int(shape[2]), "inplane_voxel_count": int(shape[0] * shape[1]),
        "affine_ras_native_units": affine.tolist(), "spatial_units": units,
        "orientation": volume.orientation, "affine_determinant_native_units": float(np.linalg.det(axes)),
        "affine_spacing_mm": (spacing * factor).tolist(),
        "header_spacing_mm": (header_spacing * factor).tolist(),
        "mean_inplane_spacing_mm": float(np.mean(spacing[:2]) * factor),
        "axis_obliquity_degrees": obliquity.tolist(), "maximum_axis_obliquity_degrees": float(max(obliquity)),
        "axis_voxel_cell_extent_mm": (shape * spacing * factor).tolist(),
        "axis_center_span_mm": ((shape - 1) * spacing * factor).tolist(),
        "world_voxel_cell_bounds_mm": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
        "world_voxel_cell_extent_mm": np.ptp(world, axis=0).tolist(),
        "sform_code": int(volume.sform_code), "qform_code": int(volume.qform_code),
        "stored_dtype": volume.stored_dtype, "decoded_dtype": str(raw.dtype),
        "proxy_slope": slope, "proxy_intercept": intercept,
        "scaled_intensity_percentiles_0_1_25_50_75_99_100": quantiles.tolist(),
        "scaled_intensity_mean": float(np.mean(raw, dtype=np.float64)),
        "scaled_intensity_sd": float(np.std(raw, dtype=np.float64)),
        "source_zero_fraction": float(np.mean(raw == 0)),
        "below_window_fraction_float32": float(np.mean(native.tensor < -20)),
        "above_window_fraction_float32": float(np.mean(native.tensor > 120)),
        "transformed_zero_fraction": float(np.mean(transformed == 0)),
        "transformed_one_fraction": float(np.mean(transformed == 1)),
        "transformed_mean": float(np.mean(transformed, dtype=np.float64)),
        "flags": flags,
    }
    # Do not emit nonfinite JSON even when derived quantities overflow.
    try:
        json.dumps(record, allow_nan=False)
    except ValueError:
        raise AnatomyQCError("nonfinite_measurement") from None
    return record


def select_visual_cases(cases: list[dict]) -> list[dict]:
    """Fixed eight selectors; canonical-order ties; no replacement for overlap."""
    ordered = sorted(cases, key=lambda case: case["canonical_index"])
    if not ordered or any(case["status"] != "ok" for case in ordered):
        return []  # Incomplete integrity QC pauses the whole recipe, not exclusion.
    selected = []
    by_index = {}
    for reason, key, maximum in SELECTORS:
        choice = (max if maximum else min)(ordered, key=lambda case: case["measurements"][key])
        index = choice["canonical_index"]
        if index not in by_index:
            by_index[index] = {"canonical_index": index, "patient_id": choice["patient_id"], "selectors": []}
            selected.append(by_index[index])
        by_index[index]["selectors"].append(reason)
    return selected


def scan_training_qc(guard: TrainingOnlyNCCTQC) -> dict:
    """One image at a time; output is PRIVATE, never suitable for terminal use."""
    cases = []
    safe_kinds = {"unreadable", "dimension", "metadata", "nonfinite", "missing", "ambiguous",
                  "ineligible", "header_spacing", "unknown_spatial_units", "scaling", "nonfinite_measurement"}
    for index, patient in enumerate(guard.plan.inner_train):
        case = {"canonical_index": index, "patient_id": patient.patient_id}
        try:
            # Library diagnostics may include filenames/header text. Keep them
            # off terminal output; retain only a fixed warning flag, never text.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                volume = guard.load(patient)
                try:
                    measurements = measure_volume(volume)
                finally:
                    del volume
            if caught:
                measurements["flags"].append("loader_or_measurement_warning")
            case.update(status="ok", measurements=measurements)
        except Exception as exc:
            kind = getattr(exc, "kind", "integrity")
            case["status"] = kind if kind in safe_kinds else "integrity"
        cases.append(case)
    counts = dict(sorted(Counter(c["status"] for c in cases).items()))
    passed = counts.get("ok", 0) == 92
    flags = Counter(flag for c in cases if c["status"] == "ok" for flag in c["measurements"]["flags"])
    summary = {"stage": "5C2D", "training_cases": 92, "status_counts": counts,
               "integrity_complete": passed, "flag_counts": dict(sorted(flags.items())),
               "human_anatomy_review_required": True, "anatomy_certified": False,
               "validation_images_accessed": False, "outer_images_accessed": False,
               "training_authorized": False}
    membership = [{"patient_id": p.patient_id, "cohort": p.cohort, "outcome": p.outcome,
                   "has_gt": p.has_gt, "fold": p.fold} for p in guard.plan.inner_train]
    # Labels bind split provenance ONLY; they do not enter measurements/selection.
    membership_digest = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "summary": summary, "training_membership_sha256": membership_digest,
            "cases": cases, "visual_selection": select_visual_cases(cases),
            "review_template": {"reviewer": None, "date": None, "decision": "pending",
                                "checks": ["complete intracranial coverage", "left-right and axes",
                                           "skull-stripping loss", "scaling consistency", "windowed brain contrast"],
                                "notes": None}}


def write_private_qc(run_paths, report: dict) -> None:
    """Exclusive 0600 output; requires paths from create_private_run()."""
    from .ncct_private_output import PrivateRunPaths, PrivateOutputError, open_private_run, open_private_binary
    if (type(run_paths) is not PrivateRunPaths
            or open_private_run(run_paths.run_dir.parent, run_paths.run_dir.name, 0) != run_paths):
        raise PrivateOutputError("QC destination must be a verified private fold-0 run")
    with open_private_binary(run_paths.fold_dir / "training_anatomy_qc.json") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8"))
