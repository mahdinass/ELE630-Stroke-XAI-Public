#!/usr/bin/env python3
"""UiS-only, explicitly authorized Stage 7 development CAM illustration export.

Does not retrain, touch outer-test patients, or alter the completed Stage 7
aggregate studies. Patient images and saliency maps remain in the private UiS
runtime. Export to any external service is a separate, human-reviewed action.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_SAFE_ID = re.compile(r"stage7_overlay_[A-Za-z0-9_-]{1,42}\Z")
_FRAME_INDEX = 20  # Fixed midpoint frame, not chosen after inspecting maps.


def select_development_ordinal(logits):
    """Middle-ranked absolute logit, with original fold order as tie-breaker."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (23,) or not np.isfinite(values).all():
        raise ValueError("exact original development cohort required")
    return sorted(range(23), key=lambda i: (abs(float(values[i])), i))[11]


def _native_maps(feature_maps, shape_zyx):
    from ele630.xai_geometry import project_native
    mapped = {}
    for method in ("gradcam", "hirescam"):
        if method not in feature_maps or "positive" not in feature_maps[method]:
            raise ValueError("missing verified CAM")
        data = project_native(feature_maps[method]["positive"], shape_zyx)
        if data.shape != tuple(shape_zyx) or not np.isfinite(data).all():
            raise ValueError("invalid projected CAM")
        mapped[method] = np.maximum(data, 0)
    return mapped


def render_triptych(image_zyx, maps, affine, *, modality):
    """Return a privacy-minimal PNG, no patient ID, raw file paths or metadata.

    Display a fixed middle native Z plane. The two maps use one shared absolute
    color scale; no claim about clinical laterality or lesion correctness.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    image = np.asarray(image_zyx, dtype=np.float64)
    matrix = np.asarray(affine, dtype=np.float64)
    if (image.ndim != 3 or min(image.shape) < 2
            or not np.isfinite(image).all() or matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12):
        raise ValueError("invalid native image or orientation matrix")
    if modality not in ("ncct", "ctp") or set(maps) != {"gradcam", "hirescam"}:
        raise ValueError("unexpected XAI panel input")
    for m in maps.values():
        if np.asarray(m).shape != image.shape or not np.isfinite(m).all() or np.min(m) < 0:
            raise ValueError("CAM and CT geometry mismatch")
    middle = image.shape[0] // 2
    cut = image[middle]
    heatmaps = [np.asarray(maps[name])[middle] for name in ("gradcam", "hirescam")]
    positive = np.concatenate([arr.ravel() for arr in heatmaps])
    vmax = float(np.percentile(positive, 99.5))
    if not np.isfinite(vmax) or vmax <= 1e-12:
        raise ValueError("noninformative or degenerate CAM slice")
    lo, hi = np.percentile(cut, [1, 99])
    if not lo < hi:
        raise ValueError("degenerate CT display slice")
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.2), constrained_layout=True)
    for ax in axs:
        ax.imshow(cut, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
        ax.axis("off")
    axs[0].set_title("CT input (model scale)")
    overlay = None
    for ax, data, label in zip(axs[1:], heatmaps, ("Grad-CAM", "HiResCAM")):
        overlay = ax.imshow(data, cmap="inferno", vmin=0, vmax=vmax,
                            alpha=np.clip(data / vmax, 0, 1) * 0.65,
                            origin="lower", interpolation="nearest")
        ax.set_title(label)
    fig.colorbar(overlay, ax=axs[1:], fraction=0.027, pad=0.02, label="Positive attribution (shared scale)")
    fig.suptitle(
        ("NCCT" if modality == "ncct" else f"CTP original frame {_FRAME_INDEX}")
        + " | fixed native-array Z midpoint | illustrative development case",
        fontsize=10,
    )
    data = io.BytesIO()
    try:
        fig.savefig(data, format="png", dpi=180, metadata={})
        return data.getvalue()
    finally:
        plt.close(fig)


def private_output_dir(runtime_root, output_id):
    """Create an exclusive 0700 directory beneath the configured private runtime."""
    if not _SAFE_ID.fullmatch(output_id):
        raise ValueError("private output ID must match fixed slug")
    root = Path(runtime_root).resolve(strict=True)
    base = root / "stage7_private_overlay_examples"
    if base.is_symlink():
        raise ValueError("symlink private output root rejected")
    base.mkdir(mode=0o700, exist_ok=True)
    if base.stat().st_mode & 0o077:
        raise ValueError("private output root must be owner-only")
    destination = base / output_id
    destination.mkdir(mode=0o700, exist_ok=False)
    return destination


def exclusive_write(directory, filename, data):
    if not re.fullmatch(r"[a-z_]+\.(png|json)", filename):
        raise ValueError("unsafe private filename")
    target = Path(directory) / filename
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


def export_one(modality, revision, protocol_hash, smoke_run, resource_run, output_id,
               *, execute_real=False, qa_original=None):
    """Produce exactly one selected development image, never export it externally."""
    if modality not in ("ncct", "ctp") or execute_real is not True:
        raise ValueError("explicit authorized real development-only export required")
    os.umask(0o077)
    from ele630 import fusion_runtime as original, xai_runtime as runtime
    from ele630.xai_runner import verify_smoke, _check_torch_equivalence
    from ele630.fusion_runner import load_plan, check_clinical_path, real_features
    from ele630.fusion_checkpoints import load_ncct, load_ctp
    from ele630.ncct_training import _seed_everything, _parameter_digest, _membership_record
    from ele630.xai_masks import real_mask_preflight
    from ele630.xai_data import real_ctp_features, real_validation_access
    from ele630.xai_ctp_cache import cached_series
    from ele630.xai_resources import ResourceProfile
    from ele630.xai_spatial import ncct_cam, ctp_cam
    import torch

    prior = None
    identity = runtime.source_identity(revision)
    runtime.approved_protocol(identity, protocol_hash, execute_real)
    dependencies = original.environment()
    dependencies["clinical"] = runtime.bridge({"probe": True})["dependencies"]
    runtime.validate_dependencies(dependencies)
    verify_smoke(smoke_run, revision, dependencies=dependencies)
    contract, plan = load_plan()
    check_clinical_path()
    mask_headers = real_mask_preflight(contract, plan)
    if not mask_headers["path_contract_verified"]:
        raise ValueError("unverified dataset mask paths")
    if modality == "ctp":
        runtime.verify_resource(resource_run, revision, dependencies, mask_headers["maximum_shape_zyx"]["ctp"])

    if qa_original is not None:
        expected_id = f"stage7_overlay_{modality}_review01"
        if qa_original != expected_id:
            raise ValueError("only original reviewed selection permitted")
        from ele630.ncct_private_output import _validate_base
        old = original.RUNTIME / "stage7_private_overlay_examples" / expected_id
        _validate_base(old)
        prior = original.read_private_json(old / "manifest.json")
        original_png = original.private_file(old / f"{modality}_development_example.png")
        from ele630.xai_visual_qa import verify_original_manifest
        verify_original_manifest(prior, original_png.read_bytes(), modality=modality, protocol=protocol_hash)
        if (not _SAFE_ID.fullmatch(output_id) or output_id == expected_id
                or (old.parent/output_id).exists() or (old.parent/output_id).is_symlink()):
            raise ValueError("fresh private QA output identity required")

    _seed_everything(630)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, provenance = load_ncct(plan) if modality == "ncct" else load_ctp(contract)
    checkpoint_hash = _parameter_digest(model)
    resources = ResourceProfile("cuda")
    with resources.attach(model), resources.phase("feature_extraction"):
        if modality == "ctp":
            train, validation, probabilities = real_ctp_features(contract, plan, model, resources=resources)
        else:
            train, validation, probabilities = real_features(contract, plan, modality, model)
    result = runtime.bridge({
        "mode": "real", "plan": asdict(plan), "train": train.tolist(),
        "validation": validation.tolist(), "imaging_probabilities": probabilities.tolist(),
        "source_revision": revision,
    })
    historical = original.read_private_json(
        original.RUNTIME / "fusion_runs" / original.RUNS[modality]
        / "outer_fold_0" / "aggregate_training.json"
    )
    runtime.historical_gate(result, modality, historical, provenance=provenance,
                            membership=_membership_record(plan))
    state = _check_torch_equivalence(result, validation, "cuda")
    del train, validation, probabilities, historical
    ordinal = select_development_ordinal(result["logits"])
    if prior is not None and ordinal != prior["selected_development_ordinal"]:
        raise ValueError("deterministic selection changed; no replacement case allowed")
    patient = plan.inner_validation[ordinal]  # No test access, no appearance-based choice.
    access = real_validation_access(contract, plan, modality, "cuda")
    raw, affine, shape_xyz = access.load(patient)
    cache = cached_series(raw, resources=resources) if modality == "ctp" else nullcontext(raw)
    with cache as value, resources.attach(model), resources.phase("baseline_cam"):
        del raw
        clinical = torch.as_tensor(result["clinical_validation"][ordinal:ordinal + 1], device="cuda")
        base = (ctp_cam if modality == "ctp" else ncct_cam)(model, value, clinical, state)
        if abs(base["logit"] - float(result["logits"][ordinal])) > 1e-6:
            raise ValueError("selected case score failed the frozen reconstruction gate")
        if modality == "ctp":
            frame = model._frame_tensor(value, _FRAME_INDEX)
            source_maps = base["frames"][_FRAME_INDEX]
        else:
            frame = value
            source_maps = base["maps"]
        image = frame[0, 0].detach().cpu().numpy()
        expected = tuple(image.shape)
        if expected != tuple(reversed(shape_xyz)):
            raise ValueError("image and native NIfTI spatial shape mismatch")
        if prior is None:
            native_maps = _native_maps(source_maps, expected)
            png = render_triptych(image, native_maps, affine, modality=modality)
        else:
            from ele630.xai_visual_qa import audit_maps, render_preview
            private_qa, public_qa, cuts = audit_maps(image, source_maps, affine, modality=modality)
            private_qa["ctp_frame_index"] = 20 if modality == "ctp" else None
            private_qa["prior_png_sha256"] = prior["png_sha256"]
            private_qa["selection_matches_original"] = True
            private_qa["target_matches_historical_gate"] = True
            private_qa["intensity_units"] = "clipped source scale; not certified HU" if modality == "ncct" else "registered processed values /255; not certified HU or perfusion units"
            if modality == "ctp":
                aggregate, _, _ = audit_maps(image, base["maps"], affine, modality=modality, aggregate=True)
                private_qa["aggregate_40_frame_map_stats"] = aggregate["methods"]
            png = render_preview(image, cuts, affine, modality=modality)
            separate_png = render_preview(image, cuts, affine, modality=modality, separate=True)
    if _parameter_digest(model) != checkpoint_hash:
        raise ValueError("frozen encoder parameters changed")
    if runtime.source_identity(revision) != identity:
        raise ValueError("source changed during private export")

    directory = private_output_dir(original.RUNTIME, output_id)
    filename = f"{modality}_development_example.png" if prior is None else f"{modality}_shared_absolute.png"
    png_sha256 = exclusive_write(directory, filename, png)
    manifest = {
        "schema": 1, "status": "PRIVATE_REVIEW_ONLY",
        "modality": modality, "source_revision": revision,
        "protocol_sha256": identity["protocol_sha256"],
        "validation_count": 23, "selected_development_ordinal": ordinal,
        "selection": "median absolute frozen fused logit; original ordinal tie-break",
        "plane": "native Z midpoint (unoriented array view)",
        "ctp_original_frame_index": _FRAME_INDEX if modality == "ctp" else None,
        "cam_target": "original predicted-class fused support logit",
        "cam_layer": "stage4.conv2", "exported_to_external_service": False,
        "filename": filename, "png_sha256": png_sha256,
    }
    if prior is not None:
        manifest["qa_mode"] = "private visualization audit only; original midpoint and frame preserved"
        manifest["separate_png_sha256"] = exclusive_write(directory, f"{modality}_separate_display.png", separate_png)
        exclusive_write(directory, "private_qa.json", json.dumps(private_qa, indent=2, allow_nan=False).encode())
        from ele630.xai_visual_qa import validate_public
        exclusive_write(directory, "shareable_summary.json", json.dumps(validate_public(public_qa), indent=2).encode())
    exclusive_write(directory, "manifest.json", json.dumps(manifest, indent=2).encode())
    if prior is not None:
        return validate_public(public_qa)
    return {"status": "PRIVATE_REVIEW_READY", "modality": modality,
            "output_id": output_id, "filename": filename,
            "validation_count": 23, "outer_test_evaluated": False,
            "external_export_completed": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("modality", choices=("ncct", "ctp"))
    p.add_argument("--source-revision", required=True)
    p.add_argument("--protocol-reviewed", required=True)
    p.add_argument("--smoke-run", required=True)
    p.add_argument("--resource-run", help="same-source synthetic resource gate, CTP only")
    p.add_argument("--output-id", required=True)
    p.add_argument("--execute-real", action="store_true")
    p.add_argument("--qa-original", help="fixed original private export ID; no new selection")
    args = p.parse_args(argv)
    try:
        data = export_one(args.modality, args.source_revision, args.protocol_reviewed,
                          args.smoke_run, args.resource_run, args.output_id,
                          execute_real=args.execute_real, qa_original=args.qa_original)
        print(json.dumps(data, indent=2))
        return 0
    except Exception:
        print("PRIVATE_OVERLAY_FAIL: inspect only in the approved UiS private environment", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
