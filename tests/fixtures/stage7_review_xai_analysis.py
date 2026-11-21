"""One development case, final fused score, all quantitative controls.

Private-memory interface: no patient paths, identities or files are accepted.
Maps remain in memory and are released after each case; records contain only
private quantitative values. Public summaries require xai_schema aggregation.
"""
from __future__ import annotations

import numpy as np
import torch

from .xai_fused_score import fused_logit_torch, logit_components
from .xai_geometry import feature_labels, project_native, validate_labels
from .xai_metrics import roi_metrics, diagnostic, compare_maps, boundary, REGIONS
from .xai_perturbation import replace_blur, fidelity
from .xai_randomization import randomized_copy
from .xai_schema import METRIC_NAMES, METHODS, COMPARABLE, validate_values
from .xai_spatial import ncct_cam, ctp_cam, require_frozen
from .xai_temporal import temporal_occlusion


def analyse_case(model, value, clinical, state, modality, affine, *, labels_xyz=None,
                 mask_reason="missing_mask", expected_logit=None, clinical_logit=None):
    require_frozen(model)
    if modality not in ("ncct", "ctp"):
        raise ValueError("fixed imaging modality required")
    values = dict.fromkeys(METRIC_NAMES)
    reasons = dict.fromkeys(METRIC_NAMES, "not_applicable")
    def put(key, value, reason=None):
        values[key] = None if value is None else float(value)
        reasons[key] = reason if value is None else None
    cam = ncct_cam if modality == "ncct" else ctp_cam
    base = cam(model, value, clinical, state)
    z, sign = base["logit"], base["sign"]
    if expected_logit is not None and abs(z-expected_logit) > 1e-6:
        raise ValueError("case reconstructed score changed before analysis")
    native_shape = tuple(value.shape[2:]) if modality == "ncct" else tuple(reversed(value.metadata.spatial_shape))
    feature_shape = tuple(base["maps"]["gradcam"]["positive"].shape)
    if labels_xyz is not None:
        labels_xyz = validate_labels(labels_xyz, affine, tuple(reversed(native_shape)), affine)
    coarse_labels = None if labels_xyz is None else feature_labels(labels_xyz, affine, feature_shape)
    labels = None if labels_xyz is None else labels_xyz.transpose(2, 1, 0)
    brain = None if labels is None else labels > 0
    with torch.no_grad():
        embedding = model.forward_features(value) if modality == "ncct" else model._mean_embedding(value)
        parts = logit_components(embedding, clinical, state)
        for name, term in parts.items():
            put(f"decomposition.{name}", term.item())
        for name in ("image", "clinical"):
            put(f"decomposition.absolute_{name}", abs(parts[name].item()))
        image_logit = (model.classifier(embedding) if modality == "ncct" else model.head(embedding)).item()
        put("reference.imaging_only_logit", image_logit)
        put("reference.clinical_only_logit", clinical_logit, "not_applicable")
        put("reference.fused_minus_imaging", z-image_logit)
        put("reference.fused_minus_clinical", None if clinical_logit is None else z-clinical_logit, "not_applicable")
    for axis, count in zip(("depth", "height", "width"), feature_shape):
        put(f"resolution.{axis}_cells", count)

    def score_mask(mask):
        with torch.no_grad():
            changed = 0
            if modality == "ncct":
                perturbed, changed = replace_blur(value, mask)
                e = model.forward_features(perturbed)
            else:
                e = torch.zeros_like(embedding)
                for i in range(40):
                    frame, n = replace_blur(model._frame_tensor(value, i), mask)
                    e.add_(model.encoder.forward_features(frame))
                    changed += n
                e = e/40
            return float((sign*fused_logit_torch(e, clinical, state)).item()), changed

    audit = {"feature_shape": list(feature_shape), "native_shape": list(native_shape),
             "spacing_mm_zyx": np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)[::-1].tolist(),
             "mask_reason": mask_reason if labels is None else None, "interventions": {},
             "frame_count": 1 if modality == "ncct" else 40,
             "blur_sigma_voxels_zyx": [2., 2., 2.], "blur_radius_voxels": 6,
             "blur_sigma_mm_zyx": (2*np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)[::-1]).tolist(),
             "patch_shape_zyx": [4, 16, 16]}
    for method in METHODS:
        h = base["maps"][method]["positive"]
        reason = diagnostic(h)
        put(f"{method}.degenerate", float(reason is not None))
        for key, failure in (("map_nonfinite", "nonfinite_map"), ("map_constant", "constant_map"), ("map_near_zero", "near_zero_map")):
            put(f"{method}.{key}", float(reason == failure))
        anatomical = roi_metrics(h, coarse_labels, mask_reason=mask_reason)
        for region in REGIONS:
            for metric in ("mass", "enrichment"):
                put(f"{method}.roi.{region}.{metric}", anatomical["regions"][region][metric], anatomical["regions"][region]["reason"])
        native = None if reason == "nonfinite_map" else project_native(h, native_shape)
        nuisance = ({"nuisance": dict.fromkeys(("nonbrain", "native_border", "brain_boundary")), "reason": reason}
                    if native is None else roi_metrics(native, labels, mask_reason=mask_reason))
        for key, metric in nuisance["nuisance"].items():
            put(f"{method}.nuisance.{key}", metric, nuisance["reason"])
        if native is None:
            for key in ("target_drop", "random_median", "excess_drop", "target_exceeds_random"):
                put(f"{method}.fidelity.{key}", None, reason)
        elif brain is not None and brain.any():
            measured = fidelity(native, brain, score_mask, sign*z)
            for key in ("target_drop", "random_median", "excess_drop", "target_exceeds_random"):
                put(f"{method}.fidelity.{key}", measured[key], measured["reason"])
            audit["interventions"][method] = {k: measured[k] for k in ("selected_voxels", "changed_voxels") if k in measured}
        else:
            for key in ("target_drop", "random_median", "excess_drop", "target_exceeds_random"):
                put(f"{method}.fidelity.{key}", None, mask_reason if brain is None else "empty_brain")
    if brain is not None and brain.any():
        put("boundary.support_drop", sign*z-score_mask(boundary(brain))[0])
    else:
        put("boundary.support_drop", None, mask_reason if brain is None else "empty_brain")
    support = None if coarse_labels is None else coarse_labels > 0
    compared = compare_maps(*(base["maps"][m]["positive"] for m in METHODS), support=support)
    for key in ("rank_correlation", "top_dice"):
        put(f"method_comparison.{key}", compared[key], compared["reason"])
    for control, upper in (("head", False), ("upper", True)):
        if state["coef"][:8].square().mean().sqrt().item() <= 1e-12:
            for method in METHODS:
                for key in ("rank_correlation", "top_dice"):
                    put(f"{method}.randomization.{control}.{key}", None, "zero_head")
            continue
        randomized, randomized_state = randomized_copy(model, state, upper_block=upper)
        randomized_maps = cam(randomized, value, clinical, randomized_state, sign=sign)
        for method in METHODS:
            compared = compare_maps(base["maps"][method]["positive"], randomized_maps["maps"][method]["positive"], support)
            for key in ("rank_correlation", "top_dice"):
                put(f"{method}.randomization.{control}.{key}", compared[key], compared["reason"])
        del randomized, randomized_state, randomized_maps
    for key in COMPARABLE:
        a, b = (values[f"{m}.{key}"] for m in METHODS)
        put(f"paired.hirescam_minus_gradcam.{key}", None if a is None or b is None else b-a,
            reasons[f"gradcam.{key}"] if a is None else reasons[f"hirescam.{key}"])
    if modality == "ctp":
        score = lambda e: fused_logit_torch(e, clinical, state)
        temporal = temporal_occlusion(model, value, score, sign*z, sign=sign)
        for i, metric in enumerate(temporal["profile"]):
            put(f"temporal.frame_{i:02d}", metric)
        for start, metric in zip(temporal["window_starts"], temporal["window_drops"]):
            put(f"temporal.window_{start:02d}", metric)
    validate_values(values, reasons)
    return {"values": values, "reasons": reasons, "audit": audit}
