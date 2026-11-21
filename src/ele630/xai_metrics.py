"""Pure NumPy quantitative XAI with explicit undefined values and reasons."""
from __future__ import annotations

import numpy as np

REGIONS = ("core", "penumbra_only", "lesion_union", "outside_lesion_brain")
REASONS = ("missing_mask", "invalid_mask", "empty_brain", "empty_region", "nonfinite_map",
           "near_zero_map", "constant_map", "zero_brain_mass", "insufficient_patients",
           "constant_rank", "zero_head", "not_applicable", "private_parameter")


def diagnostic(value):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 3 or not x.size:
        raise ValueError("nonempty 3D attribution required")
    if not np.isfinite(x).all():
        return "nonfinite_map"
    if np.any(x < 0):
        raise ValueError("positive evidence map required")
    if x.max() <= 1e-12:
        return "near_zero_map"
    if np.ptp(x) <= 1e-12:
        return "constant_map"
    return None


def boundary(brain):
    """Brain-side six-connected one-voxel boundary, outside array is nonbrain."""
    b = np.asarray(brain)
    if b.ndim != 3 or b.dtype != bool:
        raise ValueError("boolean 3D brain support required")
    padded = np.pad(b, 1, constant_values=False)
    interior = b.copy()
    for axis in range(3):
        for shift in (-1, 1):
            slices = [slice(1, n+1) for n in b.shape]
            slices[axis] = slice(1+shift, b.shape[axis]+1+shift)
            interior &= padded[tuple(slices)]
    return b & ~interior


def roi_metrics(value, labels=None, *, mask_reason="missing_mask"):
    x = np.asarray(value, dtype=np.float64)
    reason = diagnostic(x)
    if labels is None:
        reason = mask_reason
    elif (labels.shape != x.shape or not np.isin(labels, (0, 1, 2, 3)).all()):
        reason = "invalid_mask"
    brain = np.zeros(x.shape, dtype=bool) if labels is None or reason == "invalid_mask" else labels > 0
    if reason is None and not brain.any():
        reason = "empty_brain"
    if reason is None and x[brain].sum() <= 1e-12:
        reason = "zero_brain_mass"
    result = {name: {"mass": None, "enrichment": None, "reason": reason} for name in REGIONS}
    if reason is None:
        core, pen = labels == 3, labels == 2
        regions = (core, pen & ~core, core | pen, brain & ~(core | pen))
        for name, region in zip(REGIONS, regions):
            if not region.any():
                result[name]["reason"] = "empty_region"
            else:
                fraction = float(x[region].sum()/x[brain].sum())
                result[name] = {"mass": fraction,
                                "enrichment": fraction/float(region.sum()/brain.sum()), "reason": None}
    # Nuisance uses full-map mass, not within-brain normalization.
    nuisance = {name: None for name in ("nonbrain", "native_border", "brain_boundary")}
    if reason not in ("missing_mask", "invalid_mask", "empty_brain", "nonfinite_map", "near_zero_map", "constant_map"):
        border = np.zeros_like(brain)
        for axis in range(3):
            for index in (0, -1):
                slices = [slice(None)]*3
                slices[axis] = index
                border[tuple(slices)] = True
        for name, region in (("nonbrain", ~brain), ("native_border", border), ("brain_boundary", boundary(brain))):
            nuisance[name] = float(x[region].sum()/x.sum())
    return {"regions": result, "nuisance": nuisance, "reason": reason,
            "grid_voxels": int(x.size), "brain_voxels": int(brain.sum())}


def ranks(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(x).all():
        raise ValueError("finite rank input required")
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start+1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        result[order[start:end]] = .5*(start+end-1)
        start = end
    return result


def compare_maps(first, second, support=None):
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("matched 3D map shapes required")
    support = np.ones_like(a, dtype=bool) if support is None else np.asarray(support)
    if support.shape != a.shape or support.dtype != bool:
        raise ValueError("matched boolean comparison support required")
    reason = diagnostic(a) or diagnostic(b)
    if reason or not support.any():
        return {"rank_correlation": None, "top_dice": None, "reason": reason or "empty_brain"}
    ar, br = ranks(a[support]), ranks(b[support])
    if np.ptp(ar) == 0 or np.ptp(br) == 0:
        return {"rank_correlation": None, "top_dice": None, "reason": "constant_rank"}
    k = max(1, int(np.ceil(.1*support.sum())))
    ia, ib = np.argsort(-a[support], kind="stable")[:k], np.argsort(-b[support], kind="stable")[:k]
    return {"rank_correlation": float(np.corrcoef(ar, br)[0, 1]),
            "top_dice": float(len(np.intersect1d(ia, ib))/k), "reason": None}
