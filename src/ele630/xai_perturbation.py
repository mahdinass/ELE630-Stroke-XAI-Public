"""Fixed 10%-brain patch interventions and ten matched-volume controls."""
from __future__ import annotations

import numpy as np
import hashlib

PATCH_ZYX = (4, 16, 16)
RANDOM_CONTROLS = 10
BLUR_SIGMA = 2.0
BLUR_RADIUS = 6


def patches(brain):
    if brain.ndim != 3 or brain.dtype != bool or not brain.any():
        raise ValueError("nonempty boolean brain support required")
    regions = []
    for z in range(0, brain.shape[0], PATCH_ZYX[0]):
        for y in range(0, brain.shape[1], PATCH_ZYX[1]):
            for x in range(0, brain.shape[2], PATCH_ZYX[2]):
                # nonzero traverses this <=1024-voxel slice in C order, exactly
                # matching the old full-volume flatnonzero for this patch.
                dz, dy, dx = np.nonzero(brain[z:z+4, y:y+16, x:x+16])
                indices = ((dz+z)*brain.shape[1]+dy+y)*brain.shape[2]+dx+x
                if len(indices):
                    regions.append(indices)
    return regions


class MemoizedMaskScore:
    """One unchanged case/target only; retain at most13 scalar results, no masks.

    Both methods still evaluate all ten prescribed controls. Identical masks
    reuse the identical deterministic counterfactual score without CNN reruns.
    """
    def __init__(self, score):
        self.score, self.cache = score, {}
        self.hits = 0

    def __call__(self, mask):
        if mask.ndim != 3 or mask.dtype != bool:
            raise ValueError("boolean spatial intervention mask required")
        key = (mask.shape, hashlib.sha256(np.packbits(mask, bitorder="little")).digest())
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if len(self.cache) >= 13:
            raise ValueError("fixed intervention cache bound exceeded")
        result = self.score(mask)
        self.cache[key] = result
        return result


def intervention_masks(attribution, brain, *, seed=630):
    h = np.asarray(attribution)
    if h.shape != brain.shape or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("finite nonnegative native map and matched brain required")
    regions = patches(brain)
    budget = max(1, int(np.ceil(.1*brain.sum())))
    flat_h = h.ravel()  # At most one C-order copy for noncontiguous input.
    scores = np.array([flat_h[idx].mean() for idx in regions])
    def choose(order, rng=None):
        selected, remaining = np.zeros(brain.size, dtype=bool), budget
        for i in order:
            idx = regions[i]
            if len(idx) > remaining and rng is not None:
                idx = rng.permutation(idx)
            chosen = idx[:remaining]
            selected[chosen] = True
            remaining -= len(chosen)
            if remaining == 0:
                break
        return selected.reshape(brain.shape)
    salient = choose(np.argsort(-scores, kind="stable"))
    rng = np.random.default_rng(seed)
    controls = [choose(rng.permutation(len(regions)), rng) for _ in range(RANDOM_CONTROLS)]
    if any(mask.sum() != budget or np.any(mask & ~brain) for mask in [salient]+controls):
        raise ValueError("intervention volume invariant failed")
    return salient, controls


def gaussian_kernel():
    indices = np.arange(-BLUR_RADIUS, BLUR_RADIUS+1, dtype=np.float64)
    kernel = np.exp(-.5*(indices/BLUR_SIGMA)**2)
    return kernel/kernel.sum()


def blur_numpy(volume):
    """Independent CPU oracle, separable normalized-input sigma2/radius6 blur."""
    result = np.asarray(volume, dtype=np.float64).copy()
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError("finite 3D input required")
    for axis in range(3):
        pad = [(0, 0)]*3
        pad[axis] = (BLUR_RADIUS, BLUR_RADIUS)
        padded = np.pad(result, pad, mode="edge")
        blurred = np.zeros_like(result)
        for offset, weight in enumerate(gaussian_kernel()):
            sl = [slice(None)]*3
            sl[axis] = slice(offset, offset+result.shape[axis])
            blurred += weight*padded[tuple(sl)]
        result = blurred
    return result


def replace_blur(frame, mask):
    """One model tensor; no graph, no full CTP stack or change outside mask."""
    import torch
    from torch.nn import functional as F
    if (frame.ndim != 5 or frame.shape[:2] != (1, 1) or tuple(mask.shape) != tuple(frame.shape[2:])
            or mask.dtype != bool or not torch.isfinite(frame).all()):
        raise ValueError("matched finite model tensor and boolean 3D mask required")
    with torch.no_grad():
        blurred = frame.detach()
        kernel = torch.tensor(gaussian_kernel(), device=frame.device, dtype=frame.dtype)
        for axis in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[axis+2] = len(kernel)
            pad = [0]*6
            pad[2*(2-axis):2*(2-axis)+2] = [BLUR_RADIUS]*2
            blurred = F.conv3d(F.pad(blurred, pad, mode="replicate"), kernel.reshape(shape))
        support = torch.as_tensor(mask, device=frame.device).reshape(1, 1, *mask.shape)
        replaced = torch.where(support, blurred, frame)
        changed = int(((replaced != frame) & support).sum().item())
        return replaced, changed


def fidelity(attribution, brain, score_mask, baseline_support, *, seed=630):
    """score_mask returns (same fixed-class support, actually changed voxels)."""
    from .xai_metrics import diagnostic
    reason = diagnostic(attribution)
    if reason:
        return {"target_drop": None, "random_median": None, "excess_drop": None,
                "target_exceeds_random": None, "reason": reason}
    salient, controls = intervention_masks(attribution, brain, seed=seed)
    score, changed = score_mask(salient)
    drops = np.array([baseline_support-score_mask(mask)[0] for mask in controls], dtype=np.float64)
    target = float(baseline_support-score)
    if not np.isfinite(drops).all() or not np.isfinite(target):
        raise ValueError("nonfinite intervention score")
    median = float(np.median(drops))
    return {"target_drop": target, "random_median": median, "excess_drop": target-median,
            "target_exceeds_random": float(target > median), "reason": None,
            "selected_voxels": int(salient.sum()), "changed_voxels": changed,
            "random_drops": drops.tolist()}
