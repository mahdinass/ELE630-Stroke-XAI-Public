"""Strict numeric schemas shared by private analysis and aggregate publication."""
from __future__ import annotations

import math
import numpy as np

from .xai_metrics import REASONS, REGIONS

METHODS = ("gradcam", "hirescam")
COMPARABLE = tuple(f"roi.{r}.{v}" for r in REGIONS for v in ("mass", "enrichment")) + (
    "nuisance.nonbrain", "nuisance.native_border", "nuisance.brain_boundary",
    "fidelity.target_drop", "fidelity.random_median", "fidelity.excess_drop", "fidelity.target_exceeds_random")
SANITY = tuple(f"randomization.{control}.{metric}" for control in ("head", "upper")
               for metric in ("rank_correlation", "top_dice"))
MAP_DIAGNOSTICS = ("degenerate", "map_nonfinite", "map_constant", "map_near_zero")
METRIC_NAMES = tuple(f"{method}.{key}" for method in METHODS for key in COMPARABLE+SANITY+MAP_DIAGNOSTICS) + (
    "method_comparison.rank_correlation", "method_comparison.top_dice", "boundary.support_drop",
    "decomposition.intercept", "decomposition.image", "decomposition.clinical",
    "decomposition.absolute_image", "decomposition.absolute_clinical", "reference.imaging_only_logit",
    "reference.clinical_only_logit", "reference.fused_minus_imaging", "reference.fused_minus_clinical",
    "resolution.depth_cells", "resolution.height_cells", "resolution.width_cells") + tuple(
        f"paired.hirescam_minus_gradcam.{key}" for key in COMPARABLE) + tuple(
        f"temporal.frame_{i:02d}" for i in range(40)) + tuple(
        f"temporal.window_{i:02d}" for i in tuple(range(0, 36, 2))+(35,))
SUMMARY_KEYS = {"n_total", "n_valid", "mean", "sd", "median", "ci_low", "ci_high", "missing"}


def validate_values(values, reasons):
    if type(values) is not dict or set(values) != set(METRIC_NAMES) or set(reasons) != set(METRIC_NAMES):
        raise ValueError("quantitative metric schema rejected")
    for key, value in values.items():
        if value is None:
            if reasons[key] not in REASONS:
                raise ValueError("missing metric requires a fixed reason")
        elif type(value) not in (int, float) or not math.isfinite(value) or reasons[key] is not None:
            raise ValueError("quantitative metric value rejected")


def aggregate_cases(cases, *, minimum_valid=5):
    """Descriptive patient bootstrap, never voxel/frame-as-independent sampling."""
    if type(minimum_valid) is not int or minimum_valid < 2 or not cases:
        raise ValueError("positive cohort and aggregate suppression threshold required")
    for record in cases:
        validate_values(record["values"], record["reasons"])
    result = {}
    for key in METRIC_NAMES:
        if key == "decomposition.intercept":
            # This patient-constant contribution is itself a fitted coefficient.
            # Keep exact per-case decomposition private, even at cohort n23.
            result[key] = {"n_total": len(cases), "n_valid": 0, "mean": None, "sd": None,
                           "median": None, "ci_low": None, "ci_high": None,
                           "missing": {reason: len(cases) if reason == "private_parameter" else 0 for reason in REASONS}}
            continue
        values = np.array([r["values"][key] for r in cases if r["values"][key] is not None], dtype=float)
        missing = {reason: sum(r["values"][key] is None and r["reasons"][key] == reason for r in cases)
                   for reason in REASONS}
        stats = {"n_total": len(cases), "n_valid": len(values), "missing": missing,
                 "mean": None, "sd": None, "median": None, "ci_low": None, "ci_high": None}
        if len(values) >= minimum_valid:
            rng = np.random.default_rng(630)
            samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
            lo, hi = np.quantile(samples, [.025, .975])
            stats.update(mean=float(values.mean()), sd=float(values.std()), median=float(np.median(values)),
                         ci_low=float(lo), ci_high=float(hi))
        result[key] = stats
    validate_aggregate(result, len(cases), minimum_valid=minimum_valid)
    return result


def validate_aggregate(result, total, *, minimum_valid=5):
    if type(result) is not dict or set(result) != set(METRIC_NAMES) or type(total) is not int or total < 1:
        raise ValueError("aggregate schema rejected")
    for key, r in result.items():
        if key == "decomposition.intercept" and (r.get("n_valid") != 0 or r.get("missing", {}).get("private_parameter") != total):
            raise ValueError("fitted intercept must remain private")
        if (type(r) is not dict or set(r) != SUMMARY_KEYS or r["n_total"] != total
                or type(r["n_valid"]) is not int or not 0 <= r["n_valid"] <= total
                or type(r["missing"]) is not dict or set(r["missing"]) != set(REASONS)
                or any(type(v) is not int or v < 0 for v in r["missing"].values())
                or sum(r["missing"].values())+r["n_valid"] != total):
            raise ValueError("aggregate counts rejected")
        stats = [r[k] for k in ("mean", "sd", "median", "ci_low", "ci_high")]
        if r["n_valid"] < minimum_valid:
            if any(v is not None for v in stats):
                raise ValueError("small-denominator aggregate values forbidden")
        elif any(type(v) not in (int, float) or not math.isfinite(v) for v in stats):
            raise ValueError("invalid aggregate statistics")
        elif r["sd"] < 0 or r["ci_low"] > r["ci_high"]:
            raise ValueError("invalid aggregate uncertainty")
    return result
