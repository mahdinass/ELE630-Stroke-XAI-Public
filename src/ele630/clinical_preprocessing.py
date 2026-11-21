"""Admission clinical inputs shared by the final NCCT and CTP branches.

No file I/O, labels, split creation or model integration. Fit a fresh instance
for each existing FoldPlan; only inner_train rows are accessed during fitting.
Source mappings are never modified, and extra columns are never inspected.

The 11 Oct 2026 wake-up interval masking policy is a technical proposal,
not a supervisor-confirmed medical interpretation. Recorded wake-up == 1
masks the interval before parsing/imputation; unknown status does not trigger
masking. Never infer durations from raw clocks or repair recorded intervals.
Nonmissing durations must be H:MM or H:MM:SS (hours:minutes[:seconds]); numeric
durations are rejected because their units are ambiguous. Hours may exceed 23;
there is no evidence-backed upper duration cutoff yet. Private server validity
checks remain required before real-data use. AF/PAF remains an approved input
with a qualified admission-timing confirmation.

Four numeric inputs use train median imputation, then train population mean/std
scaling. Categories use train mode (ties choose the smaller code); F=0, M=1,
other binary fields retain 0/1, smoking is 0 versus >0. These encodings are
fixed schema rules, not learned from validation. Entirely missing training
columns fail rather than invent a fallback. Output is finite float32, in
FEATURE_NAMES order; zero-variance numeric columns use scale 1.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np

from .ncct_splits import FoldPlan, validate_fold_plan


FEATURE_NAMES = (
    "age", "sex", "nihss_admission", "mrs_baseline", "onset_to_ct_time",
    "af_paf", "ht", "dm", "smoking", "anticoagulation", "wake_up_stroke",
)
_NUMERIC = (0, 2, 3, 4)
_MISSING = frozenset({"", "n/a", "na", "nan", "none", "null"})
_DURATION = re.compile(r"([0-9]+):([0-9]{2})(?::([0-9]{2}))?")


class ClinicalPreprocessingError(ValueError):
    """Schema or state error; messages contain no patient values or identifiers."""


def _missing(value: object) -> bool:
    return (value is None or
            isinstance(value, str) and value.strip().lower() in _MISSING or
            isinstance(value, (float, np.floating)) and math.isnan(value))


def parse_onset_to_ct_minutes(value: object) -> float:
    """Parse recorded durations only; genuine missing values remain NaN."""
    if _missing(value):
        return math.nan
    match = _DURATION.fullmatch(value.strip()) if isinstance(value, str) else None
    if match is None:
        raise ClinicalPreprocessingError("invalid onset_to_ct_time duration format")
    try:
        hours, minutes = int(match[1]), int(match[2])
        seconds = int(match[3] or 0)
        result = hours * 60.0 + minutes + seconds / 60.0
    except (ValueError, OverflowError):
        raise ClinicalPreprocessingError("invalid onset_to_ct_time duration") from None
    if minutes >= 60 or seconds >= 60 or not math.isfinite(result):
        raise ClinicalPreprocessingError("invalid onset_to_ct_time duration components")
    return result


def _number(value: object, name: str) -> float:
    if _missing(value):
        return math.nan
    if not isinstance(value, (str, int, float, np.integer, np.floating)):
        raise ClinicalPreprocessingError(f"invalid {name} numeric representation")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ClinicalPreprocessingError(f"invalid {name} numeric representation") from None
    if not math.isfinite(result) or result < 0:
        raise ClinicalPreprocessingError(f"invalid {name} numeric value")
    return result


def clinical_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Project only approved columns; parse/mask without fitting or mutation.

    Missing keys are schema errors; missing cells must be explicitly represented.
    NIHSS and baseline mRS require integer scale values in 0..42 and 0..6.
    Sex accepts the audited F/M codes only, after trimming whitespace.
    """
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or any(name not in row for name in FEATURE_NAMES):
            raise ClinicalPreprocessingError("clinical row requires all approved columns")
        wake = _number(row["wake_up_stroke"], "wake_up_stroke")
        if not math.isnan(wake) and wake not in (0, 1):
            raise ClinicalPreprocessingError("invalid wake_up_stroke binary code")
        for column, name in enumerate(FEATURE_NAMES):
            if name == "onset_to_ct_time":
                value = math.nan if wake == 1 else parse_onset_to_ct_minutes(row[name])
            elif name == "sex":
                raw = row[name]
                if _missing(raw):
                    value = math.nan
                elif isinstance(raw, str) and raw.strip() in {"F", "M"}:
                    value = float(raw.strip() == "M")
                else:
                    raise ClinicalPreprocessingError("invalid sex category")
            elif name == "wake_up_stroke":
                value = wake
            else:
                value = _number(row[name], name)
                if not math.isnan(value):
                    if name == "smoking":
                        value = float(value > 0)
                    elif name in {"nihss_admission", "mrs_baseline"}:
                        upper = 42 if name == "nihss_admission" else 6
                        if not value.is_integer() or value > upper:
                            raise ClinicalPreprocessingError(f"invalid {name} scale value")
                    elif column not in _NUMERIC and value not in (0, 1):
                        raise ClinicalPreprocessingError(f"invalid {name} binary code")
            matrix[index, column] = value
    return matrix


class ClinicalPreprocessor:
    """Train-only fitted transformer; create a new instance for every fold.

    rows_by_patient is a caller-supplied in-memory mapping keyed by contract ID.
    IDs are used only to select training rows, never stored or returned as inputs.
    transform accepts already-selected rows; it cannot access any repository or
    create splits. Callers must keep outer tests locked during development.
    """

    feature_names = FEATURE_NAMES

    def __init__(self) -> None:
        self._state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def fit_training(self, plan: FoldPlan,
                     rows_by_patient: Mapping[str, Mapping[str, object]]) -> ClinicalPreprocessor:
        if self._state is not None:
            raise ClinicalPreprocessingError("already fitted; create a fresh fold transformer")
        validate_fold_plan(plan)
        try:
            rows = [rows_by_patient[p.patient_id] for p in plan.inner_train]
        except KeyError:
            raise ClinicalPreprocessingError("missing inner-training clinical row") from None
        matrix = clinical_matrix(rows)
        fill = np.empty(len(FEATURE_NAMES), dtype=np.float64)
        for column, name in enumerate(FEATURE_NAMES):
            observed = matrix[:, column][~np.isnan(matrix[:, column])]
            if not observed.size:
                raise ClinicalPreprocessingError(f"{name} is entirely missing in training")
            if column in _NUMERIC:
                fill[column] = np.median(observed)
            else:
                codes, counts = np.unique(observed, return_counts=True)
                fill[column] = codes[np.argmax(counts)]
        imputed = np.where(np.isnan(matrix), fill, matrix)
        center = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            center[list(_NUMERIC)] = imputed[:, _NUMERIC].mean(axis=0)
            scale[list(_NUMERIC)] = imputed[:, _NUMERIC].std(axis=0)
        scale[scale == 0] = 1
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ClinicalPreprocessingError("nonfinite training scaling statistics")
        for array in (fill, center, scale):
            array.setflags(write=False)
        self._state = fill, center, scale
        return self

    def transform(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        if self._state is None:
            raise ClinicalPreprocessingError("fit on inner training before transform")
        matrix = clinical_matrix(rows)
        fill, center, scale = self._state
        with np.errstate(over="ignore", invalid="ignore"):
            result = ((np.where(np.isnan(matrix), fill, matrix) - center) / scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ClinicalPreprocessingError("nonfinite transformed clinical inputs")
        return result
