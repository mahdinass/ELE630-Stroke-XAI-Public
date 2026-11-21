"""Aggregate-only binary outcome metrics for one fixed CTP validation set."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


class CTPMetricError(ValueError):
    """Invalid aggregate metric input; messages contain no individual values."""


@dataclass(frozen=True)
class CTPAggregateMetrics:
    bce: float
    auroc: float
    average_precision: float
    brier_score: float
    balanced_accuracy_0_5: float
    mean_predicted_probability: float
    validation_count: int
    validation_good_count: int
    validation_poor_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def aggregate_ctp_metrics(labels: list[int], probabilities: list[float],
                          per_patient_bce: list[float]) -> CTPAggregateMetrics:
    """Return only aggregates; BCE inputs are per-case Torch BCEWithLogits values.

    AUROC uses average ranks for tied scores. AP integrates precision at each
    distinct score's recall increase. The sole decision threshold is p >= 0.5.
    Inputs remain transient and are never retained in the returned object.
    """
    try:
        y = np.asarray(labels)
        p = np.asarray(probabilities, dtype=np.float64)
        losses = np.asarray(per_patient_bce, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise CTPMetricError("validation metric inputs must be numeric") from None
    if (y.ndim != 1 or p.ndim != 1 or losses.ndim != 1 or len(y) == 0
            or len(p) != len(y) or len(losses) != len(y)
            or y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.isfinite(losses).all() or np.any(losses < 0)):
        raise CTPMetricError("validation metric inputs are invalid")
    n = len(y)
    poor = int(np.count_nonzero(y))
    good = n - poor
    if good == 0 or poor == 0:
        raise CTPMetricError("AUROC and average precision require both outcome classes")

    # Mann-Whitney AUROC with midranks gives 0.5 credit to equal-score pairs.
    ascending = np.argsort(p, kind="mergesort")
    sorted_p, sorted_y = p[ascending], y[ascending]
    positive_rank_sum = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_p[end] == sorted_p[start]:
            end += 1
        positive_rank_sum += int(np.count_nonzero(sorted_y[start:end])) * (start + 1 + end) / 2
        start = end
    auroc = (positive_rank_sum - poor * (poor + 1) / 2) / (poor * good)

    # Non-interpolated AP: sum precision at each distinct-score recall jump.
    descending = np.argsort(-p, kind="mergesort")
    sorted_p, sorted_y = p[descending], y[descending]
    average_precision = 0.0
    seen_positive = 0
    end = 0
    while end < n:
        start = end
        end += 1
        while end < n and sorted_p[end] == sorted_p[start]:
            end += 1
        gained = int(np.count_nonzero(sorted_y[start:end]))
        seen_positive += gained
        average_precision += (gained / poor) * (seen_positive / end)

    predicted_poor = p >= 0.5
    sensitivity = int(np.count_nonzero(predicted_poor & (y == 1))) / poor
    specificity = int(np.count_nonzero(~predicted_poor & (y == 0))) / good
    return CTPAggregateMetrics(
        bce=math.fsum(float(value) for value in losses) / n,
        auroc=float(auroc),
        average_precision=float(average_precision),
        brier_score=math.fsum(float(value) ** 2 for value in p - y) / n,
        balanced_accuracy_0_5=(sensitivity + specificity) / 2,
        mean_predicted_probability=math.fsum(float(value) for value in p) / n,
        validation_count=n,
        validation_good_count=good,
        validation_poor_count=poor,
    )
