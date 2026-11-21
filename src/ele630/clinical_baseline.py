"""Stage 5A: fixed fold-0 clinical development; aggregate results only.

No file I/O, replacement splits, checkpoints, tuning or outer-test feature
access. The caller supplies the existing trusted FoldPlan and an in-memory
row mapping (synthetic locally; an authorized server adapter later).
scikit-learn is loaded lazily; no alternate optimizer is provided.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .clinical_preprocessing import FEATURE_NAMES, ClinicalPreprocessor
from .ctp_metrics import CTPAggregateMetrics, aggregate_ctp_metrics
from .data_contract import ACTIVE_PROFILE, _fingerprint
from .ncct_splits import FoldPlan, binary_outcome, validate_fold_plan


class ClinicalBaselineError(RuntimeError):
    """Stopped clinical development; errors never contain source values or IDs."""


@dataclass(frozen=True)
class ClinicalBaselineResult:
    training_count: int
    training_poor_prevalence: float
    logistic_regression: CTPAggregateMetrics
    training_prior: CTPAggregateMetrics

    def as_dict(self) -> dict:
        """Only aggregate counts and metrics, never model or preprocessing state."""
        return asdict(self)


def _validate_development_plan(plan: FoldPlan) -> None:
    try:
        validate_fold_plan(plan)
        if (plan.outer_fold != 0 or type(plan.seed) is not int or plan.seed != 630
                or plan.validation_fraction != 0.20):
            raise ValueError
        patients = plan.inner_train + plan.inner_validation + plan.outer_test
        if any(not isinstance(p.patient_id, str) or not p.patient_id or
               type(p.fold) is not int or p.fold not in range(5) for p in patients):
            raise ValueError
        # Same synthetic-contract convention as the imaging development modules.
        # A 143-patient plan must agree with the immutable active outer contract.
        if len(patients) == ACTIVE_PROFILE.patients and (
            tuple(map(len, (plan.inner_train, plan.inner_validation, plan.outer_test))) != (92, 23, 28)
            or _fingerprint({p.patient_id: p.fold for p in patients}) != ACTIVE_PROFILE.fingerprint
            or sum(p.outcome == "poor" for p in patients) != ACTIVE_PROFILE.poor
            or sum(p.has_gt for p in patients) != ACTIVE_PROFILE.with_gt
            or tuple(sum(p.fold == f for p in patients) for f in range(5)) != ACTIVE_PROFILE.fold_sizes
            or tuple(sum(p.fold == f and p.outcome == "poor" for p in patients)
                     for f in range(5)) != ACTIVE_PROFILE.poor_per_fold
        ):
            raise ValueError
    except Exception:
        raise ClinicalBaselineError("a valid existing fold-0 development plan is required") from None


def aggregate_probability_metrics(labels, probabilities) -> CTPAggregateMetrics:
    """Reuse imaging metrics, with float64 BCE from probabilities.

    Clip only the BCE calculation to [float64 eps, 1-eps], making exact 0/1
    predictions finite. Original probabilities drive ranks, Brier and threshold
    metrics. This is a fixed numerical rule, independent of validation outcomes.
    """
    try:
        y = np.asarray(labels)
        p = np.asarray(probabilities, dtype=np.float64)
        if (y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not y.size
                or y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all()
                or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
            raise ValueError
        eps = np.finfo(np.float64).eps
        clipped = np.clip(p, eps, 1 - eps)
        losses = np.where(y == 1, -np.log(clipped), -np.log1p(-clipped))
        return aggregate_ctp_metrics(y.tolist(), p.tolist(), losses.tolist())
    except Exception:
        raise ClinicalBaselineError("invalid aggregate clinical evaluation inputs") from None


def evaluate_training_prior(plan: FoldPlan) -> CTPAggregateMetrics:
    """Constant probability from inner-training labels; no clinical row access."""
    _validate_development_plan(plan)
    prior = sum(binary_outcome(p) for p in plan.inner_train) / len(plan.inner_train)
    labels = [binary_outcome(p) for p in plan.inner_validation]
    return aggregate_probability_metrics(labels, np.full(len(labels), prior))


def _make_logistic_regression():
    """Fixed configuration, compatible with the proposed scikit-learn 1.5.2 gate."""
    try:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            penalty="l2", C=1.0, fit_intercept=True, class_weight=None,
            solver="lbfgs", max_iter=1000, tol=1e-8, random_state=630,
            warm_start=False, verbose=0,
        )
    except Exception:
        raise ClinicalBaselineError(
            "clinical logistic regression requires a working scikit-learn/SciPy installation"
        ) from None


def _selected_rows(patients, rows_by_patient):
    """Lookup requested IDs only; discard extra columns without reading them."""
    try:
        selected = {}
        for patient in patients:
            source = rows_by_patient[patient.patient_id]
            if not isinstance(source, Mapping):
                raise TypeError
            selected[patient.patient_id] = {name: source[name] for name in FEATURE_NAMES}
        return selected
    except Exception:
        raise ClinicalBaselineError("authorized clinical rows could not be read") from None


def run_clinical_development(
    plan: FoldPlan, rows_by_patient: Mapping[str, Mapping[str, object]],
) -> ClinicalBaselineResult:
    """Fit once on inner_train; predict once on inner_validation; return aggregates.

    Supply the existing plan from the validated dataset contract. This function
    never iterates the source mapping, accepts no arbitrary evaluation patients,
    and never requests outer-test clinical rows. All model/state/predictions stay
    transient. Warnings cause a safe failure (including nonconvergence), with no
    automatic retry, iteration increase or configuration selection.
    """
    _validate_development_plan(plan)
    # Fail a missing dependency before accessing any clinical rows.
    with warnings.catch_warnings(record=True) as dependency_warnings:
        warnings.simplefilter("always")
        model = _make_logistic_regression()
    if dependency_warnings:
        raise ClinicalBaselineError("clinical estimator initialization emitted a warning")
    train_rows = _selected_rows(plan.inner_train, rows_by_patient)
    try:
        preprocessing = ClinicalPreprocessor().fit_training(plan, train_rows)
        x_train = preprocessing.transform([train_rows[p.patient_id] for p in plan.inner_train])
    except Exception:
        raise ClinicalBaselineError("clinical training preprocessing failed") from None
    y_train = np.array([binary_outcome(p) for p in plan.inner_train], dtype=np.int64)
    try:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(x_train, y_train)
        if fit_warnings:
            raise ValueError
        if (not np.array_equal(model.classes_, [0, 1])
                or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()):
            raise ValueError
    except Exception:
        raise ClinicalBaselineError("clinical logistic regression fitting failed") from None
    # Validation features are not requested until preprocessing and model fit end.
    validation_rows = _selected_rows(plan.inner_validation, rows_by_patient)
    try:
        x_validation = preprocessing.transform(
            [validation_rows[p.patient_id] for p in plan.inner_validation])
        with warnings.catch_warnings(record=True) as prediction_warnings:
            warnings.simplefilter("always")
            probabilities = np.asarray(model.predict_proba(x_validation), dtype=np.float64)
        if (prediction_warnings or probabilities.shape != (len(plan.inner_validation), 2)
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError
    except Exception:
        raise ClinicalBaselineError("clinical validation prediction failed") from None
    y_validation = [binary_outcome(p) for p in plan.inner_validation]
    prior = float(y_train.mean())
    return ClinicalBaselineResult(
        len(y_train), prior,
        aggregate_probability_metrics(y_validation, probabilities[:, 1]),
        evaluate_training_prior(plan),
    )
