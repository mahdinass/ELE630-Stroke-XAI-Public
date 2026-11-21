"""Stage 5C3 proposal v1: train-only PCA and fixed logistic fusion.

No file I/O, parameter search, checkpoint export, or outer-test feature access.
All fitted objects and individual outputs remain transient.
"""
from __future__ import annotations

import warnings
import numpy as np

from .clinical_baseline import (
    _make_logistic_regression, _validate_development_plan,
    aggregate_probability_metrics,
)
from .clinical_preprocessing import ClinicalPreprocessor
from .ncct_splits import binary_outcome


class FusionError(ValueError):
    """Sanitized contract failure."""


def matrix(value, count):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (count, 128) or not np.isfinite(value).all():
        raise FusionError("finite patient embeddings of shape (n,128) required")
    return value


class ImagePCA:
    """One fit: population standardization then full-SVD PCA, no whitening.

    No post-PCA rescaling: component variance is intentionally preserved.
    Fewer than eight nonzero training singular values is a hard failure.
    """
    def __init__(self):
        self.scaler = self.pca = None

    def fit(self, embeddings):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        x = matrix(embeddings, len(embeddings))
        if self.scaler is not None or len(x) < 9:
            raise FusionError("PCA requires a fresh transformer and at least nine training patients")
        scaler = StandardScaler(with_mean=True, with_std=True)
        pca = PCA(n_components=8, svd_solver="full", whiten=False, random_state=630)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            standardized = scaler.fit_transform(x)
            pca.fit(standardized)
        tolerance = np.finfo(np.float64).eps * max(x.shape) * pca.singular_values_[0]
        if caught or not np.isfinite(pca.components_).all() or pca.singular_values_[-1] <= tolerance:
            raise FusionError("training image representation cannot support eight PCA components")
        self.scaler, self.pca = scaler, pca
        return self

    def transform(self, embeddings):
        if self.scaler is None:
            raise FusionError("fit PCA on training patients first")
        result = self.pca.transform(self.scaler.transform(matrix(embeddings, len(embeddings))))
        if result.shape != (len(embeddings), 8) or not np.isfinite(result).all():
            raise FusionError("invalid PCA output")
        return result


def _fit(x, y):
    model = _make_logistic_regression()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    if (caught or not np.array_equal(model.classes_, [0, 1])
            or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()):
        raise FusionError("fixed logistic estimator failed or did not converge")
    return model


def metric_record(y, p):
    record = aggregate_probability_metrics(y, p).as_dict()
    record.update(probability_sd=float(np.std(p)),
                  probability_min=float(np.min(p)), probability_max=float(np.max(p)))
    return record


def run_fusion(plan, rows, training_embeddings, validation_embeddings, imaging_probabilities):
    """Matched development comparison; supplied arrays follow canonical plan order.

    Fit every transformation and coefficient before requesting validation rows.
    Only aggregate metrics escape. No estimator/preprocessor is returned.
    """
    _validate_development_plan(plan)
    train = matrix(training_embeddings, len(plan.inner_train))
    validation = matrix(validation_embeddings, len(plan.inner_validation))
    preprocessor = ClinicalPreprocessor().fit_training(plan, rows)
    clinical_train = preprocessor.transform([rows[p.patient_id] for p in plan.inner_train])
    image_transform = ImagePCA().fit(train)
    image_train = image_transform.transform(train)
    y_train = np.array([binary_outcome(p) for p in plan.inner_train], dtype=np.int64)
    clinical = _fit(clinical_train, y_train)
    fusion = _fit(np.column_stack((image_train, clinical_train)), y_train)
    no_image = _fit(np.column_stack((np.zeros((len(train), 8)), clinical_train)), y_train)
    clinical_validation = preprocessor.transform([rows[p.patient_id] for p in plan.inner_validation])
    image_validation = image_transform.transform(validation)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p_clinical = clinical.predict_proba(clinical_validation)[:, 1]
        p_fusion = fusion.predict_proba(np.column_stack((image_validation, clinical_validation)))[:, 1]
        p_no_image = no_image.predict_proba(np.column_stack((np.zeros((len(validation), 8)), clinical_validation)))[:, 1]
    if caught:
        raise FusionError("prediction emitted a warning")
    prior = float(y_train.mean())
    y = np.array([binary_outcome(p) for p in plan.inner_validation], dtype=np.int64)
    results = {name: metric_record(y, np.asarray(p, dtype=np.float64)) for name, p in (
        ("imaging_only", imaging_probabilities), ("fusion", p_fusion),
        ("clinical_only", p_clinical), ("no_image", p_no_image),
        ("clinical_prior_shrinkage_0_5", 0.5 * p_clinical + 0.5 * prior),
        ("training_prior", np.full(len(y), prior)),
    )}
    return {
        "metrics": results,
        "fusion_minus_reference": {name: {
            key: results["fusion"][key] - results[name][key]
            for key in ("bce", "auroc", "average_precision", "brier_score", "balanced_accuracy_0_5", "probability_sd")
        } for name in ("imaging_only", "clinical_only", "no_image", "clinical_prior_shrinkage_0_5")},
        "training_count": len(train), "validation_count": len(validation),
        "image_components": 8, "clinical_features": 11, "fusion_features": 19,
        "outer_test_evaluated": False,
    }
