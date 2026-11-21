"""Private, train-only reconstruction of the Stage 5C3 lightweight estimators.

No Torch dependency, file I/O or estimator export. Returned arrays are PRIVATE
IPC/memory values and must never be included in aggregate reports.
"""
from __future__ import annotations

import numpy as np
import warnings

from .clinical_baseline import _validate_development_plan
from .clinical_preprocessing import ClinicalPreprocessor
from .fusion import ImagePCA, _fit, matrix, metric_record
from .ncct_splits import binary_outcome


STATE_SHAPES = {"mean": (128,), "scale": (128,), "pca_mean": (128,),
                "components": (8, 128), "coef": (19,), "intercept": (1,)}


def validate_state(state):
    if not isinstance(state, dict) or set(state) != set(STATE_SHAPES):
        raise ValueError("invalid private fusion state schema")
    result = {k: np.array(state[k], dtype=np.float64, copy=True) for k in state}
    if any(v.shape != STATE_SHAPES[k] or not np.isfinite(v).all() for k, v in result.items()):
        raise ValueError("invalid private fusion state values")
    if np.any(result["scale"] <= 0):
        raise ValueError("positive training scaler scales required")
    return result


def reconstruct(plan, rows, training_embeddings, validation_embeddings, imaging_probabilities):
    """Fit original constructors in original order before reading validation rows.

    Caller must first independently verify original membership for real data.
    The synthetic convention of existing FoldPlan validators is retained here.
    """
    _validate_development_plan(plan)
    train = matrix(training_embeddings, len(plan.inner_train))
    validation = matrix(validation_embeddings, len(plan.inner_validation))
    preprocessor = ClinicalPreprocessor().fit_training(plan, rows)
    c_train = preprocessor.transform([rows[p.patient_id] for p in plan.inner_train])
    transform = ImagePCA().fit(train)
    p_train = transform.transform(train)
    y_train = np.array([binary_outcome(p) for p in plan.inner_train], dtype=np.int64)
    clinical = _fit(c_train, y_train)
    fusion = _fit(np.column_stack((p_train, c_train)), y_train)
    no_image = _fit(np.column_stack((np.zeros((len(train), 8)), c_train)), y_train)
    c_val = preprocessor.transform([rows[p.patient_id] for p in plan.inner_validation])
    features = np.column_stack((transform.transform(validation), c_val))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logits = fusion.decision_function(features)
        probabilities = fusion.predict_proba(features)[:, 1]
        clinical_probabilities = clinical.predict_proba(c_val)[:, 1]
        clinical_logits = clinical.decision_function(c_val)
        no_image_probabilities = no_image.predict_proba(np.column_stack((np.zeros((len(validation), 8)), c_val)))[:, 1]
    if caught:
        raise ValueError("reconstructed prediction emitted a warning")
    y_val = np.array([binary_outcome(p) for p in plan.inner_validation], dtype=np.int64)
    metrics = {name: metric_record(y_val, p) for name, p in (
        ("fusion", probabilities), ("clinical_only", clinical_probabilities),
        ("imaging_only", imaging_probabilities),
        ("no_image", no_image_probabilities),
    )}
    state = validate_state({"mean": transform.scaler.mean_, "scale": transform.scaler.scale_,
                            "pca_mean": transform.pca.mean_, "components": transform.pca.components_,
                            "coef": fusion.coef_[0], "intercept": fusion.intercept_})
    return {"state": {k: v.tolist() for k, v in state.items()},
            "clinical_validation": c_val.tolist(), "logits": logits.tolist(),
            "clinical_logits": clinical_logits.tolist(),
            "probabilities": probabilities.tolist(), "metrics": metrics}
