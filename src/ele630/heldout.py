"""Load/transform/predict and aggregate primitives. No fitting or Torch imports."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .clinical_preprocessing import FEATURE_NAMES, ClinicalPreprocessor
from .clinical_baseline import aggregate_probability_metrics
from .ctp_metrics import aggregate_ctp_metrics

MODELS = ('ncct', 'ctp', 'clinical', 'ncct_clinical', 'ctp_clinical')
SHAPES = {'fill': (11,), 'center': (11,), 'scale': (11,),
          'clinical_coef': (11,), 'clinical_intercept': (1,)}
IMAGE_SHAPES = {'mean': (128,), 'scale': (128,), 'pca_mean': (128,),
                'components': (8, 128), 'coef': (19,), 'intercept': (1,)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def arrays(value, shapes):
    if not isinstance(value, dict) or set(value) != set(shapes):
        raise ValueError('frozen state schema mismatch')
    result = {k: np.asarray(value[k], dtype=np.float64) for k in shapes}
    if any(v.shape != shapes[k] or not np.isfinite(v).all() for k, v in result.items()):
        raise ValueError('frozen state dimensions or finite values invalid')
    if 'scale' in result and np.any(result['scale'] <= 0):
        raise ValueError('frozen scales must be positive')
    return result


def validate_state(state):
    if set(state) != {'clinical', 'ncct', 'ctp', 'features'} or state['features'] != list(FEATURE_NAMES):
        raise ValueError('complete five-model state and feature order required')
    arrays(state['clinical'], SHAPES)
    for mode in ('ncct', 'ctp'):
        a = arrays(state[mode], IMAGE_SHAPES)
        if not np.allclose(a['components'] @ a['components'].T, np.eye(8), atol=1e-8, rtol=0):
            raise ValueError('PCA components must be orthonormal')
    return state


def clinical_transform(state, rows):
    a = arrays(state['clinical'], SHAPES)
    transformer = ClinicalPreprocessor()
    # Restore the existing transform implementation; no fit constructor is called.
    transformer._state = tuple(a[k].copy() for k in ('fill', 'center', 'scale'))
    return transformer.transform(rows)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -z))


def predict(state, clinical, embeddings=None, mode=None):
    c = np.asarray(clinical)
    if c.ndim != 2 or c.shape[1] != 11 or not np.isfinite(c).all():
        raise ValueError('clinical interface requires finite n by 11')
    if mode is None:
        a = arrays(state['clinical'], SHAPES)
        return sigmoid(c @ a['clinical_coef'] + a['clinical_intercept'][0])
    if mode not in ('ncct', 'ctp'):
        raise ValueError('unknown modality')
    a = arrays(state[mode], IMAGE_SHAPES)
    e = np.asarray(embeddings, dtype=np.float64)
    if e.shape != (len(c), 128) or not np.isfinite(e).all():
        raise ValueError('embedding interface requires finite n by 128')
    pcs = (((e-a['mean'])/a['scale'])-a['pca_mean']) @ a['components'].T
    return sigmoid(np.column_stack((pcs, c)) @ a['coef'] + a['intercept'][0])


def metrics(y, p, losses=None):
    record = (aggregate_probability_metrics(y, p) if losses is None else
              aggregate_ctp_metrics(list(y), list(p), list(losses))).as_dict()
    for old, new in [('validation_count', 'count'), ('validation_good_count', 'good'),
                     ('validation_poor_count', 'poor')]:
        record[new] = record.pop(old)
    y, predicted = np.asarray(y), np.asarray(p) >= .5
    record['confusion_matrix'] = [[int(np.sum((y == actual) & (predicted == pred)))
                                   for pred in (0, 1)] for actual in (0, 1)]
    return record


def compare_predictions(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=1e-10, rtol=0):
        raise ValueError('fixed development probability equivalence failed')


def report_rows(record):
    if record.get('status') != 'FINAL_PASS' or set(record['test']) != set(MODELS):
        raise ValueError('complete final aggregate required')
    rows = []
    for model in MODELS:
        for cohort in ('development', 'test'):
            m = record[cohort][model]
            count = 23 if cohort == 'development' else 28
            expected_classes=(16,7) if cohort=='development' else (19,9)
            if m['count'] != count or (m['good'],m['poor'])!=expected_classes:
                raise ValueError('aggregate cohort count mismatch')
            for k in ('bce', 'auroc', 'average_precision', 'brier_score', 'balanced_accuracy_0_5'):
                if not np.isfinite(m[k]) or m[k] < -1e-12 or (k != 'bce' and m[k] > 1+1e-12):
                    raise ValueError('aggregate metric range invalid')
            cm=np.asarray(m['confusion_matrix'])
            if (cm.shape!=(2,2) or cm.dtype.kind not in 'iu' or np.any(cm<0)
                    or int(cm.sum())!=count or list(cm.sum(axis=1))!=[m['good'],m['poor']]
                    or m['good']<=0 or m['poor']<=0
                    or abs((cm[0,0]/m['good']+cm[1,1]/m['poor'])/2-m['balanced_accuracy_0_5'])>1e-12):
                raise ValueError('aggregate confusion matrix inconsistent')
            rows.append((model, cohort, m))
    return rows
