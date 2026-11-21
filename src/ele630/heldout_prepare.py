"""Private clinical worker: original artifact import OR authorized train92 fitting.

Never invoked by final evaluation. stdout is private IPC, never a terminal.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
import numpy as np

from .clinical_preprocessing import ClinicalPreprocessor, FEATURE_NAMES
from .fusion import ImagePCA, _fit, metric_record
from .heldout import validate_state, predict, compare_predictions, clinical_transform
from .ncct_splits import binary_outcome


def export_clinical(pre, model):
    if pre._state is None or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('fitted clinical artifacts required')
    return {**{k: v.tolist() for k, v in zip(('fill', 'center', 'scale'), pre._state)},
            'clinical_coef': model.coef_[0].tolist(),
            'clinical_intercept': model.intercept_.tolist()}


def export_image(scaler, pca, model):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('fitted binary fusion estimator required')
    return {k: np.asarray(v).tolist() for k, v in {
        'mean': scaler.mean_, 'scale': scaler.scale_, 'pca_mean': pca.mean_,
        'components': pca.components_, 'coef': model.coef_[0],
        'intercept': model.intercept_}.items()}


def fit_state(plan, rows, train):
    """Original Stage7 fit order, train92 only. No validation/test rows requested."""
    pre = ClinicalPreprocessor().fit_training(plan, rows)
    c = pre.transform([rows[p.patient_id] for p in plan.inner_train])
    y = np.array([binary_outcome(p) for p in plan.inner_train], dtype=np.int64)
    models, transforms, state = {}, {}, {'features': list(FEATURE_NAMES)}
    for mode in ('ncct', 'ctp'):
        transform = ImagePCA().fit(np.asarray(train[mode]))
        clinical = _fit(c, y)
        fused = _fit(np.column_stack((transform.transform(train[mode]), c)), y)
        # Preserve original construction order including the fixed no-image control.
        _fit(np.column_stack((np.zeros((len(c), 8)), c)), y)
        exported = export_clinical(pre, clinical)
        if 'clinical' in state and state['clinical'] != exported:
            raise ValueError('matched clinical estimator disagreement')
        state['clinical'] = exported
        state[mode] = export_image(transform.scaler, transform.pca, fused)
        models[mode], transforms[mode] = fused, transform
    return validate_state(state)


def original_state(spec, membership, checkpoints):
    """Explicitly selected trusted local joblib files only, after provenance checks.

    Never probe arbitrary pickle objects during discovery. Spec entries must
    name original files; converting an unknown format is a blocker, not refitting.
    """
    import hashlib
    import joblib
    from pathlib import Path
    from . import fusion_runtime as runtime
    from .clinical_baseline import _make_logistic_regression
    expected = {'clinical_preprocessor', 'clinical_model', 'ncct_scaler', 'ncct_pca',
                'ncct_model', 'ctp_scaler', 'ctp_pca', 'ctp_model'}
    if (set(spec) != {'schema', 'source_revision', 'membership', 'checkpoints', 'files'}
            or spec['schema'] != 1 or spec['source_revision'] != '54f289161a35206c00a517d7f12e2fdf1e7a4127'
            or spec['membership'] != membership or spec['checkpoints'] != checkpoints
            or set(spec['files']) != expected):
        raise ValueError('original artifact provenance spec mismatch')
    objects = {}
    for name in sorted(expected):
        entry = spec['files'][name]
        if set(entry) != {'path', 'sha256'}:
            raise ValueError('original file identity required')
        path = runtime.private_file(Path(entry['path']))
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('original artifact checksum mismatch')
        objects[name] = joblib.load(path)  # Explicit trusted-file opt-in, no discovery deserialization.
    if type(objects['clinical_preprocessor']) is not ClinicalPreprocessor:
        raise ValueError('unsupported original clinical preprocessor type')
    params = _make_logistic_regression().get_params()
    for name in ('clinical_model', 'ncct_model', 'ctp_model'):
        if type(objects[name]) is not type(_make_logistic_regression()) or objects[name].get_params() != params:
            raise ValueError('original logistic configuration mismatch')
    state = {'features': list(FEATURE_NAMES), 'clinical': export_clinical(
        objects['clinical_preprocessor'], objects['clinical_model'])}
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    for mode in ('ncct', 'ctp'):
        scaler, pca = objects[mode+'_scaler'], objects[mode+'_pca']
        if (type(scaler) is not StandardScaler or not scaler.with_mean or not scaler.with_std
                or type(pca) is not PCA or pca.n_components != 8 or pca.whiten
                or pca.svd_solver != 'full' or pca.random_state != 630
                or np.any(np.asarray(scaler.n_samples_seen_) != 92) or pca.n_samples_ != 92):
            raise ValueError('original image preprocessing configuration mismatch')
        state[mode] = export_image(scaler, pca, objects[mode+'_model'])
    return validate_state(state)


def process(payload):
    from .fusion_runner import load_plan, check_clinical_path
    from .heldout_runner import selected_clinical
    from . import fusion_runtime as runtime
    from .heldout_runner import membership
    runtime.source_identity(payload['revision'])
    if payload.get('synthetic') is True:
        from .fusion_synthetic import fictional_plan, fictional_rows
        plan = fictional_plan()
        rows = fictional_rows(plan)
    else:
        _, plan = load_plan()
        check_clinical_path()
        rows = selected_clinical(plan.inner_train + plan.inner_validation)
    if payload['path'] == 'reconstruct':
        state = fit_state(plan, rows, payload['train'])
    elif payload['path'] == 'original':
        state = original_state(payload['spec'], membership(plan), payload['checkpoints'])
    else:
        raise ValueError('unknown preparation path')
    c = clinical_transform(state, [rows[p.patient_id] for p in plan.inner_validation])
    y = np.array([binary_outcome(p) for p in plan.inner_validation], dtype=np.int64)
    result, probabilities = {}, {}
    for mode in ('ncct', 'ctp'):
        p = predict(state, c, payload['validation'][mode], mode)
        pc = predict(state, c)
        result[mode] = {'metrics': {
            'fusion': metric_record(y, p), 'clinical_only': metric_record(y, pc),
            'imaging_only': metric_record(y, payload['imaging'][mode])}}
        if payload['path'] == 'reconstruct':
            # Independent unchanged Stage7 implementation, probability-level check.
            from .xai_reconstruction import reconstruct
            reference = reconstruct(plan, rows, payload['train'][mode],
                                    payload['validation'][mode], payload['imaging'][mode])
            compare_predictions(p, reference['probabilities'])
            result[mode]['metrics']['no_image']=reference['metrics']['no_image']
        probabilities[mode] = p.tolist()
    return {'state': state, 'results': result, 'probabilities': probabilities,
            'clinical_probabilities': predict(state, c).tolist()}


def main():
    try:
        if os.environ.get('ELE630_HELDOUT_PRIVATE_IPC') != '1' or sys.stdin.isatty() or sys.stdout.isatty():
            raise ValueError('private parent-owned pipes required')
        from .fusion_clinical_worker import dependencies
        dependencies()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            payload = json.load(sys.stdin)
            if payload == {'probe': True}:
                result = {'dependencies': dependencies()}
            elif payload == {'synthetic_tests': True}:
                import io
                import unittest
                from .fusion_runtime import PROJECT
                sys.path.insert(0,str(PROJECT/'tests'))
                from test_final_heldout import FitTests
                tested=unittest.TextTestRunner(stream=io.StringIO()).run(
                    unittest.defaultTestLoader.loadTestsFromTestCase(FitTests))
                if not tested.wasSuccessful() or tested.skipped or tested.testsRun!=3:
                    raise ValueError('clinical original-import/reconstruction synthetic tests failed or skipped')
                result={'tests':tested.testsRun,'skipped':0}
            else:
                result = process(payload)
        if caught:
            raise ValueError('preparation emitted warnings')
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        import traceback
        from pathlib import Path
        frames=traceback.extract_tb(error.__traceback__)
        last=frames[-1] if frames else None
        name=Path(last.filename).name if last else 'environment'
        detail=str(error) if (last and name in {'heldout_prepare.py','heldout.py','heldout_runner.py'}
                             and type(error) is ValueError) else 'guard or dependency check failed'
        print('HELDOUT_PREPARATION_WORKER_FAILED type='+type(error).__name__+
              ' at='+name+':'+str(last.lineno if last else 0)+'; '+detail,file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
