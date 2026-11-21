"""UiS orchestration. No data access at import. Final never invokes preparation."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np

from . import fusion_runtime as runtime
from .heldout import digest, validate_state, clinical_transform, predict, metrics, report_rows
from .clinical_preprocessing import FEATURE_NAMES

PROTOCOL = 'docs/final_heldout_protocol.md'
BASE = runtime.RUNTIME / 'final_heldout'
_DEVICE = 'cuda'  # Internal test seam only; real context requires allocated CUDA.


def membership(plan):
    from .ncct_model_config import record_signature
    return {name: {'count': len(part), 'signature': record_signature({'patients': sorted(
        [[p.patient_id, p.cohort, p.outcome, p.fold, p.has_gt] for p in part])})}
        for name, part in (('inner_train', plan.inner_train), ('inner_validation', plan.inner_validation))}


def selected_clinical(patients):
    """Index feature cells only after canonical patient authorization."""
    from .clinical_runner import _identifier
    authorized = {p.patient_id for p in patients}
    selected, seen = {}, set()
    with (runtime.DATASET/'phenotypes/filtered_clinical_updated.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if len(header) != len(set(header)) or not {'patient_id', *FEATURE_NAMES}.issubset(header):
            raise ValueError('clinical header missing approved features or duplicate columns')
        id_col = header.index('patient_id')
        cols = {name: header.index(name) for name in FEATURE_NAMES}
        for cells in reader:
            if len(cells) != len(header):
                raise ValueError('clinical row width mismatch')
            pid = _identifier(cells[id_col])
            if not pid:
                continue
            if pid in seen:
                raise ValueError('clinical duplicate linkage')
            seen.add(pid)
            if pid in authorized:
                selected[pid] = {k: cells[i] for k, i in cols.items()}
    if set(selected) != authorized:
        raise ValueError('clinical authorized coverage incomplete')
    return selected


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    from .ncct_private_output import open_private_binary
    with open_private_binary(path) as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())


def identity(revision):
    result = runtime.source_identity(revision)
    result['heldout_protocol_sha256'] = file_hash(runtime.PROJECT/PROTOCOL)
    required = ['heldout.py', 'heldout_runner.py', 'heldout_prepare.py']
    tracked = runtime.git('ls-files').decode().splitlines()
    if any('src/ele630/'+p not in tracked for p in required) or PROTOCOL not in tracked:
        raise ValueError('commit all reviewed heldout implementation files first')
    return result


def bridge(payload):
    env = runtime.clinical_environment()
    env['ELE630_HELDOUT_PRIVATE_IPC'] = '1'
    proc = subprocess.run([str(runtime.PYTHON), '-B', '-m', 'ele630.heldout_prepare'],
                          input=json.dumps(payload, allow_nan=False), text=True,
                          capture_output=True, cwd=runtime.RUNTIME, env=env, timeout=1800)
    if proc.returncode or proc.stderr.strip():
        import re
        safe=proc.stderr.strip()
        if (safe.startswith('HELDOUT_PREPARATION_WORKER_FAILED ') and len(safe)<500
                and re.fullmatch(r'[a-zA-Z0-9_.:; =-]+',safe)):
            raise ValueError(safe)
        raise ValueError('clinical worker failed; verify pinned environment or original artifact spec')
    return json.loads(proc.stdout)


def context(revision, *, gpu=False):
    from .fusion_runner import load_plan, check_clinical_path
    from .fusion_checkpoints import load_ncct, load_ctp
    from .ncct_private_output import _validate_base
    ident = identity(revision)
    deps = runtime.environment(gpu=gpu)
    if gpu:
        import torch
        from .ncct_training import _seed_everything
        torch.set_num_threads(1)
        _seed_everything(630)
    deps['clinical'] = bridge({'probe': True})['dependencies']
    _validate_base(BASE)
    if not os.access(BASE,os.W_OK|os.X_OK):
        raise ValueError('private output base is not writable/searchable')
    contract, plan = load_plan()
    check_clinical_path()
    # Header only: no clinical cells, imaging voxels or predictions in preflight.
    with (runtime.DATASET/'phenotypes/filtered_clinical_updated.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if len(header) != len(set(header)) or not {'patient_id', *FEATURE_NAMES}.issubset(header):
            raise ValueError('clinical feature header invalid')
        from .clinical_runner import _identifier
        id_col=header.index('patient_id')
        seen=set()
        for cells in reader:
            if len(cells)!=len(header):
                raise ValueError('clinical CSV row width invalid')
            pid=_identifier(cells[id_col])
            if pid:
                if pid in seen:
                    raise ValueError('clinical CSV duplicate linkage')
                seen.add(pid)
        if not {p.patient_id for p in contract.patients}.issubset(seen):
            raise ValueError('clinical CSV missing canonical patients')
    tables = runtime.PROJECT/'outputs/tables'
    with (tables/'patient_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 150:
        raise ValueError('expected 150 imaging inventory records')
    # load_plan() already enforces paired modalities for the immutable eligible
    # cohort. Inventory-only patients must not acquire that eligibility rule.
    from .ncct import NCCTRepository
    from .ctp import RegisteredCTPRepository
    ncct, ctp = NCCTRepository(contract, runtime.DATASET), RegisteredCTPRepository(contract, runtime.DATASET)
    for patient in plan.outer_test:
        n, c = ncct.read_header(patient), ctp.read_header(patient)
        if len(n.shape) != 3 or min(n.shape) < 1 or c.frame_count != 40:
            raise ValueError('test image header dimensions invalid')
    loaded = {'ncct': load_ncct(plan), 'ctp': load_ctp(contract)}
    checkpoints = {k: v[1] for k, v in loaded.items()}
    binding = {'identity': ident, 'dependencies': deps, 'checkpoints': checkpoints,
               'membership': digest(asdict(plan)), 'files': {
                   'manifest': file_hash(tables/'patient_manifest.csv'),
                   'folds': file_hash(tables/'outcome_cv_folds_private.csv'),
                   'clinical': file_hash(runtime.DATASET/'phenotypes/filtered_clinical_updated.csv')}}
    return contract, plan, loaded, binding


def bundle_path(run_id):
    import re
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}', run_id or ''):
        raise ValueError('non-identifying run slug required')
    return BASE/run_id/'outer_fold_0'


def load_bundle(run_id, binding):
    path = bundle_path(run_id)
    frozen = runtime.read_private_json(path/'frozen.json')
    seal = runtime.read_private_json(path/'seal.json')
    if digest(frozen) != seal['frozen_sha256'] or frozen['binding'] != binding:
        raise ValueError('frozen bundle changed or source/data/dependency/checkpoint identity mismatch')
    validate_state(frozen['state'])
    if frozen['status'] != 'FROZEN' or set(frozen['development']) != {'ncct','ctp','clinical','ncct_clinical','ctp_clinical'}:
        raise ValueError('complete validated frozen bundle required')
    return frozen, seal


def preflight(run_id, revision):
    _, _, _, binding = context(revision)
    _, seal = load_bundle(run_id, binding)
    if not os.access(bundle_path(run_id),os.W_OK|os.X_OK):
        raise ValueError('frozen run output directory is not writable/searchable')
    if (BASE/'final_claim.json').exists():
        raise ValueError('a final attempt already exists; inspect status and use unchanged technical retry only')
    return {'status': 'PREFLIGHT_PASS', 'models': 5, 'train': 92, 'development': 23,
            'test': 28, 'image_voxels_accessed': False, 'performance_generated': False,
            'protocol_sha256': seal['frozen_sha256']}


def prepare(run_id, revision, *, originals=None, unavailable=False, prediction_reference=None):
    if (originals is None) == (not unavailable):
        raise ValueError('choose original spec OR explicitly attest originals unavailable')
    if (BASE/'final_claim.json').exists():
        raise ValueError('no preparation after any final test attempt')
    from .ncct_private_output import create_private_run
    contract, plan, loaded, binding = context(revision, gpu=True)
    destination = create_private_run(BASE, run_id, 0).fold_dir
    from .fusion_runner import real_features
    from .fusion_features import freeze
    train, val, imaging, before = {}, {}, {}, {}
    for mode in ('ncct','ctp'):
        model = freeze(loaded[mode][0])
        before[mode] = weight_digest(model)
        t, v, p = real_features(contract, plan, mode, model)
        train[mode], val[mode], imaging[mode] = t.tolist(), v.tolist(), p.tolist()
        if weight_digest(model) != before[mode]:
            raise ValueError('imaging parameters changed')
    payload = {'revision': revision, 'path': 'original' if originals else 'reconstruct',
               'train': train, 'validation': val, 'imaging': imaging,
               'checkpoints': binding['checkpoints']}
    if originals:
        payload['spec'] = runtime.read_private_json(Path(originals))
    result = bridge(payload)
    if prediction_reference:
        reference = runtime.read_private_json(Path(prediction_reference))
        from .heldout import compare_predictions
        if (set(reference) != {'source_revision','membership','probabilities'}
                or reference['source_revision'] != '54f289161a35206c00a517d7f12e2fdf1e7a4127'
                or reference['membership'] != membership(plan)
                or set(reference['probabilities']) != {'clinical','ncct_clinical','ctp_clinical'}):
            raise ValueError('original development prediction reference provenance mismatch')
        compare_predictions(result['clinical_probabilities'],reference['probabilities']['clinical'])
        for mode in ('ncct','ctp'):
            compare_predictions(result['probabilities'][mode],reference['probabilities'][mode+'_clinical'])
    from .xai_runtime import historical_gate
    development = {}
    y = np.array([int(p.outcome == 'poor') for p in plan.inner_validation], dtype=np.int64)
    # Exact private historical identity gates and fixed tolerances BEFORE test data.
    for mode in ('ncct','ctp'):
        historical = runtime.read_private_json(runtime.RUNTIME/'fusion_runs'/runtime.RUNS[mode]/'outer_fold_0/aggregate_training.json')
        if originals:
            # No new no-image fit in original reuse. Compare the actual clinical
            # prediction with the ORIGINAL historical zero-image control.
            result['results'][mode]['metrics']['no_image']=historical['results']['metrics']['no_image']
        historical_gate(result['results'][mode], mode, historical,
                        provenance=binding['checkpoints'][mode], membership=membership(plan))
        # Numerical state is also evaluated in the imaging interpreter/NumPy stack.
        rows = selected_clinical(plan.inner_validation)
        c = clinical_transform(result['state'], [rows[p.patient_id] for p in plan.inner_validation])
        from .heldout import compare_predictions
        compare_predictions(predict(result['state'], c, val[mode], mode), result['probabilities'][mode])
        compare_predictions(predict(result['state'], c), result['clinical_probabilities'])
        development[mode] = metrics(y, imaging[mode])
        development[mode+'_clinical'] = metrics(y, result['probabilities'][mode])
    development['clinical'] = metrics(y, result['clinical_probabilities'])
    # Recheck inputs/source after long preparation. No test inference occurs.
    if context(revision)[3] != binding:
        raise ValueError('source or inputs changed during preparation')
    frozen = {'status':'FROZEN', 'binding':binding, 'state':result['state'],
              'development':development, 'path':payload['path'],
              'configuration':{'train_count':92,'development_count':23,'test_count':28,
                  'seed':630,'threshold':.5,'primary_metric':'unweighted_bce',
                  'clinical_features':list(FEATURE_NAMES),'image_components':8,
                  'pca_solver':'full','pca_whiten':False,'logistic_C':1.,
                  'logistic_solver':'lbfgs','logistic_tol':1e-8,'logistic_max_iter':1000,
                  'ncct_window':[-20,120],'ctp_frames':40,'ctp_divisor':255},
              'acceptance': 'historical Stage5C3 5e-7; Stage7 probabilities 1e-10; cross-stack probabilities 1e-10',
              'original_case_predictions_available': bool(prediction_reference)}
    if prediction_reference:
        frozen['prediction_reference_sha256'] = file_hash(prediction_reference)
    if originals:
        frozen['original_spec_sha256'] = file_hash(originals)
    publish(destination/'frozen.json', frozen)
    publish(destination/'seal.json', {'frozen_sha256':digest(frozen)})
    return {'status':'FROZEN', 'models':5, 'test_evaluated':False}


def weight_digest(model):
    h = hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def acquire_lock(path):
    import fcntl
    descriptor = os.open(path, os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def final(run_id, revision, *, reviewed, retry=False):
    contract, plan, loaded, binding = context(revision, gpu=True)
    frozen, seal = load_bundle(run_id, binding)
    if reviewed != seal['frozen_sha256']:
        raise ValueError('explicit frozen protocol digest acknowledgement required')
    path = bundle_path(run_id)
    if (path/'aggregate_final.json').exists():
        raise ValueError('completed final evaluation cannot be repeated')
    claim = {'frozen_sha256':reviewed, 'run_id':run_id}
    if (BASE/'final_claim.json').exists():
        if not retry or runtime.read_private_json(BASE/'final_claim.json') != claim:
            raise ValueError('final attempt exists; only unchanged technical retry permitted')
    else:
        if retry:
            raise ValueError('technical retry requires original claim')
        publish(BASE/'final_claim.json', claim)
    # Cross-process lock, automatically released by the OS after technical failure.
    lock = acquire_lock(path/'execution.lock')
    try:
        if (path/'aggregate_final.json').exists():
            raise ValueError('completed final evaluation cannot be repeated')
        state = frozen['state']
        state_hash = digest(state)
        rows = selected_clinical(plan.outer_test)
        c = clinical_transform(state, [rows[p.patient_id] for p in plan.outer_test])
        # Test labels used only after all guards and exclusive claim.
        y = np.array([int(p.outcome == 'poor') for p in plan.outer_test], dtype=np.int64)
        results = {'clinical': metrics(y, predict(state,c))}
        from .fusion_features import freeze, extract
        from .ncct import NCCTRepository
        from .native_ncct_input import prepare_native_ncct_input
        from .ncct_intensity import clip_scale_ncct
        from .ctp import RegisteredCTPRepository
        import torch
        for mode in ('ncct','ctp'):
            model = freeze(loaded[mode][0]).to(_DEVICE)
            before = weight_digest(model)
            repository = NCCTRepository(contract,runtime.DATASET) if mode=='ncct' else RegisteredCTPRepository(contract,runtime.DATASET)
            embeddings, probabilities, losses = [], [], []
            for patient, target in zip(plan.outer_test, y):
                if mode=='ncct':
                    volume = repository.load(patient)
                    value = clip_scale_ncct(prepare_native_ncct_input(volume.data,volume.affine).tensor, lower=-20,upper=120)
                else:
                    value = repository.open_series(patient)
                e,p = extract(model,value,mode)
                with torch.inference_mode():
                    head = model.classifier if mode=='ncct' else model.head
                    logit = head(torch.from_numpy(e.copy()).reshape(1,128).to(_DEVICE)).reshape(1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor([float(target)],device=_DEVICE))
                embeddings.append(e)
                probabilities.append(p)
                losses.append(float(loss.item()))
            results[mode] = metrics(y, probabilities, losses)
            results[mode+'_clinical'] = metrics(y,predict(state,c,embeddings,mode))
            if before != weight_digest(model) or any(p.grad is not None for p in model.parameters()):
                raise ValueError('frozen model mutated')
            model.to('cpu')
        if digest(state)!=state_hash or context(revision)[3]!=binding:
            raise ValueError('frozen state/source/data changed during final evaluation')
        record = {'status':'FINAL_PASS','frozen_sha256':reviewed,
                  'source_revision':revision,'development':frozen['development'],'test':results,
                  'test_driven_tuning':False,'threshold':.5,
                  'evaluation':'independent_internal_fold0_test',
                  'checkpoint_hashes':{k:v['checkpoint_sha256'] for k,v in binding['checkpoints'].items()}}
        report_rows(record)
        publish(path/'aggregate_final.json',record)
        return {'status':'FINAL_PASS','models':5,'test_count':28}
    finally:
        os.close(lock)


def discover(revision):
    """Read-only bounded trusted runtime walk. No deserialization or patient reads.

    Detailed paths stay in the UiS terminal; do not paste this inventory remotely.
    """
    import re
    identity(revision)
    extensions={'.joblib','.pkl','.pickle','.npz','.json','.pt','.pth'}
    found=[]
    for directory, dirs, files in os.walk(runtime.RUNTIME, followlinks=False):
        rel=Path(directory).relative_to(runtime.RUNTIME)
        dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d not in {'logs','cache','__pycache__'}
                       and not (Path(directory)/d).is_symlink() and len(rel.parts)<6)
        for name in sorted(files):
            path=Path(directory)/name
            if path.suffix in extensions and not path.is_symlink():
                relative=str(path.relative_to(runtime.RUNTIME))
                if re.search(r'sub[-_]stroke|stroke[-_]0?[123][-_][0-9]{3}|fictional-',relative,re.I):
                    relative='<patient-like filename withheld>'
                found.append({'relative_path':relative,
                              'bytes':path.stat().st_size,'sha256':file_hash(path)})
    return {'status':'DISCOVERY_ONLY','private_terminal_output':True,'files':found,
            'deserialized':False,'test_evaluated':False}


def verify_smoke(revision):
    record=runtime.read_private_json(BASE/('synthetic_smoke_'+revision+'.json'))
    if record != {'status':'SMOKE_PASS','identity':identity(revision),
                  'dependencies':runtime.environment(gpu=False),'patient_data_access':False}:
        raise ValueError('same-source and same-stack synthetic smoke required')
    return record


def smoke(revision):
    """Allocated GPU, fictional inputs only, pinned private clinical IPC as in real preparation."""
    import io
    import sys
    import unittest
    from .ncct_private_output import _validate_base
    ident=identity(revision)
    deps=runtime.environment(gpu=True)
    _validate_base(BASE)
    sys.path.insert(0,str(runtime.PROJECT/'tests'))
    from test_final_heldout import CoreTests, TorchTests
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                             for cls in (CoreTests,TorchTests))
    tested=unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    if not tested.wasSuccessful() or tested.skipped:
        raise ValueError('synthetic core/GPU tests failed or skipped')
    bridge({'synthetic_tests':True})
    from .fusion_synthetic import synthetic_embeddings
    payload={'revision':revision,'path':'reconstruct','synthetic':True,
             'train':{},'validation':{},'imaging':{}}
    for mode in ('ncct','ctp'):
        _,t,v,p=synthetic_embeddings(mode,'cuda')
        payload['train'][mode],payload['validation'][mode],payload['imaging'][mode]=t.tolist(),v.tolist(),p.tolist()
    result=bridge(payload)
    validate_state(result['state'])
    from .fusion_synthetic import fictional_plan,fictional_rows
    from .heldout import compare_predictions
    plan=fictional_plan()
    rows=fictional_rows(plan)
    c=clinical_transform(result['state'],[rows[p.patient_id] for p in plan.inner_validation])
    for mode in ('ncct','ctp'):
        compare_predictions(predict(result['state'],c,payload['validation'][mode],mode),result['probabilities'][mode])
    record={'status':'SMOKE_PASS','identity':ident,'dependencies':deps,'patient_data_access':False}
    publish(BASE/('synthetic_smoke_'+revision+'.json'),record)
    return {'status':'SMOKE_PASS','patient_data_access':False}
