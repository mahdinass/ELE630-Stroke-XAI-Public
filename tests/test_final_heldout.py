"""Fictional-only held-out workflow checks; never opens UiS medical data."""
from __future__ import annotations
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ele630 import heldout as h, heldout_runner as r
from ele630.fusion_synthetic import fictional_plan, fictional_rows


def state():
    clinical={'fill':[0.]*11,'center':[0.]*11,'scale':[1.]*11,
              'clinical_coef':[.01]*11,'clinical_intercept':[.1]}
    image={'mean':[0.]*128,'scale':[1.]*128,'pca_mean':[0.]*128,
           'components':np.eye(128)[:8].tolist(),'coef':[.01]*19,'intercept':[.2]}
    return {'features':list(h.FEATURE_NAMES),'clinical':clinical,
            'ncct':copy.deepcopy(image),'ctp':copy.deepcopy(image)}


class CoreTests(unittest.TestCase):
    def _inventory_context(self, *, count=150, excluded_missing=(),
                           eligible_missing=None, eligible_record=1):
        """Real CSV contract/planner with a scaled fictional eligible cohort.

        Only server/environment/checkpoint/header boundaries are replaced. The
        150-row inventory and original eligibility/fold validation execute.
        """
        import csv
        from test_data_contract import fixture_rows, synthetic_fingerprint, MANIFEST_FIELDS, FOLD_FIELDS
        from ele630.data_contract import ValidationProfile, load_and_validate
        from ele630.ncct_splits import plan_outer_fold
        manifest, folds=fixture_rows()
        manifest.extend(dict(patient_id='fictional-excluded-'+str(i),cohort='03',
            has_ncct_skull_stripped=1,has_ctp_registered=1,has_ctp_registered_mask=0,
            mrs_3months='',mrs_binary='',eligible_fusion_mrs=0) for i in range(count-10))
        for field in excluded_missing:
            manifest[-1][field]=0
        if eligible_missing:
            manifest[0][eligible_missing]=0
            manifest[0]['eligible_fusion_mrs']=eligible_record
        profile=ValidationProfile(name='synthetic',patients=10,good=5,poor=5,with_gt=9,
            fold_sizes=(2,)*5,poor_per_fold=(1,)*5,fingerprint=synthetic_fingerprint(folds))
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            project=Path(temp)/'project';tables=project/'outputs/tables';tables.mkdir(parents=True)
            dataset=Path(temp)/'dataset';(dataset/'phenotypes').mkdir(parents=True)
            for path,fields,rows in ((tables/'patient_manifest.csv',MANIFEST_FIELDS,manifest),
                    (tables/'outcome_cv_folds_private.csv',FOLD_FIELDS,folds)):
                with path.open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            with (dataset/'phenotypes/filtered_clinical_updated.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(('patient_id',*h.FEATURE_NAMES))
                writer.writerows((row['patient_id'],*(['0']*11)) for row in manifest[:10])
            def load_plan():
                contract=load_and_validate(tables/'patient_manifest.csv',tables/'outcome_cv_folds_private.csv',profile)
                return contract,plan_outer_fold(contract,0,seed=630,validation_fraction=.2)
            stack.enter_context(patch.object(r.runtime,'PROJECT',project))
            stack.enter_context(patch.object(r.runtime,'DATASET',dataset))
            stack.enter_context(patch.object(r,'identity',return_value={}))
            stack.enter_context(patch.object(r.runtime,'environment',return_value={}))
            stack.enter_context(patch.object(r,'bridge',return_value={'dependencies':{}}))
            stack.enter_context(patch.object(r.os,'access',return_value=True))
            stack.enter_context(patch('ele630.ncct_private_output._validate_base'))
            stack.enter_context(patch('ele630.fusion_runner.load_plan',side_effect=load_plan))
            stack.enter_context(patch('ele630.fusion_runner.check_clinical_path'))
            for mode in ('ncct','ctp'):
                stack.enter_context(patch('ele630.fusion_checkpoints.load_'+mode,return_value=(None,{})))
            ncct=stack.enter_context(patch('ele630.ncct.NCCTRepository')).return_value
            ctp=stack.enter_context(patch('ele630.ctp.RegisteredCTPRepository')).return_value
            ncct.read_header.return_value=SimpleNamespace(shape=(4,16,16))
            ctp.read_header.return_value=SimpleNamespace(frame_count=40)
            ncct.load.side_effect=AssertionError('no image voxels in context')
            ctp.open_series.side_effect=AssertionError('no image voxels in context')
            return r.context('fictional')

    @unittest.skipUnless(importlib.util.find_spec('nibabel'),'requires imaging header dependency')
    def test_context_allows_missing_imaging_in_excluded_inventory_patient(self):
        for missing in (('has_ncct_skull_stripped',),('has_ctp_registered',),
                        ('has_ncct_skull_stripped','has_ctp_registered')):
            with self.subTest(missing=missing):
                contract,plan,_,_=self._inventory_context(excluded_missing=missing)
                self.assertEqual(len(contract.patients),10)
                self.assertEqual(tuple(map(len,(plan.inner_train,plan.inner_validation,plan.outer_test))),(6,2,2))

    @unittest.skipUnless(importlib.util.find_spec('nibabel'),'requires imaging header dependency')
    def test_context_still_rejects_missing_imaging_in_eligible_patient(self):
        from ele630.data_contract import ContractError
        for field in ('has_ncct_skull_stripped','has_ctp_registered'):
            for recorded in (0,1):
                with self.subTest(field=field,recorded=recorded), self.assertRaises(ContractError):
                    self._inventory_context(eligible_missing=field,eligible_record=recorded)

    @unittest.skipUnless(importlib.util.find_spec('nibabel'),'requires imaging header dependency')
    def test_context_preserves_150_inventory_requirement(self):
        for count in (149,151):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError,'expected 150 imaging inventory records'):
                self._inventory_context(count=count)

    def test_complete_state(self):
        self.assertEqual(h.validate_state(state()),state())

    def test_corrupt_or_incomplete_states_rejected(self):
        for key,value in [('scale',[0.]*11),('clinical_coef',[np.nan]*11),('fill',[0.]*10)]:
            s=state();s['clinical'][key]=value
            with self.assertRaises(ValueError):h.validate_state(s)
        s=state();s.pop('ctp')
        with self.assertRaises(ValueError):h.validate_state(s)

    def test_feature_order_and_pca_rejected(self):
        s=state();s['features'].reverse()
        with self.assertRaises(ValueError):h.validate_state(s)
        s=state();s['ctp']['components'][0]=[0.]*128
        with self.assertRaises(ValueError):h.validate_state(s)

    def test_transform_cannot_fit_or_mutate_state(self):
        s=state();before=h.digest(s)
        rows=list(fictional_rows(fictional_plan()).values())
        with patch.object(h.ClinicalPreprocessor,'fit_training',side_effect=AssertionError('fit forbidden')):
            c=h.clinical_transform(s,rows)
            self.assertEqual(c.shape,(24,11))
            self.assertTrue(np.isfinite(h.predict(s,c)).all())
        self.assertEqual(h.digest(s),before)

    def test_prediction_manual_oracle(self):
        s=state();c=np.ones((2,11),dtype=np.float32);e=np.ones((2,128))
        np.testing.assert_allclose(h.predict(s,c),h.sigmoid([.21,.21]),atol=1e-12)
        np.testing.assert_allclose(h.predict(s,c,e,'ncct'),h.sigmoid([.39,.39]),atol=1e-12)

    def test_sigmoid_stable(self):
        np.testing.assert_array_equal(h.sigmoid([-1000.,0.,1000.]),[0.,.5,1.])

    def test_interfaces_reject_malformed(self):
        with self.assertRaises(ValueError):h.predict(state(),np.zeros((2,10)))
        with self.assertRaises(ValueError):h.predict(state(),np.zeros((2,11)),np.zeros((1,128)),'ncct')
        with self.assertRaises(ValueError):h.predict(state(),np.zeros((2,11)),None,'bad')

    def test_metrics_same_as_development_and_threshold_ties(self):
        y=np.array([0,0,1,1]);p=[.1,.5,.5,.9]
        record=h.metrics(y,p)
        ref=h.aggregate_probability_metrics(y,p).as_dict()
        for key in ('bce','auroc','average_precision','brier_score','balanced_accuracy_0_5'):
            self.assertEqual(record[key],ref[key])
        self.assertEqual(record['confusion_matrix'],[[1,1],[0,2]])
        self.assertEqual(record['count'],4)

    def test_logits_loss_preserved(self):
        m=h.metrics(np.array([0,1]),[.25,.75],[.3,.4])
        self.assertAlmostEqual(m['bce'],.35)
        self.assertAlmostEqual(m['brier_score'],.0625)

    def test_metric_invalid_values(self):
        for p in ([np.nan,.3],[-.1,.3],[.2],[[.2,.3]]):
            with self.assertRaises(Exception):h.metrics(np.array([0,1]),p)

    def test_fixed_equivalence_tolerance(self):
        h.compare_predictions([.3],[.3+1e-11])
        for p in ([.3+1e-8],[np.nan],[]):
            with self.assertRaises(ValueError):h.compare_predictions([.3],p)

    def test_digest_seals_numerical_content(self):
        a=state();b=copy.deepcopy(a)
        self.assertEqual(h.digest(a),h.digest(b))
        b['clinical']['clinical_intercept'][0]+=.01
        self.assertNotEqual(h.digest(a),h.digest(b))

    def test_frozen_bundle_mismatch_fails(self):
        frozen={'binding':{'x':1},'state':state(),'status':'FROZEN',
                'development':{k:{} for k in h.MODELS}}
        seal={'frozen_sha256':h.digest(frozen)}
        with patch.object(r.runtime,'read_private_json',side_effect=[frozen,seal]):
            with self.assertRaises(ValueError):r.load_bundle('fictional',{'x':2})
        with patch.object(r.runtime,'read_private_json',side_effect=[frozen,{'frozen_sha256':'bad'}]):
            with self.assertRaises(ValueError):r.load_bundle('fictional',{'x':1})

    def test_slug_rejects_paths(self):
        for slug in ('../outside','/tmp/data','', 'a/b'):
            with self.assertRaises(ValueError):r.bundle_path(slug)

    def test_discovery_read_only_no_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'fictional.joblib').write_bytes(b'not pickle')
            with patch.object(r.runtime,'RUNTIME',root),patch.object(r,'identity'):
                result=r.discover('fictional')
            self.assertFalse(result['deserialized'])
            self.assertEqual(len(result['files']),1)
            self.assertEqual(len(list(root.iterdir())),1)

    def test_clinical_authorization_before_feature_parse(self):
        from ele630.clinical_preprocessing import FEATURE_NAMES
        plan=fictional_plan();patient=plan.inner_train[0]
        header=','.join(('patient_id',*FEATURE_NAMES))+'\n'
        content=header+patient.patient_id+','+','.join(['1']*11)+'\n'
        content+=plan.outer_test[0].patient_id+','+','.join(['SECRET_INVALID']*11)+'\n'
        with patch.object(Path,'open',return_value=io.StringIO(content)):
            rows=r.selected_clinical([patient])
        self.assertEqual(set(rows),{patient.patient_id})

    def test_clinical_duplicates_or_missing_coverage_fail(self):
        plan=fictional_plan();header=','.join(('patient_id',*h.FEATURE_NAMES))+'\n'
        for content in (header,header+plan.inner_train[0].patient_id+','+','.join(['0']*11)+'\n'+plan.inner_train[0].patient_id+','+','.join(['0']*11)+'\n'):
            with patch.object(Path,'open',return_value=io.StringIO(content)):
                with self.assertRaises(ValueError):r.selected_clinical(plan.inner_train)

    def test_final_wrong_digest_before_any_test_access(self):
        with patch.object(r,'context',return_value=(None,None,None,{})),patch.object(r,'load_bundle',return_value=({}, {'frozen_sha256':'right'})),patch.object(r,'selected_clinical',side_effect=AssertionError('test access')):
            with self.assertRaisesRegex(ValueError,'acknowledgement'):r.final('fictional','rev',reviewed='wrong')

    def test_completed_final_cannot_repeat(self):
        with patch.object(r,'context',return_value=(None,None,None,{})),patch.object(r,'load_bundle',return_value=({}, {'frozen_sha256':'right'})),patch.object(Path,'exists',return_value=True),patch.object(r,'selected_clinical',side_effect=AssertionError('test access')):
            with self.assertRaisesRegex(ValueError,'cannot be repeated'):r.final('fictional','rev',reviewed='right')

    def test_new_comparison_and_changed_technical_retry_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);path=base/'fictional'/'outer_fold_0';path.mkdir(parents=True)
            (base/'final_claim.json').write_text('{}')
            with patch.object(r,'BASE',base),patch.object(r,'context',return_value=(None,None,None,{})),\
                 patch.object(r,'load_bundle',return_value=({}, {'frozen_sha256':'right'})),\
                 patch.object(r.runtime,'read_private_json',return_value={'frozen_sha256':'different','run_id':'old'}),\
                 patch.object(r,'selected_clinical',side_effect=AssertionError('test access')):
                for retry in (False,True):
                    with self.assertRaisesRegex(ValueError,'unchanged technical retry'):
                        r.final('fictional','rev',reviewed='right',retry=retry)

    def test_no_preparation_after_final_claim(self):
        with patch.object(Path,'exists',return_value=True),patch.object(r,'context',side_effect=AssertionError('context access')):
            with self.assertRaisesRegex(ValueError,'no preparation after'):
                r.prepare('fictional','rev',unavailable=True)

    def test_final_has_no_fit_or_preparation_calls(self):
        import inspect
        tree=ast.parse(inspect.getsource(r.final))
        calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden={'fit','fit_transform','fit_training','backward','step','prepare','bridge'}
        self.assertFalse(any((isinstance(f,ast.Attribute) and f.attr in forbidden) or
                             (isinstance(f,ast.Name) and f.id in forbidden) for f in calls))

    def test_prepare_requires_explicit_artifact_policy(self):
        with self.assertRaises(ValueError):r.prepare('fictional','rev')
        with self.assertRaises(ValueError):r.prepare('fictional','rev',originals='a',unavailable=True)

    def test_report_completeness_and_bad_counts(self):
        record={'status':'FINAL_PASS','development':{},'test':{}}
        for cohort,n in [('development',23),('test',28)]:
            poor=7 if cohort=='development' else 9
            m=h.metrics(np.array([0]*(n-poor)+[1]*poor),[.3]*n)
            record[cohort]={k:copy.deepcopy(m) for k in h.MODELS}
        self.assertEqual(len(h.report_rows(record)),10)
        record['test']['clinical']['count']=27
        with self.assertRaises(ValueError):h.report_rows(record)

    def test_perfect_average_precision_roundoff_is_reportable(self):
        record={'status':'FINAL_PASS','development':{},'test':{}}
        for cohort,good,poor in [('development',16,7),('test',19,9)]:
            m=h.metrics(np.array([1]*poor+[0]*good),np.linspace(.99,.01,good+poor))
            record[cohort]={k:copy.deepcopy(m) for k in h.MODELS}
        self.assertAlmostEqual(record['test']['clinical']['average_precision'],1.)
        self.assertEqual(len(h.report_rows(record)),10)


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'requires clinical SciPy/sklearn environment')
class FitTests(unittest.TestCase):
    def test_private_worker_synthetic_process_complete(self):
        from ele630.heldout_prepare import process
        rng=np.random.default_rng(630)
        payload={'revision':'fictional','path':'reconstruct','synthetic':True,
                 'train':{m:rng.normal(size=(18,128)).tolist() for m in ('ncct','ctp')},
                 'validation':{m:rng.normal(size=(6,128)).tolist() for m in ('ncct','ctp')},
                 'imaging':{m:[.3]*6 for m in ('ncct','ctp')}}
        with patch.object(r.runtime,'source_identity'),\
             patch.object(r,'selected_clinical',side_effect=AssertionError('real data access')):
            result=process(payload)
        h.validate_state(result['state'])
        self.assertEqual(len(result['clinical_probabilities']),6)
        self.assertIn('no_image',result['results']['ctp']['metrics'])

    def test_original_import_is_fit_free_and_matches_saved_objects(self):
        import joblib
        from ele630.heldout_prepare import original_state, export_clinical, export_image
        from ele630.fusion import ImagePCA, _fit
        from ele630.clinical_preprocessing import ClinicalPreprocessor
        from ele630.data_contract import OutcomePatient
        from dataclasses import replace
        plan=fictional_plan()
        plan=replace(plan,inner_train=tuple(OutcomePatient('fictional-train-'+str(i),'fictional',
                     'poor' if i%3==0 else 'good',False,1) for i in range(92)))
        rows=fictional_rows(plan)
        pre=ClinicalPreprocessor().fit_training(plan,rows)
        c=pre.transform([rows[p.patient_id] for p in plan.inner_train])
        y=np.array([int(p.outcome=='poor') for p in plan.inner_train])
        clinical=_fit(c,y)
        objects={'clinical_preprocessor':pre,'clinical_model':clinical}
        expected={'features':list(h.FEATURE_NAMES),'clinical':export_clinical(pre,clinical)}
        rng=np.random.default_rng(630)
        for mode in ('ncct','ctp'):
            e=rng.normal(size=(92,128));t=ImagePCA().fit(e)
            model=_fit(np.column_stack((t.transform(e),c)),y)
            objects.update({mode+'_scaler':t.scaler,mode+'_pca':t.pca,mode+'_model':model})
            expected[mode]=export_image(t.scaler,t.pca,model)
        spec={'schema':1,'source_revision':'54f289161a35206c00a517d7f12e2fdf1e7a4127',
              'membership':r.membership(plan),'checkpoints':{},'files':{}}
        with tempfile.TemporaryDirectory() as temp:
            for name,obj in objects.items():
                path=Path(temp)/(name+'.joblib');joblib.dump(obj,path)
                spec['files'][name]={'path':str(path),'sha256':r.file_hash(path)}
            with patch.object(r.runtime,'private_file',side_effect=lambda p:p),patch.object(ClinicalPreprocessor,'fit_training',side_effect=AssertionError('fit forbidden')):
                self.assertEqual(original_state(spec,r.membership(plan),{}),expected)
                spec['files']['clinical_model']['sha256']='bad'
                with self.assertRaisesRegex(ValueError,'checksum'):original_state(spec,r.membership(plan),{})

    def test_reconstruction_matches_unchanged_stage7_and_train_only(self):
        from ele630.heldout_prepare import fit_state
        from ele630.xai_reconstruction import reconstruct
        plan=fictional_plan();rows=fictional_rows(plan)
        rng=np.random.default_rng(630)
        train={m:rng.normal(size=(18,128)) for m in ('ncct','ctp')}
        val={m:rng.normal(size=(6,128)) for m in ('ncct','ctp')}
        train_rows={p.patient_id:rows[p.patient_id] for p in plan.inner_train}
        s=fit_state(plan,train_rows,train)
        c=h.clinical_transform(s,[rows[p.patient_id] for p in plan.inner_validation])
        for mode in ('ncct','ctp'):
            ref=reconstruct(plan,rows,train[mode],val[mode],[.3]*6)
            h.compare_predictions(h.predict(s,c,val[mode],mode),ref['probabilities'])
        with patch.object(h.ClinicalPreprocessor,'fit_training',side_effect=AssertionError('fit forbidden')):
            h.predict(s,h.clinical_transform(s,[rows[p.patient_id] for p in plan.inner_validation]),val['ncct'],'ncct')


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires imaging Torch environment')
class TorchTests(unittest.TestCase):
    def test_final_end_to_end_fictional_28_and_repeat_protection(self):
        import torch
        from types import SimpleNamespace
        from dataclasses import replace
        from ele630.data_contract import OutcomePatient
        from ele630.resnet3d import CompactResNet10
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_synthetic import FictionalCTP
        torch.set_num_threads(1)
        plan=replace(fictional_plan(),outer_test=tuple(OutcomePatient('fictional-outer-'+str(i),
                     'fictional','poor' if i>=19 else 'good',False,0) for i in range(28)))
        model1,model2=CompactResNet10(width=16),CompactCTPMean(width=16)
        loaded={'ncct':(model1,{}),'ctp':(model2,{})}
        binding={'checkpoints':{'ncct':{'checkpoint_sha256':'fictional'},'ctp':{'checkpoint_sha256':'fictional'}}}
        y=np.array([0]*16+[1]*7)
        frozen={'status':'FROZEN','binding':binding,'state':state(),
                'development':{k:h.metrics(y,[.3]*23) for k in h.MODELS}}
        first_row=next(iter(fictional_rows(fictional_plan()).values()))
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);path=base/'fictional'/'outer_fold_0';path.mkdir(parents=True)
            (path/'frozen.json').write_text(json.dumps(frozen))
            (path/'seal.json').write_text(json.dumps({'frozen_sha256':h.digest(frozen)}))
            with patch.object(r,'BASE',base),patch.object(r,'_DEVICE','cpu'),\
                 patch.object(r,'acquire_lock',side_effect=lambda p:os.open(p,os.O_CREAT|os.O_RDWR,0o600)),\
                 patch.object(r,'context',return_value=(None,plan,loaded,binding)),\
                 patch.object(r.runtime,'read_private_json',side_effect=lambda p:json.loads(Path(p).read_text())),\
                 patch.object(r,'selected_clinical',return_value={p.patient_id:first_row for p in plan.outer_test}),\
                 patch('ele630.ncct.NCCTRepository') as ncct,patch('ele630.ctp.RegisteredCTPRepository') as ctp,\
                 patch.object(h.ClinicalPreprocessor,'fit_training',side_effect=AssertionError('fit forbidden')):
                ncct.return_value.load.return_value=SimpleNamespace(data=np.ones((16,16,4),dtype=np.float32),affine=np.eye(4))
                ctp.return_value.open_series.side_effect=lambda p:FictionalCTP(630)
                result=r.final('fictional','fictional',reviewed=h.digest(frozen))
                self.assertEqual(result['test_count'],28)
                record=json.loads((path/'aggregate_final.json').read_text())
                self.assertEqual(len(h.report_rows(record)),10)
                self.assertEqual(record['test']['ncct']['confusion_matrix'][1][0]+record['test']['ncct']['confusion_matrix'][1][1],9)
                with self.assertRaisesRegex(ValueError,'cannot be repeated'):
                    r.final('fictional','fictional',reviewed=h.digest(frozen),retry=True)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in model1.parameters()))

    def test_frozen_ncct_and_ctp_interface_no_weight_changes(self):
        import torch
        from ele630.resnet3d import CompactResNet10
        from ele630.ctp_model import CompactCTPMean
        from ele630.fusion_features import freeze,extract
        from ele630.fusion_synthetic import FictionalCTP
        torch.set_num_threads(1)
        for mode,model in [('ncct',CompactResNet10(width=16)),('ctp',CompactCTPMean(width=16))]:
            freeze(model)
            before=r.weight_digest(model)
            value=np.ones((1,4,16,16),dtype=np.float32) if mode=='ncct' else FictionalCTP(630)
            embedding,p=extract(model,value,mode)
            self.assertEqual(embedding.shape,(128,))
            self.assertTrue(0<=p<=1)
            self.assertEqual(before,r.weight_digest(model))
            self.assertTrue(all(x.grad is None and not x.requires_grad for x in model.parameters()))


if __name__=='__main__':unittest.main()
