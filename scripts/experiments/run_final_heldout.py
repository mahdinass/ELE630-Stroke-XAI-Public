#!/usr/bin/env python3
"""Explicitly separated read-only discovery, preparation, preflight and final modes."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['discover','prepare','preflight','final','results','seal','smoke'])
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--run-id',default='final_train92_v1')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--original-spec')
    group.add_argument('--originals-unavailable',action='store_true')
    parser.add_argument('--development-predictions')
    parser.add_argument('--execute-final',action='store_true')
    parser.add_argument('--protocol-reviewed')
    parser.add_argument('--technical-retry',action='store_true')
    args=parser.parse_args(argv)
    try:
        from ele630 import heldout_runner as r
        if args.mode=='discover':
            output=r.discover(args.source_revision)
        elif args.mode=='prepare':
            r.verify_smoke(args.source_revision)
            output=r.prepare(args.run_id,args.source_revision, originals=args.original_spec,
                             unavailable=args.originals_unavailable,
                             prediction_reference=args.development_predictions)
        elif args.mode=='preflight':
            r.verify_smoke(args.source_revision)
            output=r.preflight(args.run_id,args.source_revision)
        elif args.mode=='final':
            if not args.execute_final:
                raise ValueError('explicit --execute-final is required')
            r.verify_smoke(args.source_revision)
            output=r.final(args.run_id,args.source_revision,reviewed=args.protocol_reviewed,
                           retry=args.technical_retry)
        elif args.mode=='smoke':
            output=r.smoke(args.source_revision)
        elif args.mode=='seal':
            binding=r.context(args.source_revision)[3]
            _,seal=r.load_bundle(args.run_id,binding)
            print(seal['frozen_sha256'])
            return 0
        else:
            binding=r.context(args.source_revision)[3]
            frozen,seal=r.load_bundle(args.run_id,binding)
            record=r.runtime.read_private_json(r.bundle_path(args.run_id)/'aggregate_final.json')
            if (record['frozen_sha256']!=seal['frozen_sha256'] or record['source_revision']!=args.source_revision
                    or record['development']!=frozen['development']):
                raise ValueError('aggregate frozen provenance mismatch')
            # Fixed numeric/enum projection only. No arbitrary private JSON is printed.
            from ele630.heldout import report_rows
            print('model\tcohort\tn\tgood\tpoor\tBCE\tAUROC\tAP\tBrier\tBA_at_0.5\tTN\tFP\tFN\tTP')
            for name,cohort,m in report_rows(record):
                cm=m['confusion_matrix']
                print('\t'.join(map(str,[name,cohort,m['count'],m['good'],m['poor'],
                    m['bce'],m['auroc'],m['average_precision'],m['brier_score'],m['balanced_accuracy_0_5'],
                    *cm[0],*cm[1]])))
            return 0
        print(json.dumps(output,allow_nan=False))
        return 0
    except Exception as error:
        # Fixed messages from this workflow are safe. Third-party exception text
        # and tracebacks may contain patient identifiers, so never print them.
        import traceback
        frames=traceback.extract_tb(error.__traceback__)
        location=next((f for f in reversed(frames) if Path(f.filename).name.startswith('heldout')),None)
        where=(Path(location.filename).name+':'+str(location.lineno)) if location else 'environment'
        detail=''
        if (frames and Path(frames[-1].filename).name in
                {'heldout.py','heldout_runner.py','heldout_prepare.py','run_final_heldout.py',
                 'fusion_runtime.py','fusion_checkpoints.py','ncct_private_output.py'}
                and isinstance(error,ValueError)):
            detail='; '+str(error)
        print('HELDOUT_FAIL type='+type(error).__name__+' at='+where+detail+
              '; no fallback performed',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
