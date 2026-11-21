#!/usr/bin/env python3
"""Separate guarded Stage7 modes. No outer-test evaluator or implicit job runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "mask-preflight", "smoke", "resource-check", "ncct", "ctp", "verify-smoke", "results", "report"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--protocol-reviewed", help="SHA256 of the reviewed Stage7 protocol")
    parser.add_argument("--smoke-run")
    parser.add_argument("--resource-run")
    parser.add_argument("--resource-shape", type=int, nargs=3, default=(72, 512, 512), metavar=("D", "H", "W"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modality", choices=("ncct", "ctp"))
    parser.add_argument("--report-id")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        from ele630 import xai_runner as runner, xai_runtime as runtime
        if args.mode not in ("preflight", "mask-preflight") and not args.run_id:
            raise ValueError("run identity required")
        if args.mode in ("preflight", "mask-preflight", "verify-smoke", "results", "report") and (
                args.execute_real or args.protocol_reviewed or args.smoke_run or args.resume or args.resource_run):
            raise ValueError("read-only modes reject execution flags")
        if args.mode == "preflight":
            output = runner.preflight(args.source_revision)
        elif args.mode == "mask-preflight":
            output = runner.mask_preflight(args.source_revision)
        elif args.mode == "resource-check":
            if args.execute_real or args.protocol_reviewed or args.smoke_run or args.resume or args.resource_run:
                raise ValueError("patient-free resource check rejects real-execution flags")
            output = runner.execute_resource(args.source_revision, args.run_id, args.resource_shape)
        elif args.mode == "verify-smoke":
            runner.verify_smoke(args.run_id, args.source_revision)
            output = {"status": "UIS_SYNTHETIC_VERIFIED", "patient_data_access": False}
        elif args.mode in ("results", "report"):
            if not args.modality:
                raise ValueError("explicit result modality required")
            output = runtime.read_results(args.run_id, args.modality, args.source_revision)
            if args.mode == "report":
                from ele630.xai_reporting import private_report
                private_report(args.run_id, args.modality, args.report_id, output, figures=args.figures)
                output = {"status": "AGGREGATE_REPORT_WRITTEN", "patient_artifacts_exported": False}
        else:
            output = runner.execute(args.mode, args.source_revision, args.run_id,
                                    execute_real=args.execute_real, protocol_approval=args.protocol_reviewed,
                                    smoke_run=args.smoke_run, resume=args.resume, resource_run=args.resource_run)
        print(json.dumps(output, indent=2, allow_nan=False))
        return 1 if args.mode in ("preflight", "mask-preflight") and not output["ready_for_real_execution"] else 0
    except Exception as error:
        from ele630.xai_runner import XAIFailure
        print(str(error) if type(error) is XAIFailure else
              "STAGE7_FAIL: guard, dependency, provenance or workflow check failed; inspect private diagnostics", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
