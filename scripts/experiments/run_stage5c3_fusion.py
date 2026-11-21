#!/usr/bin/env python3
"""Three fixed Stage 5C3 modes, read-only preflight and safe aggregate reader."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "ncct", "ctp", "preflight", "verify-smoke", "results"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execute-real", action="store_true")
    args = parser.parse_args(argv)
    try:
        from ele630 import fusion_runtime as runtime
        from ele630 import fusion_runner as runner
        if args.mode in {"verify-smoke", "results"}:
            identity = runtime.source_identity(args.source_revision)
            if args.mode == "verify-smoke":
                record = runtime.read_private_json(runtime.RUNTIME / "fusion_runs" / runtime.RUNS["smoke"] / "outer_fold_0/aggregate_training.json")
                runner.smoke_record(identity, record["dependencies"])
                output = {"status": "SMOKE_PASS", "source_revision": args.source_revision,
                          "synthetic_modalities": 2, "patient_data_access": False}
            else:
                records = []
                for mode in ("ncct", "ctp"):
                    record = runtime.read_private_json(runtime.RUNTIME / "fusion_runs" / runtime.RUNS[mode] / "outer_fold_0/aggregate_training.json")
                    if (record.get("status") != "PASS" or record.get("mode") != mode
                            or record.get("identity") != identity or record.get("outer_test_evaluated") is not False):
                        raise ValueError("completed matching experiment required")
                    runtime.validate_result(record["results"])
                    records.append(record)
                clinical = [r["results"]["metrics"]["clinical_only"] for r in records]
                if (records[0]["provenance"]["membership"] != records[1]["provenance"]["membership"]
                        or any(abs(clinical[0][key] - clinical[1][key]) > 1e-10 for key in clinical[0])):
                    raise ValueError("experiments do not have matched patients and clinical reference")
                output = {mode: record["results"] for mode, record in zip(("ncct", "ctp"), records)}
            print(json.dumps(output, indent=2, allow_nan=False))
        elif args.mode == "preflight":
            print(json.dumps(runner.preflight(args.source_revision), sort_keys=True))
        else:
            print(json.dumps(runner.execute(args.mode, args.source_revision, execute_real=args.execute_real)))
        return 0
    except Exception as error:
        from ele630.fusion_runner import FusionRunError
        print(str(error) if type(error) is FusionRunError else
              "STAGE5C3_FAIL: guard, dependency, provenance or workflow check failed; retain private diagnostics", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
