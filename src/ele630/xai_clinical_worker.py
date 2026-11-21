"""PRIVATE IPC entry point; stdout includes fitted state, never invoke in a terminal.

Only xai_runtime.bridge consumes this pipe. Dependencies and real membership
are checked independently. Errors are fixed strings and contain no pipe data.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict


def process(payload):
    from .data_contract import OutcomePatient
    from .ncct_splits import FoldPlan
    from .fusion_synthetic import fictional_rows
    from .xai_reconstruction import reconstruct
    if set(payload) != {"mode", "plan", "train", "validation", "imaging_probabilities", "source_revision"}:
        raise ValueError("private IPC request rejected")
    p = payload["plan"]
    plan = FoldPlan(p["outer_fold"], p["seed"], p["validation_fraction"],
                    *(tuple(OutcomePatient(**row) for row in p[name])
                      for name in ("inner_train", "inner_validation", "outer_test")),
                    tuple(p["cohort_stratified_outcomes"]))
    if payload["mode"] == "synthetic":
        if any(not p.patient_id.startswith("fictional-") for p in plan.inner_train+plan.inner_validation+plan.outer_test):
            raise ValueError("fictional-only request required")
        rows = fictional_rows(plan)
    elif payload["mode"] == "real":
        from . import fusion_runtime as original
        from .xai_runtime import source_identity
        from .fusion_runner import load_plan, check_clinical_path
        from .clinical_runner import read_authorized_clinical
        source_identity(payload["source_revision"])
        if asdict(load_plan()[1]) != asdict(plan):
            raise ValueError("independent original membership check failed")
        check_clinical_path()
        with (original.DATASET/"phenotypes/filtered_clinical_updated.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = read_authorized_clinical(stream, plan)
    else:
        raise ValueError("invalid private IPC mode")
    return reconstruct(plan, rows, payload["train"], payload["validation"], payload["imaging_probabilities"])


def main():
    try:
        if os.environ.get("ELE630_XAI_PRIVATE_IPC") != "1" or sys.stdin.isatty() or sys.stdout.isatty():
            raise ValueError("private parent-owned pipes required")
        from .fusion_clinical_worker import dependencies
        versions = dependencies()
        payload = json.load(sys.stdin)
        if payload == {"probe": True}:
            result = {"dependencies": versions}
        elif payload == {"synthetic_tests": True}:
            import io
            import unittest
            from .fusion_runtime import PROJECT
            sys.path.insert(0, str(PROJECT/"tests"))
            from test_xai_foundation import ReconstructionTests
            # The GPU test module performs Torch equivalence; this interpreter
            # intentionally has no Torch and must not import the GPU stack.
            suite = unittest.TestSuite([ReconstructionTests("test_reconstruction_matches_original_aggregates_and_train_only_fit")])
            tested = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
            if not tested.wasSuccessful() or tested.skipped or tested.testsRun != 1:
                raise ValueError("clinical reconstruction synthetic test failed")
            result = {"tests": tested.testsRun, "skipped": 0, "dependencies": versions}
        else:
            result = process(payload)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception:
        print("STAGE7_CLINICAL_WORKER_FAIL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
