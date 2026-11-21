"""Private stdin/stdout IPC worker in the existing frozen clinical environment.

No individual inputs, predictions or fitted parameters are written to files.
This internal entry point emits either strict aggregate JSON or a fixed error.
"""
from __future__ import annotations
import json
import sys
from dataclasses import asdict
from .fusion_runtime import PYTHON, DATASET, validate_result


def dependencies():
    import importlib
    import platform
    from pathlib import Path
    from .clinical_runner import _APPROVED_VERSIONS
    if platform.python_version() != "3.10.11" or Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("clinical interpreter mismatch")
    versions = {name: importlib.import_module(name).__version__ for name in _APPROVED_VERSIONS}
    if versions != _APPROVED_VERSIONS:
        raise ValueError("clinical dependency mismatch")
    return versions


def process(payload):
    from .fusion import run_fusion
    from .data_contract import OutcomePatient
    from .ncct_splits import FoldPlan
    from .clinical_runner import read_authorized_clinical
    plan = payload["plan"]
    plan = FoldPlan(plan["outer_fold"], plan["seed"], plan["validation_fraction"],
                    *(tuple(OutcomePatient(**p) for p in plan[name])
                      for name in ("inner_train", "inner_validation", "outer_test")),
                    tuple(plan["cohort_stratified_outcomes"]))
    if payload["mode"] == "synthetic":
        from .fusion_synthetic import fictional_rows
        if any(not p.patient_id.startswith("fictional-") for p in plan.inner_train + plan.inner_validation + plan.outer_test):
            raise ValueError("synthetic identifiers required")
        rows = fictional_rows(plan)
    elif payload["mode"] == "real":
        # Independently reconstruct canonical membership in this interpreter.
        from .fusion_runtime import source_identity
        source_identity(payload["source_revision"])
        from .fusion_runner import load_plan
        if asdict(load_plan()[1]) != asdict(plan):
            raise ValueError("IPC membership disagrees with original cohort")
        path = DATASET / "phenotypes/filtered_clinical_updated.csv"
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(DATASET.resolve(strict=True)):
            raise ValueError("clinical path leaves approved dataset")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = read_authorized_clinical(stream, plan)
    else:
        raise ValueError("unknown IPC mode")
    return validate_result(run_fusion(plan, rows, payload["train"], payload["validation"], payload["imaging_probabilities"]))


def main():
    try:
        versions = dependencies()  # before reading the pipe or private clinical source
        payload = json.load(sys.stdin)
        if payload == {"probe": True}:
            result = {"dependencies": versions}
        elif payload == {"synthetic_tests": True}:
            import io
            import unittest
            from .fusion_runtime import PROJECT
            sys.path.insert(0, str(PROJECT / "tests"))
            import test_fusion
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                                       for cls in (test_fusion.FusionSourceTests, test_fusion.FusionEstimatorTests))
            tested = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
            if not tested.wasSuccessful() or tested.skipped:
                raise ValueError("clinical synthetic tests failed or skipped")
            result = {"status": "PASS", "tests": tested.testsRun, "skipped": 0}
        else:
            result = process(payload)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception:
        print("Stage 5C3 clinical worker failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
