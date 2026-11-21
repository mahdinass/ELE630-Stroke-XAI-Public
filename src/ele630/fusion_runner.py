"""Guarded Stage 5C3 orchestration; no I/O at import and no outer evaluator."""
from __future__ import annotations
import math
import os
import time
from dataclasses import asdict

from . import fusion_runtime as runtime


class FusionRunError(RuntimeError):
    """Only fixed, non-identifying workflow diagnostics may use this type."""


def _validate_no_image_control(result):
    """Fixed absolute agreement gate; no validation-based tolerance selection."""
    clinical, control = (result["metrics"][name] for name in ("clinical_only", "no_image"))
    for key in ("bce", "brier_score", "mean_predicted_probability"):
        if (not math.isfinite(clinical[key]) or not math.isfinite(control[key])
                or not math.isclose(control[key], clinical[key], rel_tol=0.0, abs_tol=1e-8)):
            raise ValueError("zero-image control does not reproduce clinical-only reference")


def load_plan():
    from .data_contract import load_and_validate
    from .ncct_splits import plan_outer_fold
    tables = runtime.PROJECT / "outputs/tables"
    for path in (tables / "patient_manifest.csv", tables / "outcome_cv_folds_private.csv"):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(runtime.PROJECT.resolve()):
            raise ValueError("original private contract tables required")
    contract = load_and_validate(tables / "patient_manifest.csv", tables / "outcome_cv_folds_private.csv")
    plan = plan_outer_fold(contract, 0, seed=630, validation_fraction=.20)
    if (contract.fingerprint != "3db8c4fd5fef0e4e"
            or tuple(map(len, (plan.inner_train, plan.inner_validation, plan.outer_test))) != (92, 23, 28)):
        raise ValueError("original 143-patient fold0 membership required")
    return contract, plan


def check_clinical_path():
    path = runtime.DATASET / "phenotypes/filtered_clinical_updated.csv"
    if (path.is_symlink() or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(runtime.DATASET.resolve(strict=True))):
        raise ValueError("documented clinical source missing or unsafe")


def smoke_record(identity, dependencies):
    record = runtime.read_private_json(runtime.RUNTIME / "fusion_runs" / runtime.RUNS["smoke"] / "outer_fold_0/aggregate_training.json")
    if (record.get("status") != "PASS" or record.get("mode") != "smoke"
            or record.get("identity") != identity or record.get("dependencies") != dependencies
            or set(record.get("results", {})) != {"ncct", "ctp"}
            or record.get("patient_data_access") is not False):
        raise ValueError("passing smoke from this exact source and stack is required")
    for result in record["results"].values():
        runtime.validate_result(result)
    tests = record.get("synthetic_tests", {})
    if (tests.get("skipped") != 0 or tests.get("clinical", 0) < 28
            or tests.get("features", 0) < 33):
        raise ValueError("complete clinical and imaging smoke tests required")
    return record


def payload(plan, train, validation, probabilities, *, synthetic):
    return {"mode": "synthetic" if synthetic else "real", "plan": asdict(plan),
            "train": train.tolist(), "validation": validation.tolist(),
            "imaging_probabilities": probabilities.tolist()}


def real_features(contract, plan, modality, model):
    from .fusion_features import DevelopmentFeatures
    if modality == "ncct":
        from .ncct import NCCTRepository
        from .ncct_samples import LazyNCCTSamples
        samples = LazyNCCTSamples(contract, NCCTRepository(contract, runtime.DATASET), lower=-20, upper=120)
        loader = lambda p: samples.load(p).image
    else:
        from .ctp import RegisteredCTPRepository
        # Existing proxy-based one-frame loader; no full 4D tensors or cache.
        repository = RegisteredCTPRepository(contract, runtime.DATASET)
        loader = repository.open_series
    source = DevelopmentFeatures(plan, loader, model.to("cuda"), modality)
    train, _ = source.partition(plan.inner_train)
    validation, probabilities = source.partition(plan.inner_validation)
    return train, validation, probabilities


def preflight(expected_revision):
    """Read-only metadata/checkpoint checks, for the authorized archive; no image/clinical cells."""
    phase = "source"
    try:
        identity = runtime.source_identity(expected_revision)
        phase = "GPU_imports_on_login"
        dependencies = runtime.environment(gpu=False)
        phase = "clinical_dependencies"
        runtime.bridge({"probe": True})
        phase = "private_output"
        from .ncct_private_output import _validate_base
        _validate_base(runtime.RUNTIME / "fusion_runs")
        phase = "immutable_contract"
        contract, plan = load_plan()
        phase = "clinical_source_path"
        check_clinical_path()
        from .fusion_checkpoints import load_ncct, load_ctp
        phase = "NCCT_job486314_checkpoint"
        load_ncct(plan)
        phase = "CTP_Stage4R_checkpoint"
        load_ctp(contract)
        return {"status": "PREFLIGHT_PASS", "checkpoint_sources": 2, "training_count": 92,
                "validation_count": 23, "image_or_clinical_values_accessed": False,
                "identity": identity, "dependencies": dependencies}
    except Exception:
        raise FusionRunError(f"STAGE5C3_PREFLIGHT_FAIL phase={phase}") from None


def execute(mode, expected_revision, *, execute_real=False):
    if mode not in runtime.RUNS or (mode != "smoke" and execute_real is not True):
        raise ValueError("real development requires explicit execution flag")
    os.umask(0o077)
    phase, destination = "source", None
    started = time.monotonic()
    try:
        identity = runtime.source_identity(expected_revision)
        phase = "environment"
        dependencies = runtime.environment()
        clinical_dependencies = runtime.bridge({"probe": True})["dependencies"]
        dependencies["clinical"] = clinical_dependencies
        import torch
        torch.set_num_threads(1)
        from .ncct_training import _seed_everything, _membership_record
        from .ncct_private_output import create_private_run
        _seed_everything(630)
        phase = "private_output"
        base = runtime.RUNTIME / "fusion_runs"
        if mode != "smoke":
            smoke_record(identity, dependencies)
        destination = create_private_run(base, runtime.RUNS[mode], 0)
        # Fictional checkpoint fixtures and library scratch files also remain
        # inside this run's approved private storage, never shared /tmp.
        os.environ["TMPDIR"] = str(destination.fold_dir)
        import tempfile
        tempfile.tempdir = None
        result, provenance = {}, {}
        if mode == "smoke":
            phase = "synthetic_tests"
            clinical_tests = runtime.bridge({"synthetic_tests": True})
            import io
            import sys
            import unittest
            sys.path.insert(0, str(runtime.PROJECT / "tests"))
            import test_fusion
            import test_ncct_stride
            import test_ctp_evaluation
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                                       for cls in (test_fusion.FusionFeatureTests,
                                                   test_ncct_stride.CheckpointTests,
                                                   test_ctp_evaluation.CTPAggregateEvaluatorTests))
            tested = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
            if not tested.wasSuccessful() or tested.skipped:
                raise ValueError("GPU feature tests failed or skipped")
            from .fusion_synthetic import synthetic_embeddings
            for modality in ("ncct", "ctp"):
                phase = f"synthetic_{modality}"
                plan, train, validation, probabilities = synthetic_embeddings(modality, "cuda")
                result[modality] = runtime.validate_result(runtime.bridge(payload(plan, train, validation, probabilities, synthetic=True)))
        else:
            phase = "contract"
            contract, plan = load_plan()
            check_clinical_path()
            phase = "checkpoint"
            from .fusion_checkpoints import load_ncct, load_ctp
            model, provenance = load_ncct(plan) if mode == "ncct" else load_ctp(contract)
            phase = "frozen_features"
            from .ncct_training import _parameter_digest
            before = _parameter_digest(model)
            train, validation, probabilities = real_features(contract, plan, mode, model)
            if before != _parameter_digest(model) or any(p.grad is not None for p in model.parameters()):
                raise ValueError("frozen weights changed during extraction")
            phase = "fusion"
            request = payload(plan, train, validation, probabilities, synthetic=False)
            request["source_revision"] = expected_revision
            result = runtime.validate_result(runtime.bridge(request))
            phase = "matched_references"
            if abs(result["metrics"]["imaging_only"]["bce"] - {"ncct": .614388, "ctp": .6143472169}[mode]) > 5e-7:
                raise ValueError("imaging baseline does not reproduce frozen development result")
            if abs(result["metrics"]["clinical_only"]["bce"] - .2313358984) > 1e-7:
                raise ValueError("clinical comparator does not reproduce Stage 5B")
            _validate_no_image_control(result)
            provenance["membership"] = _membership_record(plan)
        phase = "publication"
        if runtime.source_identity(expected_revision) != identity:
            raise ValueError("source identity changed during execution")
        if list(destination.fold_dir.iterdir()):
            raise ValueError("unexpected artifact survived before aggregate publication")
        record = {"stage": "5C3", "protocol_version": "proposal-v1", "status": "PASS",
                  "mode": mode, "identity": identity, "dependencies": dependencies,
                  "patient_data_access": mode != "smoke", "outer_test_evaluated": False,
                  "individual_artifacts_written": False, "provenance": provenance,
                  "elapsed_seconds": time.monotonic() - started,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "results": result}
        if mode == "smoke":
            record["synthetic_tests"] = {"clinical": clinical_tests["tests"],
                                         "features": tested.testsRun, "skipped": 0}
        runtime.publish(destination.metrics_path, record)
        runtime.private_file(destination.metrics_path)
        return {"status": "PASS", "mode": mode, "aggregate_artifact_written": True,
                "outer_test_evaluated": False}
    except Exception as error:
        # Fixed phase and exception class are useful without patient content,
        # paths, raw exception messages, traceback locals or pipe contents.
        if destination is not None:
            try:
                runtime.publish(destination.fold_dir / "failure.json", {
                    "status": "FAIL", "phase": phase, "exception_type": type(error).__name__,
                    "mode": mode, "elapsed_seconds": time.monotonic() - started})
            except Exception:
                pass
        raise FusionRunError(f"STAGE5C3_FAIL phase={phase}; retain private diagnostics") from None
