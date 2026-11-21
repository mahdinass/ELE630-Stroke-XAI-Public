"""Guarded Stage7 orchestration; no execution or I/O at import."""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from contextlib import nullcontext

from . import fusion_runtime as original
from . import xai_runtime as runtime


class XAIFailure(RuntimeError):
    """Only fixed workflow phases may reach terminal diagnostics."""


def mask_preflight(revision):
    """Dedicated metadata-only mode: no checkpoint, image voxel or clinical fit."""
    phase = "source"
    try:
        identity = runtime.source_identity(revision)
        phase = "header_dependencies"
        original.environment(gpu=False)
        from .fusion_runner import load_plan
        from .xai_masks import real_mask_preflight
        phase = "anatomical_mask_headers"
        masks = real_mask_preflight(*load_plan())
        return {"status": "MASK_HEADER_PREFLIGHT_PASS" if masks["path_contract_verified"] else "MASK_HEADER_PREFLIGHT_BLOCKED",
                "identity": identity, "mask_headers": masks, "ready_for_real_execution": masks["path_contract_verified"],
                "image_or_clinical_values_accessed": False, "checkpoints_loaded": False}
    except Exception:
        raise XAIFailure(f"STAGE7_MASK_PREFLIGHT_FAIL phase={phase}") from None


def preflight(revision):
    phase = "source"
    try:
        identity = runtime.source_identity(revision)
        phase = "dependencies"
        dependencies = original.environment(gpu=False)
        dependencies["clinical"] = runtime.bridge({"probe": True})["dependencies"]
        phase = "private_output"
        from .ncct_private_output import _validate_base
        _validate_base(original.RUNTIME/"stage7_runs")
        phase = "original_contract_checkpoints"
        from .fusion_runner import preflight as original_preflight
        original_preflight(revision)
        phase = "anatomical_mask_headers"
        from .fusion_runner import load_plan
        from .xai_masks import real_mask_preflight
        masks = real_mask_preflight(*load_plan())
        return {"status": "STAGE7_PREFLIGHT_PASS" if masks["path_contract_verified"] else "STAGE7_PREFLIGHT_BLOCKED_MASK_PATHS",
                "identity": identity, "dependencies": dependencies,
                "image_or_clinical_values_accessed": False, "training_count": 92, "validation_count": 23,
                "mask_headers": masks, "ready_for_real_execution": masks["path_contract_verified"]}
    except Exception:
        raise XAIFailure(f"STAGE7_PREFLIGHT_FAIL phase={phase}") from None


def verify_smoke(run_id, revision, *, dependencies=None):
    identity = runtime.source_identity(revision)
    paths = runtime.run_path(run_id, "smoke", resume=True)
    record = original.read_private_json(paths.fold_dir/"synthetic_gate.json")
    if (set(record) != {"status", "identity", "dependencies", "patient_data_access", "clinical_tests", "gpu_tests", "device_checks"}
            or record["status"] != "XAI_STAGE7AB_UIS_SYNTHETIC_PASS"
            or record["identity"] != identity or record["patient_data_access"] is not False
            or (dependencies is not None and record["dependencies"] != dependencies)
            or record["clinical_tests"] != {"tests": 1, "skipped": 0}
            or record["gpu_tests"].get("tests", 0) < 20 or record["gpu_tests"].get("skipped") != 0
            or set(record["device_checks"]) != {"ncct", "ctp"}):
        raise ValueError("complete same-source UiS CUDA smoke required")
    for checks in record["device_checks"].values():
        if checks != {"sklearn_equivalence_cases": 6, "complete_quantitative_cases": 1, "frozen_weights_unchanged": True}:
            raise ValueError("incomplete synthetic device checks")
    runtime.validate_dependencies(record["dependencies"])
    return record


def _check_torch_equivalence(result, validation, device):
    import torch
    from .xai_fused_score import torch_state, fused_logit_torch
    state = torch_state(result["state"], device)
    if (len(result["clinical_validation"]) != len(validation) or len(result["logits"]) != len(validation)
            or len(result["probabilities"]) != len(validation) or len(result["clinical_logits"]) != len(validation)):
        raise ValueError("private reconstructed array count mismatch")
    with torch.no_grad():
        for i, embedding in enumerate(validation):
            e = torch.as_tensor(embedding, device=device).reshape(1, 128)
            c = torch.as_tensor(result["clinical_validation"][i:i+1], device=device)
            z = fused_logit_torch(e, c, state)
            if abs(z.item()-result["logits"][i]) > 1e-8 or abs(z.sigmoid().item()-result["probabilities"][i]) > 1e-10:
                raise ValueError("Torch/sklearn equivalence failed before CAM")
    return state


def execute(mode, revision, run_id, *, execute_real=False, protocol_approval=None, smoke_run=None, resume=False, resource_run=None):
    if mode not in ("smoke", "ncct", "ctp"):
        raise ValueError("explicit Stage7 run mode required")
    if mode != "smoke" and (execute_real is not True or not protocol_approval or not smoke_run):
        raise ValueError("real execution requires explicit authorization, protocol review and smoke identity")
    if mode == "smoke" and (resume or execute_real or protocol_approval or smoke_run or resource_run):
        raise ValueError("synthetic smoke requires a fresh independent run")
    if mode == "ctp" and not resource_run:
        raise ValueError("CTP execution requires a reviewed native synthetic resource run")
    os.umask(0o077)
    phase, paths, resources = "source", None, None
    started = time.monotonic()
    try:
        identity = runtime.source_identity(revision)
        if mode != "smoke":
            runtime.approved_protocol(identity, protocol_approval, execute_real)
        phase = "environment"
        dependencies = original.environment()
        dependencies["clinical"] = runtime.bridge({"probe": True})["dependencies"]
        import torch
        from .ncct_training import _seed_everything, _parameter_digest, _membership_record
        _seed_everything(630)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if mode != "smoke":
            phase = "same_source_smoke"
            verify_smoke(smoke_run, revision, dependencies=dependencies)
        phase = "exclusive_output"
        paths = runtime.run_path(run_id, mode, resume=resume)
        os.environ["TMPDIR"] = str(paths.fold_dir)
        import tempfile
        tempfile.tempdir = None
        from .xai_store import run_lock
        with run_lock(paths.fold_dir):
            if mode == "smoke":
                phase = "clinical_synthetic_tests"
                clinical_tests = runtime.bridge({"synthetic_tests": True})
                phase = "gpu_stack_tests"
                from .xai_smoke import gpu_test_suite, device_smoke
                tests = gpu_test_suite(original.PROJECT)
                phase = "cuda_fictional_pipeline"
                checked = device_smoke("cuda", runtime.bridge, revision)
                if runtime.source_identity(revision) != identity:
                    raise ValueError("source changed during synthetic execution")
                original.publish(paths.fold_dir/"synthetic_gate.json", {
                    "status": "XAI_STAGE7AB_UIS_SYNTHETIC_PASS", "identity": identity,
                    "dependencies": dependencies, "patient_data_access": False,
                    "clinical_tests": {k: clinical_tests[k] for k in ("tests", "skipped")},
                    "gpu_tests": tests, "device_checks": checked})
            else:
                phase = "contract_checkpoint"
                from .fusion_runner import load_plan, check_clinical_path, real_features
                from .fusion_checkpoints import load_ncct, load_ctp
                contract, plan = load_plan()
                check_clinical_path()
                phase = "anatomical_mask_headers"
                from .xai_masks import real_mask_preflight
                mask_headers = real_mask_preflight(contract, plan)
                if not mask_headers["path_contract_verified"]:
                    raise ValueError("declared anatomical mask paths require review")
                if mode == "ctp":
                    phase = "native_resource_gate"
                    runtime.verify_resource(resource_run, revision, dependencies, mask_headers["maximum_shape_zyx"]["ctp"])
                phase = "checkpoint"
                model, provenance = load_ncct(plan) if mode == "ncct" else load_ctp(contract)
                before = _parameter_digest(model)
                from .xai_resources import ResourceProfile, validate_profile
                resources = ResourceProfile("cuda")
                phase = "train_only_reconstruction"
                with resources.attach(model), resources.phase("feature_extraction"):
                    if mode == "ctp":
                        from .xai_data import real_ctp_features
                        train, validation, probabilities = real_ctp_features(contract, plan, model, resources=resources)
                    else:
                        train, validation, probabilities = real_features(contract, plan, mode, model)
                result = runtime.bridge({"mode": "real", "plan": asdict(plan), "train": train.tolist(),
                                         "validation": validation.tolist(), "imaging_probabilities": probabilities.tolist(),
                                         "source_revision": revision})
                phase = "strict_historical_gate"
                membership = _membership_record(plan)
                historical = original.read_private_json(original.RUNTIME/"fusion_runs"/original.RUNS[mode]/"outer_fold_0/aggregate_training.json")
                runtime.historical_gate(result, mode, historical, provenance=provenance, membership=membership)
                phase = "sklearn_torch_gate"
                state = _check_torch_equivalence(result, validation, "cuda")
                del train, validation, probabilities, historical
                manifest = {"identity": identity, "dependencies": dependencies, "mode": mode,
                            "checkpoint": provenance, "membership": membership, "smoke_run": smoke_run,
                            "protocol_approval": protocol_approval, "case_count": 23,
                            "reconstruction_sha256": runtime.record_digest(result),
                            "resource_run": resource_run, "mask_header_preflight": mask_headers,
                            "full_perturbation_case_count": 23}
                phase = "resume_identity"
                manifest_path = paths.fold_dir/"manifest.json"
                if resume:
                    if original.read_private_json(manifest_path) != manifest:
                        raise ValueError("resume provenance/reconstruction mismatch")
                    if (paths.fold_dir/"aggregate_xai.json").exists():
                        raise ValueError("completed aggregate run cannot be resumed")
                else:
                    original.publish(manifest_path, manifest)
                signature = runtime.record_digest(manifest)
                from .xai_data import real_validation_access
                from .xai_analysis import analyse_case
                from .xai_store import read_case, write_case
                from .xai_schema import aggregate_cases
                access = real_validation_access(contract, plan, mode, "cuda")
                cases = []
                for i, patient in enumerate(plan.inner_validation):
                    phase = "development_case_analysis"
                    record = read_case(paths.fold_dir, i, signature) if resume else None
                    if record is None:
                        raw, affine, shape_xyz = access.load(patient)
                        from .xai_ctp_cache import cached_series
                        # Cache closes on success AND every analysis/mask failure.
                        with cached_series(raw, resources=resources) if mode == "ctp" else nullcontext(raw) as value:
                            del raw
                            labels, mask_reason = access.labels(patient, shape_xyz, affine)
                            c = torch.as_tensor(result["clinical_validation"][i:i+1], device="cuda")
                            with resources.attach(model), resources.phase("case_total"):
                                record = analyse_case(model, value, c, state, mode, affine, labels_xyz=labels,
                                                      mask_reason=mask_reason, expected_logit=result["logits"][i],
                                                      clinical_logit=result["clinical_logits"][i], resources=resources)
                        if before != _parameter_digest(model) or any(p.grad is not None or p.requires_grad for p in model.parameters()):
                            raise ValueError("frozen model changed during XAI")
                        write_case(paths.fold_dir, i, signature, record)
                        del value, labels, c
                    cases.append(record)
                phase = "aggregate_publication"
                if before != _parameter_digest(model) or runtime.source_identity(revision) != identity:
                    raise ValueError("frozen source/model changed before publication")
                public = {"schema": 1, "status": "XAI_STAGE7C_AGGREGATE_PASS", "mode": mode,
                          "source_revision": revision, "protocol_sha256": identity["protocol_sha256"],
                          "validation_count": 23, "patient_data_access": True, "outer_test_evaluated": False,
                          "between_seed_stability_assessed": False, "summary": aggregate_cases(cases),
                          "dependencies": dependencies}
                runtime.validate_public_record(public)
                # Aggregate phase totals only; never per-patient timings/values.
                job = os.environ["SLURM_JOB_ID"]
                original.publish(paths.fold_dir/f"resources_job_{job}.json", validate_profile(resources.record()))
                original.publish(paths.fold_dir/"aggregate_xai.json", public)
                # Environment/source metadata stays private and is bound by the
                # manifest; the public reader exports only an exact allowlist.
        return {"status": "PASS", "mode": mode, "patient_data_access": mode != "smoke",
                "outer_test_evaluated": False}
    except Exception as error:
        if paths is not None:
            # Exclusive diagnostic filename per scheduler attempt; never replace
            # an earlier failure or echo arbitrary exception contents.
            job = os.environ.get("SLURM_JOB_ID", "")
            if job.isdigit():
                diagnostic = paths.fold_dir/f"failure_job_{job}.json"
                if not diagnostic.exists():
                    try:
                        record = {"phase": phase, "exception_type": type(error).__name__, "elapsed_seconds": time.monotonic()-started}
                        if resources is not None:
                            try:
                                record["aggregate_resources"] = resources.record()
                            except Exception:
                                pass
                        original.publish(diagnostic, record)
                    except Exception:
                        pass
        raise XAIFailure(f"STAGE7_FAIL phase={phase}; inspect owner-private diagnostics") from None


def execute_resource(revision, run_id, shape_zyx):
    """Separate patient-free allocated-GPU mode; never fits or loads a checkpoint."""
    os.umask(0o077)
    phase, paths = "source", None
    try:
        identity = runtime.source_identity(revision)
        phase = "environment"
        dependencies = original.environment()
        dependencies["clinical"] = runtime.bridge({"probe": True})["dependencies"]
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        phase = "exclusive_output"
        paths = runtime.run_path(run_id, "resource")
        phase = "native_synthetic_resource_check"
        from .xai_store import run_lock
        from .xai_resource_check import synthetic_resource_check
        with run_lock(paths.fold_dir):
            check = synthetic_resource_check(tuple(shape_zyx), device="cuda", fixture_path=paths.fold_dir/"fictional_resource_ctp.nii.gz")
            if runtime.source_identity(revision) != identity:
                raise ValueError("source changed during native resource check")
            original.publish(paths.fold_dir/"synthetic_resources.json", {"identity": identity, "dependencies": dependencies, "check": check})
        return {"status": "SYNTHETIC_RESOURCE_CHECK_RECORDED", "patient_data_access": False,
                "shape_zyx": check["shape_zyx"], "profile": check["profile"], "cache": check["cache"],
                "projected_23_case_seconds_with_2x_margin": check["projected_23_case_seconds_with_2x_margin"],
                "within_24h_with_20pct_headroom": check["within_24h_with_20pct_headroom"]}
    except Exception as error:
        if paths is not None:
            try:
                original.publish(paths.fold_dir/"resource_failure.json", {"phase": phase, "exception_type": type(error).__name__})
            except Exception:
                pass
        raise XAIFailure(f"STAGE7_RESOURCE_FAIL phase={phase}; no resource verification issued") from None
