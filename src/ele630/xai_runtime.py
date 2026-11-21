"""UiS-only source/privacy guards. No patient access or output at import."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess

from . import fusion_runtime as original
from .ncct_private_output import _validate_base, create_private_run, open_private_run
from .xai_schema import validate_aggregate

BASE_REVISION = "54f289161a35206c00a517d7f12e2fdf1e7a4127"
PROTOCOL = "docs/stage7_xai_protocol.md"
PLAN = "docs/ELE630_Strategy_V4_XAI_Full_Implementation_Plan.md"
SLUG = re.compile(r"stage7_(smoke|resource|ncct|ctp)_[a-zA-Z0-9_-]{1,42}\Z")


def source_identity(revision):
    identity = original.source_identity(revision)
    original.git("merge-base", "--is-ancestor", BASE_REVISION, "HEAD")
    tracked = original.git("ls-files").decode().splitlines()
    required = {PROTOCOL, PLAN, "src/ele630/xai_runner.py", "src/ele630/xai_spatial.py",
                "src/ele630/xai_clinical_worker.py", "tests/test_xai_foundation.py",
                "scripts/slurm/submit_stage7.sh"}
    if not required.issubset(tracked):
        raise ValueError("reviewed and committed Stage7 sources required")
    # All original source contracts are immutable during this additive workflow.
    for name in original.git("ls-tree", "-r", "--name-only", BASE_REVISION, "src").decode().splitlines():
        if original.git("show", f"{revision}:{name}") != original.git("show", f"{BASE_REVISION}:{name}"):
            raise ValueError("original source contract changed")
    identity.update(protocol_sha256=hashlib.sha256((original.PROJECT/PROTOCOL).read_bytes()).hexdigest(),
                    implementation_plan_sha256=hashlib.sha256((original.PROJECT/PLAN).read_bytes()).hexdigest(),
                    baseline_revision=BASE_REVISION)
    return identity


def bridge(payload):
    """Private pipes only; inherited GPU site-packages removed by old guard."""
    environment = original.clinical_environment()
    environment["ELE630_XAI_PRIVATE_IPC"] = "1"
    proc = subprocess.run([str(original.PYTHON), "-B", "-m", "ele630.xai_clinical_worker"],
                          input=json.dumps(payload, allow_nan=False), text=True, capture_output=True,
                          env=environment, cwd=original.RUNTIME, timeout=1800)
    if proc.returncode or proc.stderr.strip():
        raise ValueError("Stage7 clinical dependency or private IPC failure")
    return json.loads(proc.stdout)


def run_path(run_id, mode, *, resume=False):
    match = SLUG.fullmatch(run_id or "")
    if not match or match[1] != mode:
        raise ValueError("exclusive Stage7 modality-specific run identity required")
    base = original.RUNTIME/"stage7_runs"
    return (open_private_run if resume else create_private_run)(base, run_id, 0)


def approved_protocol(identity, approval, execute_real):
    if execute_real is not True or approval != identity["protocol_sha256"]:
        raise ValueError("explicit real-execution flag and reviewed protocol SHA256 required")


def validate_public_record(record):
    """Reject arbitrary run content; reader never echoes unknown values/strings."""
    keys = {"schema", "status", "mode", "source_revision", "protocol_sha256", "validation_count",
            "patient_data_access", "outer_test_evaluated", "between_seed_stability_assessed", "summary", "dependencies"}
    if type(record) is not dict or set(record) != keys or record["schema"] != 1:
        raise ValueError("public aggregate schema rejected")
    mode = record["mode"]
    if (mode not in ("ncct", "ctp") or record["status"] != "XAI_STAGE7C_AGGREGATE_PASS"
            or record["validation_count"] != 23 or record["patient_data_access"] is not True
            or record["outer_test_evaluated"] is not False
            or record["between_seed_stability_assessed"] is not False
            or not re.fullmatch("[0-9a-f]{40}", record["source_revision"] or "")
            or not re.fullmatch("[0-9a-f]{64}", record["protocol_sha256"] or "")):
        raise ValueError("public aggregate contract rejected")
    validate_aggregate(record["summary"], 23)
    validate_dependencies(record["dependencies"])
    return record


def validate_dependencies(dependencies):
    from .clinical_runner import _APPROVED_VERSIONS
    if dependencies != {"python": "3.10.11", "numpy": "1.26.4", "torch": "2.3.1+cu118",
                        "nibabel": "5.4.2", "clinical": _APPROVED_VERSIONS}:
        raise ValueError("unverified environment metadata rejected")
    return dependencies


def read_results(run_id, mode, revision):
    identity = source_identity(revision)
    path = run_path(run_id, mode, resume=True).fold_dir
    manifest = original.read_private_json(path/"manifest.json")
    if manifest["identity"] != identity or manifest["mode"] != mode:
        raise ValueError("aggregate source identity mismatch")
    record = validate_public_record(original.read_private_json(path/"aggregate_xai.json"))
    if record["source_revision"] != revision or record["protocol_sha256"] != identity["protocol_sha256"] or record["mode"] != mode:
        raise ValueError("aggregate provenance mismatch")
    return record


def historical_gate(result, modality, historical, *, provenance, membership):
    """Frozen tolerances, baseline artifact identity and historical means required."""
    if (historical.get("status") != "PASS" or historical.get("mode") != modality
            or historical.get("identity", {}).get("source_revision") != BASE_REVISION
            or historical.get("outer_test_evaluated") is not False
            or historical.get("patient_data_access") is not True
            or historical.get("provenance", {}).get("membership") != membership
            or any(historical.get("provenance", {}).get(k) != v for k, v in provenance.items())):
        raise ValueError("original Stage5C3 private reference identity mismatch")
    baseline = original.validate_result(historical["results"])
    if (baseline["training_count"], baseline["validation_count"]) != (92, 23):
        raise ValueError("historical development counts mismatch")
    fixed = {"ncct": {"fusion": (.2452544303, .0745231080, .9732142857, .7857142857),
                       "imaging_only": (.6143876931, .2116712247, .6160714286, .5)},
             "ctp": {"fusion": (.1720028729, .0482148614, 1., .8571428571),
                      "imaging_only": (.6143472110, .2116544701, .6785714286, .5)}}[modality]
    fixed["clinical_only"] = (.2313358984, .0663344280, .9821428571, .9285714286)
    for model, reference in fixed.items():
        for metric, number in zip(("bce", "brier_score", "auroc", "balanced_accuracy_0_5"), reference):
            value = result["metrics"][model][metric]
            if not math.isfinite(value) or abs(value-number) > 5e-7:
                raise ValueError("fixed historical metric reproduction failed")
        for metric in ("bce", "brier_score", "mean_predicted_probability", "auroc", "balanced_accuracy_0_5"):
            a, b = result["metrics"][model][metric], baseline["metrics"][model][metric]
            if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 5e-7:
                raise ValueError("private historical reference reproduction failed")
    from .fusion_runner import _validate_no_image_control
    _validate_no_image_control(result)


def record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verify_resource(run_id, revision, dependencies, required_shape):
    from .xai_resources import validate_profile, ctp_workload, planning_seconds, RESOURCE_VERSION
    from .xai_ctp_cache import validate_cache_record, prepared_bytes, MAX_CACHE_BYTES
    identity = source_identity(revision)
    directory = run_path(run_id, "resource", resume=True).fold_dir
    record = original.read_private_json(directory/"synthetic_resources.json")
    if set(record) != {"identity", "dependencies", "check"} or record["identity"] != identity or record["dependencies"] != dependencies:
        raise ValueError("same-source/dependency synthetic resource check required")
    check = record["check"]
    expected = {"schema", "resource_version", "shape_zyx", "device", "patient_data_access", "io_backend", "indexed_gzip_available",
                "profile", "workload", "projected_23_case_seconds_with_2x_margin", "within_24h_with_20pct_headroom", "cache"}
    if (set(check) != expected or check["schema"] != RESOURCE_VERSION or check["resource_version"] != RESOURCE_VERSION or check["device"] != "cuda"
            or check["patient_data_access"] is not False or check["io_backend"] != "nifti_gzip_proxy"
            or check["workload"] != ctp_workload() or check["within_24h_with_20pct_headroom"] is not True
            or len(check["shape_zyx"]) != 3 or len(required_shape) != 3
            or any(type(n) is not int or n < needed for n, needed in zip(check["shape_zyx"], required_shape))):
        raise ValueError("native resource check shape, workload or planning bound rejected")
    cache = validate_cache_record(check["cache"])
    if (cache["backend"] != "sequential_nifti_proxy" or cache["frame_hits"] != 47
            or cache["limit_bytes"] != MAX_CACHE_BYTES
            or cache["prepared_bytes"] != prepared_bytes(tuple(reversed(check["shape_zyx"])))):
        raise ValueError("native sequential CPU cache samples required")
    profile = validate_profile(check["profile"])
    if (type(profile["peak_cuda_allocated_bytes"]) is not int or profile["peak_cuda_allocated_bytes"] <= 0
            or type(profile["peak_cuda_reserved_bytes"]) is not int or profile["peak_cuda_reserved_bytes"] < profile["peak_cuda_allocated_bytes"]
            or profile["encoder_forwards"] != 50 or any(profile["phases"][p]["calls"] != 1 for p in (
            "resource_fixture", "resource_warmup", "resource_forward", "resource_baseline", "resource_cam",
            "resource_upper_cam", "resource_spatial", "resource_temporal", "resource_cache_creation"))
            or profile["cache"] != {"builds": 1, "decoded_frames": 40, "frame_hits": 47,
                                   "peak_prepared_bytes": cache["prepared_bytes"], "limit_bytes": MAX_CACHE_BYTES}):
        raise ValueError("incomplete native synthetic resource samples")
    estimate = planning_seconds(profile)
    if abs(check["projected_23_case_seconds_with_2x_margin"]-estimate) > 1e-6 or estimate > .8*24*3600:
        raise ValueError("native CTP planning budget exceeded")
    return record
