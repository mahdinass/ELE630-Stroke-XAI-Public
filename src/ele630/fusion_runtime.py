"""Stage 5C3 archive guards and aggregate-only publication logic."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path

from .ncct_private_output import _validate_base, open_private_binary

_ARCHIVE_ROOT = Path("/ele630-restricted-archive")
PROJECT = _ARCHIVE_ROOT / "source"
RUNTIME = _ARCHIVE_ROOT / "runtime"
DATASET = _ARCHIVE_ROOT / "dataset"
PYTHON = _ARCHIVE_ROOT / "python"
GPU_PACKAGES = RUNTIME / ".venv/lib/python3.10/site-packages"
BASE_REVISION = "c3ef2902d12e7d8c01acabf790e06a6b7b7dac23"
NCCT_REVISION = "714ce2cbbbe1e235f9cc5e95bb5540747d20d991"
PROTOCOL = "docs/stage5c3_clinical_imaging_fusion_protocol.md"
RUNS = {"smoke": "stage5c3_smoke_02", "ncct": "stage5c3_ncct_01", "ctp": "stage5c3_ctp_01"}
METRICS = {"bce", "auroc", "average_precision", "brier_score", "balanced_accuracy_0_5",
           "mean_predicted_probability", "validation_count", "validation_good_count",
           "validation_poor_count", "probability_sd", "probability_min", "probability_max"}
MODELS = {"imaging_only", "fusion", "clinical_only", "no_image", "clinical_prior_shrinkage_0_5", "training_prior"}


def git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT, check=True,
                          capture_output=True).stdout


def source_identity(expected):
    if not re.fullmatch("[0-9a-f]{40}", expected or ""):
        raise ValueError("registered source revision required")
    if PROJECT == _ARCHIVE_ROOT / "source" or Path(__file__).resolve().parents[2] != PROJECT.resolve(strict=True):
        raise ValueError("approved UiS source checkout required")
    if git("rev-parse", "HEAD").decode().strip() != expected:
        raise ValueError("checkout changed since submission")
    git("merge-base", "--is-ancestor", BASE_REVISION, "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked source changes are not permitted")
    # Untracked Python cannot silently replace the reviewed implementation.
    if git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts/experiments", "scripts/slurm", "tests"):
        raise ValueError("untracked executable source is not permitted")
    tracked = git("ls-files", "--", "src", "scripts/experiments", "scripts/slurm", "tests", "docs").decode().splitlines()
    required = {"src/ele630/fusion.py", "src/ele630/fusion_runner.py", PROTOCOL}
    if not required.issubset(tracked):
        raise ValueError("Stage 5C3 must be reviewed and committed before execution")
    for name in tracked:
        if name.endswith((".py", ".sh", ".sbatch", ".md")):
            path = PROJECT / name
            if path.is_symlink() or path.read_bytes() != git("show", f"HEAD:{name}"):
                raise ValueError("source bytes differ from registered commit")
    return {"source_revision": expected, "protocol_sha256": hashlib.sha256((PROJECT / PROTOCOL).read_bytes()).hexdigest()}


def private_file(path):
    path = Path(path)
    _validate_base(path.parent)
    if path.is_symlink() or not path.is_file():
        raise ValueError("private artifact is missing or unsafe")
    info = path.stat()
    if os.name == "posix" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private artifacts must have mode 0600")
    return path


def read_private_json(path):
    return json.loads(private_file(path).read_text(encoding="utf-8"))


def environment(*, gpu=True):
    if (PYTHON == _ARCHIVE_ROOT / "python" or os.name != "posix"
            or platform.python_version() != "3.10.11"
            or Path(sys.executable).resolve(strict=True) != PYTHON.resolve(strict=True)):
        raise ValueError("documented UiS Python 3.10.11 interpreter required")
    if gpu and (not os.environ.get("SLURM_JOB_ID", "").isdigit()
                or os.environ.get("SLURM_JOB_PARTITION") != "gpu"
                or socket.gethostname().split(".")[0] != os.environ.get("SLURMD_NODENAME")):
        raise ValueError("allocated UiS gpu worker required")
    import numpy, nibabel, torch
    if (numpy.__version__ != "1.26.4" or nibabel.__version__ != "5.4.2"
            or str(torch.__version__) != "2.3.1+cu118" or torch.version.cuda != "11.8"):
        raise ValueError("documented GPU dependency stack required")
    if gpu:
        if (not os.environ.get("CUDA_VISIBLE_DEVICES") or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1):
            raise ValueError("exactly one allocated CUDA GPU required")
        torch.zeros(1, device="cuda")  # Initialize before allocator statistics.
        torch.cuda.reset_peak_memory_stats()
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "torch": str(torch.__version__), "nibabel": nibabel.__version__}


def clinical_environment():
    # No inherited GPU PYTHONPATH: preserve the frozen Stage 5B NumPy/sklearn stack.
    result = dict(os.environ)
    result["PYTHONPATH"] = str(PROJECT / "src")
    result["PYTHONNOUSERSITE"] = "1"
    result["OMP_NUM_THREADS"] = result["OPENBLAS_NUM_THREADS"] = result["MKL_NUM_THREADS"] = "1"
    return result


def bridge(payload):
    proc = subprocess.run([str(PYTHON), "-B", "-m", "ele630.fusion_clinical_worker"],
                          input=json.dumps(payload, allow_nan=False), text=True, capture_output=True,
                          env=clinical_environment(), cwd=RUNTIME, timeout=1800)
    # Never echo subprocess diagnostics: pipes may have contained clinical inputs.
    if proc.returncode or proc.stderr.strip():
        raise ValueError("clinical worker dependency, fitting or IPC check failed")
    return json.loads(proc.stdout)


def validate_result(result):
    """Strict output allowlist: never serialize arbitrary caller-supplied objects."""
    import math
    expected = {"metrics", "fusion_minus_reference", "training_count", "validation_count",
                "image_components", "clinical_features", "fusion_features", "outer_test_evaluated"}
    if type(result) is not dict or set(result) != expected or set(result["metrics"]) != MODELS:
        raise ValueError("aggregate output schema rejected")
    if (result["outer_test_evaluated"] is not False or result["image_components"] != 8
            or result["clinical_features"] != 11 or result["fusion_features"] != 19):
        raise ValueError("aggregate output contract rejected")
    for record in result["metrics"].values():
        if set(record) != METRICS:
            raise ValueError("unexpected metric fields")
        for value in record.values():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid aggregate number")
        counts = ("validation_count", "validation_good_count", "validation_poor_count")
        if (any(type(record[k]) is not int or record[k] <= 0 for k in counts)
                or record["validation_count"] != result["validation_count"]
                or record["validation_good_count"] + record["validation_poor_count"] != result["validation_count"]):
            raise ValueError("inconsistent aggregate counts")
        if record["bce"] < 0 or any(not 0 <= record[k] <= 1 for k in METRICS - set(counts) - {"bce"}):
            raise ValueError("aggregate metric range invalid")
    references = MODELS - {"fusion", "training_prior"}
    if set(result["fusion_minus_reference"]) != references:
        raise ValueError("invalid comparison schema")
    keys = {"bce", "auroc", "average_precision", "brier_score", "balanced_accuracy_0_5", "probability_sd"}
    for record in result["fusion_minus_reference"].values():
        if set(record) != keys or any(type(v) not in (int, float) or not math.isfinite(v) for v in record.values()):
            raise ValueError("invalid comparison numbers")
    for name, record in result["fusion_minus_reference"].items():
        if any(abs(value - (result["metrics"]["fusion"][k] - result["metrics"][name][k])) > 1e-12
               for k, value in record.items()):
            raise ValueError("comparison arithmetic disagrees")
    if any(type(result[k]) is not int or result[k] <= 0 for k in ("training_count", "validation_count")):
        raise ValueError("invalid aggregate counts")
    return result


def publish(path, record):
    with open_private_binary(path) as stream:
        stream.write((json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
