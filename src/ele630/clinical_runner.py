"""Restricted Stage 5B UiS adapter; imports perform no I/O or training."""

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import re
import socket
import sys
import warnings
from pathlib import Path
from typing import TextIO

from .clinical_baseline import run_clinical_development
from .clinical_preprocessing import FEATURE_NAMES
from .config import ServerPaths
from .data_contract import load_and_validate
from .ncct_private_output import create_private_run, open_private_binary
from .ncct_splits import FoldPlan, plan_outer_fold, validate_fold_plan


class ClinicalRunnerError(RuntimeError):
    """Sanitized runner failure; never echo paths, IDs, cells or traceback context."""


_APPROVED_PYTHON = None
_APPROVED_VERSIONS = {
    "numpy": "1.24.3", "scipy": "1.10.1", "sklearn": "1.2.2",
    "joblib": "1.1.1", "threadpoolctl": "2.2.0",
}
_SOURCE_ID = re.compile(r"(?:sub[-_]stroke[-_])?0?([123])[_-]([0-9]{3})\Z")


def _identifier(raw: str) -> str:
    """Trim IDs and normalize anchored source aliases; no substring matching."""
    value = raw.strip().lower()
    match = _SOURCE_ID.fullmatch(value)
    if match:
        return f"sub-stroke_{int(match[1]):02d}_{match[2]}"
    return value


def read_authorized_clinical(stream: TextIO, plan: FoldPlan) -> dict[str, dict[str, str]]:
    """Retain approved features only after development-ID authorization.

    CSV lexical tokenization is unavoidable. Excluded rows are checked only for
    structural width and identifier duplication, then discarded. Their clinical
    cells are never indexed, interpreted, validated or retained in the mapping.
    Only transient identifiers are used to detect duplicates across the file.
    Empty IDs (known non-patient rows) are discarded without inspecting cells.
    """
    try:
        validate_fold_plan(plan)
        if plan.outer_fold != 0 or plan.seed != 630 or plan.validation_fraction != 0.20:
            raise ValueError
        authorized = {p.patient_id for p in plan.inner_train + plan.inner_validation}
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if (len(header) != len(set(header)) or any(not name for name in header)
                or not {"patient_id", *FEATURE_NAMES}.issubset(header)):
            raise ValueError
        id_column = header.index("patient_id")
        columns = {name: header.index(name) for name in FEATURE_NAMES}
        selected = {}
        seen_ids = set()
        for cells in reader:
            if len(cells) != len(header):
                raise ValueError
            identifier = _identifier(cells[id_column])
            if not identifier:
                del cells
                continue
            if identifier in seen_ids:
                raise ValueError
            seen_ids.add(identifier)
            if identifier not in authorized:
                del cells
                continue
            selected[identifier] = {name: cells[column] for name, column in columns.items()}
            del cells
        if set(selected) != authorized:
            raise ValueError
        return selected
    except Exception:
        raise ClinicalRunnerError("clinical CSV schema, linkage or authorized coverage is invalid") from None


def _execution_metadata() -> dict:
    """Require the approved CPU worker/interpreter and exact verified stack."""
    try:
        if (_APPROVED_PYTHON is None or os.name != "posix"
                or not os.environ.get("SLURM_JOB_ID", "").isdigit()
                or os.environ.get("SLURM_JOB_PARTITION") != "cpu20"
                or socket.gethostname().split(".")[0] != os.environ.get("SLURMD_NODENAME")
                or Path(sys.executable).resolve(strict=True) != _APPROVED_PYTHON.resolve(strict=True)
                or platform.python_version() != "3.10.11"):
            raise ValueError
        versions = {}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for name, expected in _APPROVED_VERSIONS.items():
                module = importlib.import_module(name)
                versions[name] = module.__version__
                if versions[name] != expected:
                    raise ValueError
        if captured:
            raise ValueError
        return {"python": "3.10.11", **versions}
    except Exception:
        raise ClinicalRunnerError("approved UiS cpu20 worker and dependency environment are required") from None


def _load_plan(paths: ServerPaths) -> FoldPlan:
    """Default active-profile validation only; no alternate profile or split input."""
    try:
        contract = load_and_validate(paths.manifest, paths.folds)
        return plan_outer_fold(contract, 0, seed=630, validation_fraction=0.20)
    except Exception:
        raise ClinicalRunnerError("active dataset contract or canonical development plan is invalid") from None


def execute_clinical_runner(
    paths: ServerPaths, clinical_csv: Path, output_dir: Path, run_id: str,
    *, execute_real: bool = False,
) -> dict:
    """Guarded server-only action: one private aggregate artifact, no checkpoints.

    No public synthetic mode or output-root override. Tests patch only trusted
    boundaries and use temporary fictional files. A new exclusive run is reserved
    after input validation and before fitting; failure may leave an empty run.
    """
    if execute_real is not True:
        raise ClinicalRunnerError("explicit real-data execution acknowledgement is required")
    dependencies = _execution_metadata()
    try:
        paths.check_server_locations()
        source = Path(clinical_csv).resolve(strict=True)
        dataset = paths.dataset_root.resolve(strict=True)
        project = paths.project_root.resolve(strict=True)
        output = Path(output_dir).resolve(strict=True)
        if (not source.is_file() or not source.is_relative_to(dataset)
                or output == project or output.is_relative_to(project)
                or output == dataset or output.is_relative_to(dataset)):
            raise ValueError
    except Exception:
        raise ClinicalRunnerError("server-local input or private output location is invalid") from None
    plan = _load_plan(paths)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = read_authorized_clinical(stream, plan)
    except Exception:
        raise ClinicalRunnerError("authorized clinical source could not be loaded") from None
    try:
        # Preserve the original spelling so the existing helper can reject
        # caller-supplied symlinks below the trusted root before resolution.
        destination = create_private_run(output_dir, run_id, 0)
    except Exception:
        raise ClinicalRunnerError("exclusive private run could not be reserved") from None
    try:
        result = run_clinical_development(plan, rows)
        record = {
            "stage": "5B", "evaluation": "descriptive_fold0_inner_validation",
            "outer_test_clinical_processed": False,
            "configuration": {
                "outer_fold": 0, "seed": 630, "validation_fraction": 0.20,
                "features": list(FEATURE_NAMES), "positive_class": 1,
                "penalty": "l2", "C": 1.0, "fit_intercept": True,
                "class_weight": None, "solver": "lbfgs", "max_iter": 1000,
                "tol": 1e-8, "warm_start": False, "verbose": 0, "threshold": 0.5,
            },
            "dependencies": dependencies, "results": result.as_dict(),
        }
        payload = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with open_private_binary(destination.metrics_path) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return record
    except Exception:
        raise ClinicalRunnerError("clinical development or private aggregate publication failed") from None
