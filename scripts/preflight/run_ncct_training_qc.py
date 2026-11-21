#!/usr/bin/env python3
"""Stage 5C2D: default data-free plan; explicit private UiS CPU-worker QC only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
DEFAULT_PROJECT_ROOT = Path(__file__).absolute().parents[2]


def _worker_available() -> bool:
    return os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-qc", action="store_true")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    for name in ("manifest", "folds", "dataset-root", "private-output-dir", "protocol"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)
    if not args.execute_qc:
        print(json.dumps({"stage": "5C2D", "mode": "plan-only", "patient_data_access": False,
                          "outer_fold": 0, "seed": 630, "validation_fraction": .20,
                          "active_fingerprint": "3db8c4fd5fef0e4e", "training_cases": 92,
                          "validation_images_accessed": False, "outer_images_accessed": False,
                          "maximum_visual_cases": 8, "input_window": [-20, 120],
                          "spatial_changes": False, "anatomy_certified": False}, sort_keys=True))
        return 0
    if (not _worker_available()
            or any(v is None for v in (args.private_output_dir, args.run_id, args.source_revision, args.protocol))):
        print("NCCT QC requires a UiS worker and private run/provenance arguments", file=sys.stderr)
        return 2
    stage = "provenance"
    try:
        from ele630.config import ServerPaths
        from ele630.data_contract import load_and_validate
        from ele630.ncct_anatomy_qc import TrainingOnlyNCCTQC, validate_qc_plan, scan_training_qc, write_private_qc
        from ele630.ncct_private_output import create_private_run
        if (len(args.source_revision) != 40
                or any(c not in "0123456789abcdef" for c in args.source_revision)):
            raise ValueError("invalid revision")
        actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.project_root,
                                capture_output=True, text=True, check=True).stdout.strip()
        if actual != args.source_revision:
            raise ValueError("revision mismatch")
        protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        paths = ServerPaths.from_overrides(project_root=args.project_root, manifest=args.manifest,
                                           folds=args.folds, dataset_root=args.dataset_root)
        stage = "contract"
        contract = load_and_validate(paths.manifest, paths.folds)
        plan = validate_qc_plan(contract)
        stage = "private_output"
        run = create_private_run(args.private_output_dir, args.run_id, 0)
        stage = "repository_setup"
        # Import/construct the image loader only after contract/split validation.
        from ele630.ncct import NCCTRepository
        guard = TrainingOnlyNCCTQC(contract, NCCTRepository(contract, paths.dataset_root), plan=plan)
        stage = "training_integrity"
        report = scan_training_qc(guard)
        report["provenance"] = {
            "source_revision": actual, "protocol_sha256": protocol_digest,
            "source_sha256": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "python": sys.version.split()[0], "fingerprint": contract.fingerprint,
            "outer_fold": 0, "seed": 630, "validation_fraction": .20,
            "input_policy": "NIfTI-proxy scaling; native float32 czyx; clip [-20,120] to [0,1]",
        }
        import nibabel, numpy
        from ele630 import ncct_anatomy_qc, ncct, native_ncct_input, ncct_intensity, ncct_splits, data_contract
        for module in (ncct_anatomy_qc, ncct, native_ncct_input, ncct_intensity, ncct_splits, data_contract):
            report["provenance"]["source_sha256"][module.__name__] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        report["provenance"].update(nibabel=nibabel.__version__, numpy=numpy.__version__)
        stage = "private_write"
        write_private_qc(run, report)
        # Explicit allowlist: no private report, case measurements, IDs or paths.
        print(json.dumps(report["summary"], sort_keys=True))
        return 0 if report["summary"]["integrity_complete"] else 1
    except Exception:
        print(json.dumps({"stage": "5C2D", "status": "failed", "failure_stage": stage,
                          "anatomy_certified": False, "training_authorized": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
