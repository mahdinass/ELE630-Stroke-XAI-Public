#!/usr/bin/env python3
"""Fold-0 NCCT development comparison; default prints a data-free plan only.

--execute-development is an explicit execution boundary for a subsequent UiS
GPU allocation. This script has no outer-test evaluation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ele630.config import ServerPaths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-development", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--private-output-dir", type=Path)
    parser.add_argument("--run-prefix")
    parser.add_argument("--source-revision")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)
    if not args.execute_development:
        print(json.dumps({
            "mode": "plan-only", "patient_data_access": False,
            "outer_test_evaluation": False, "outer_fold": 0,
            "arms": [{"pooling": "avg", "stage4_stride": [n, n, n],
                      "max_epochs": 10, "patience": 3, "seed": 630,
                      "diagnostic_epoch": 3} for n in (2, 1)],
        }, sort_keys=True))
        return 0
    # Reject absent execution/provenance resources before reading data files.
    if (os.name != "posix" or not os.environ.get("SLURM_JOB_ID")
            or any(value is None for value in (args.private_output_dir, args.run_prefix,
                                               args.source_revision, args.protocol))):
        print("NCCT execution requires a Slurm worker and registered private run/provenance", file=sys.stderr)
        return 2
    try:
        import torch
        from ele630.data_contract import load_and_validate
        from ele630.ncct import NCCTRepository
        from ele630.ncct_development import DevelopmentNCCTSamples, evaluate_selected_development
        from ele630.ncct_model_config import record_signature
        from ele630.ncct_private_output import _RUN_ID, _validate_base, open_private_binary
        from ele630.ncct_samples import LazyNCCTSamples
        from ele630.ncct_splits import plan_outer_fold
        from ele630.ncct_training import TrainingConfig, run_one_fold
        if not torch.cuda.is_available():
            raise ValueError("allocated CUDA device unavailable")
        if (len(args.source_revision) != 40
                or any(c not in "0123456789abcdef" for c in args.source_revision)):
            raise ValueError("registered source revision required")
        current_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.project_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if args.source_revision != current_revision:
            raise ValueError("registered source revision disagrees with checkout")
        if not _RUN_ID.fullmatch(args.run_prefix) or len(args.run_prefix) > 50:
            raise ValueError("non-identifying run prefix required")
        private_base = _validate_base(args.private_output_dir)
        run_names = [f"{args.run_prefix}_s{stride}" for stride in (2, 1)]
        if any((private_base / name).exists() or (private_base / name).is_symlink() for name in run_names):
            raise ValueError("paired run names must both be new")
        summary_path = private_base / f"{args.run_prefix}_comparison.json"
        if summary_path.exists() or summary_path.is_symlink():
            raise ValueError("paired summary must be new")
        protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        paths = ServerPaths.from_overrides(project_root=args.project_root, manifest=args.manifest,
                                           folds=args.folds, dataset_root=args.dataset_root)
        paths.check_server_locations()
        # Contract metadata validation is not image processing or test scoring.
        contract = load_and_validate(paths.manifest, paths.folds)
        plan = plan_outer_fold(contract, 0, seed=630, validation_fraction=.20)
        repository = NCCTRepository(contract, paths.dataset_root)
        samples = DevelopmentNCCTSamples(plan, LazyNCCTSamples(contract, repository, lower=-20, upper=120))
        results = []
        # Run both fits before inspecting comparative validation results.
        for stride, run_name in zip((2, 1), run_names):
            results.append(run_one_fold(
                plan, samples, private_output_dir=private_base, run_id=run_name,
                config=TrainingConfig(device="cuda", stage4_stride=(stride,) * 3, diagnostic_epoch=3),
                source_revision=args.source_revision, protocol_sha256=protocol_digest,
                require_complete_provenance=True,
                expected_initial_parameters_sha256=(results[0].run_record["initial_parameters_sha256"]
                                                     if results else None),
            ))
        if results[0].run_record["initial_parameters_sha256"] != results[1].run_record["initial_parameters_sha256"]:
            raise ValueError("paired initialization differs")
        scores = [evaluate_selected_development(plan, samples, result, device="cuda") for result in results]
        summary = {
            "schema_version": 1, "outer_test_evaluated": False,
            "protocol_sha256": protocol_digest,
            "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "development_input_consistency_signature": samples.input_consistency_signature,
            "arms": [{"stage4_stride": [stride] * 3,
                      "run_signature": record_signature(result.run_record),
                      "diagnostics": list(result.diagnostics), "scores": score}
                     for stride, result, score in zip((2, 1), results, scores)],
            "challenger_minus_control_bce": scores[1]["model"]["bce"] - scores[0]["model"]["bce"],
            "interpretation": "reused-development-validation; no independent generalization claim",
        }
        with open_private_binary(summary_path) as stream:
            stream.write(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False).encode("utf-8"))
        print(json.dumps({"status": "completed", "development_fits": 2,
                          "outer_test_evaluated": False, "summary": "private aggregate record written"}))
        return 0
    except Exception:
        print("NCCT development failed; retain private artifacts for diagnosis", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
