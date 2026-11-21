"""Read-only preflight for the server-only active outcome cohort.

Run from the repository root: python scripts/preflight/validate_active_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct execution without an editable installation or extra dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ele630.config import ServerPaths  # noqa: E402
from ele630.data_contract import ContractError, load_and_validate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, help="UiS code/project directory")
    parser.add_argument("--manifest", type=Path, help="server-side patient_manifest.csv")
    parser.add_argument("--folds", type=Path, help="server-side active outcome_cv_folds_private.csv")
    parser.add_argument("--dataset-root", type=Path, help="read-only curated dataset directory")
    parser.add_argument("--runtime-root", type=Path, help="local runtime/output directory")
    args = parser.parse_args(argv)
    paths = ServerPaths.from_overrides(**vars(args))
    try:
        paths.check_server_locations()
        contract = load_and_validate(paths.manifest, paths.folds)
    except (ValueError, ContractError) as exc:
        print("ELE630 active dataset preflight", file=sys.stderr)
        print("Status: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1

    print("ELE630 active dataset preflight")
    print("-------------------------------")
    print(f"Eligible outcome patients: {len(contract.patients)}")
    print(f"Good / poor: {contract.good} / {contract.poor}")
    print(f"Patients with GT: {contract.with_gt}")
    print(f"Folds: {len(contract.fold_summaries)} (seed {contract.seed})")
    print("Fold sizes: " + " / ".join(str(f.patients) for f in contract.fold_summaries))
    print("Poor per fold: " + " / ".join(str(f.poor) for f in contract.fold_summaries))
    for fold in contract.fold_summaries:
        cohorts = ", ".join(f"{cohort}={count}" for cohort, count in fold.cohorts)
        print(f"Fold {fold.fold}: good={fold.good}, poor={fold.poor}, cohorts: {cohorts}")
    print(f"Fingerprint: {contract.fingerprint}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
