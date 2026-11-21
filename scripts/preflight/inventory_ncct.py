"""Aggregate admission-NCCT inventory for the fixed outcome cohort.

Use --headers-only for lightweight login-node geometry checks. Full mode reads
all 143 voxel arrays sequentially and belongs on an approved compute host.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ele630.config import ServerPaths  # noqa: E402
from ele630.data_contract import load_and_validate  # noqa: E402
def _range_by_axis(values: list[tuple[int | float, int | float, int | float]]) -> str:
    return ", ".join(
        f"{axis}={min(row[index] for row in values):g}..{max(row[index] for row in values):g}"
        for index, axis in enumerate("xyz")
    )


def _private_output_path(requested: Path, paths: ServerPaths) -> Path:
    target = requested.expanduser().resolve()
    allowed = (
        (paths.project_root / "outputs" / "tables").resolve(),
        paths.runtime_root.resolve(),
    )
    if target.suffix.lower() != ".csv" or not any(target.is_relative_to(root) for root in allowed):
        raise ValueError(
            "--details-csv must be a CSV under project outputs/tables or the runtime root"
        )
    return target


def _write_details(path: Path, inventory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation and 0600 permissions keep patient-level diagnostics private.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("patient_id", "status", "shape", "orientation"))
        for case in inventory.cases:
            writer.writerow((
                case.patient_id,
                case.status,
                "x".join(map(str, case.shape)) if case.shape else "",
                case.orientation or "",
            ))


def _print_summary(inventory) -> None:
    counts = inventory.counts
    unresolved = counts["missing"] + counts["ambiguous"] + counts["ineligible"]
    print("ELE630 active NCCT inventory")
    print("---------------------------")
    print(f"Mode: {'headers only' if inventory.headers_only else 'full voxel sanity'}")
    print(f"Eligible outcome patients: {inventory.total}")
    print(f"Resolved NCCT: {inventory.total - unresolved} / {inventory.total}")
    print(f"Missing / ambiguous paths: {counts['missing']} / {counts['ambiguous']}")
    print(f"NIfTI opened / unreadable: {inventory.total - unresolved - counts['unreadable']} / {counts['unreadable']}")
    print(f"Invalid dimensionality / metadata: {counts['dimension']} / {counts['metadata']}")
    if inventory.headers_only:
        print("Finite voxels and intensities: not checked in header-only mode")
    else:
        print(f"Finite / non-finite volumes: {counts['ok']} / {counts['nonfinite']}")
    if inventory.shapes:
        print("Shape ranges: " + _range_by_axis(list(inventory.shapes)))
        common = inventory.shapes.most_common(3)
        print("Common shapes: " + ", ".join(
            f"{'x'.join(map(str, shape))} ({count})" for shape, count in common
        ))
        print("Orientations: " + ", ".join(
            f"{code}={count}" for code, count in sorted(inventory.orientations.items())
        ))
        print("Voxel spacing mm: " + _range_by_axis(inventory.spacings))
        print("Center-to-center physical span mm: " + _range_by_axis(inventory.extents))
    if inventory.minima:
        print(f"Observed voxel intensity min / max: {min(inventory.minima):g} / {max(inventory.maxima):g}")
    print(f"Status: {'PASS' if inventory.passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--headers-only", action="store_true")
    parser.add_argument(
        "--details-csv", type=Path,
        help="optional private case table under ignored outputs/tables or runtime",
    )
    args = parser.parse_args(argv)
    paths = ServerPaths.from_overrides(
        project_root=args.project_root, manifest=args.manifest, folds=args.folds,
        dataset_root=args.dataset_root, runtime_root=args.runtime_root,
    )
    try:
        paths.check_server_locations(require_runtime=not args.headers_only)
        details = _private_output_path(args.details_csv, paths) if args.details_csv else None
        contract = load_and_validate(paths.manifest, paths.folds)
        if args.headers_only:
            from ele630.ncct_headers import scan_ncct_headers

            inventory = scan_ncct_headers(contract, paths.dataset_root)
        else:
            from ele630.ncct import scan_ncct

            inventory = scan_ncct(contract, paths.dataset_root)
        if details is not None:
            _write_details(details, inventory)
    except (OSError, ValueError) as exc:
        print("ELE630 active NCCT inventory", file=sys.stderr)
        print("Status: FAIL", file=sys.stderr)
        # Contract/path errors have no patient identifiers by design.
        print(f"Reason: {exc}", file=sys.stderr)
        return 1
    _print_summary(inventory)
    return 0 if inventory.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
