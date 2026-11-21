"""Guarded Stage 5B runner: use only on an approved UiS Slurm cpu20 worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ele630.clinical_runner import execute_clinical_runner
from ele630.config import ServerPaths


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes malformed argument contents, possibly paths.
        self.exit(2, "Stage 5B: invalid command-line arguments\n")


def main(argv: list[str] | None = None) -> int:
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--execute-real", action="store_true", help="explicitly authorize this server run")
    for name in ("project-root", "manifest", "folds", "dataset-root", "runtime-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--clinical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="new non-identifying run slug")
    args = parser.parse_args(argv)
    if not args.execute_real:
        print("Stage 5B: explicit --execute-real acknowledgement required", file=sys.stderr)
        return 2
    try:
        paths = ServerPaths.from_overrides(
            project_root=args.project_root, manifest=args.manifest, folds=args.folds,
            dataset_root=args.dataset_root, runtime_root=args.runtime_root,
        )
        record = execute_clinical_runner(paths, args.clinical_csv, args.output_dir,
                                         args.run_id, execute_real=True)
        print(json.dumps(record, sort_keys=True, allow_nan=False))
        return 0
    except Exception:
        # Do not dump exceptions or traceback locals at the real-data boundary.
        print("Stage 5B: FAIL; execution stopped without patient-level diagnostics", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
