"""Path configuration for explicit, local dataset checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_DATASET_ROOT = Path("data")
DEFAULT_RUNTIME_ROOT = Path("outputs")


@dataclass(frozen=True)
class ServerPaths:
    project_root: Path
    manifest: Path
    folds: Path
    dataset_root: Path
    runtime_root: Path

    @classmethod
    def from_overrides(
        cls,
        *,
        project_root: str | Path | None = None,
        manifest: str | Path | None = None,
        folds: str | Path | None = None,
        dataset_root: str | Path | None = None,
        runtime_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> ServerPaths:
        env = os.environ if environ is None else environ
        project = Path(
            project_root or env.get("ELE630_PROJECT") or Path.cwd()
        ).expanduser()
        tables = project / "outputs" / "tables"
        return cls(
            project_root=project,
            manifest=Path(
                manifest or env.get("ELE630_MANIFEST") or tables / "patient_manifest.csv"
            ).expanduser(),
            folds=Path(
                folds or env.get("ELE630_FOLDS") or tables / "outcome_cv_folds_private.csv"
            ).expanduser(),
            dataset_root=Path(
                dataset_root or env.get("DATA_ROOT") or DEFAULT_DATASET_ROOT
            ).expanduser(),
            runtime_root=Path(
                runtime_root or env.get("ELE630_RUNTIME") or DEFAULT_RUNTIME_ROOT
            ).expanduser(),
        )

    def check_server_locations(self, *, require_runtime: bool = True) -> None:
        for label, path in (
            ("project root", self.project_root),
            ("read-only dataset root", self.dataset_root),
        ):
            if not path.is_dir():
                raise ValueError(f"{label} is not a directory: {path}")
        if require_runtime and not self.runtime_root.is_dir():
            raise ValueError(f"runtime/output root is not a directory: {self.runtime_root}")
        for label, path in (("patient manifest", self.manifest), ("outcome folds", self.folds)):
            if not path.is_file():
                raise ValueError(f"{label} is not a file: {path}")
