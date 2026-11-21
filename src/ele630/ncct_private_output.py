"""Private, exclusive output locations for one UiS NCCT training run.

This module is importable without PyTorch or patient files. The two fixed UiS
roots are replaced only by synthetic tests; the public API has no root override.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_UNCONFIGURED_RUNTIME_ROOT = Path("/ele630-restricted-archive/runtime")
_APPROVED_PRIVATE_ROOT = _UNCONFIGURED_RUNTIME_ROOT
_TRUSTED_CANONICAL_ROOT = _UNCONFIGURED_RUNTIME_ROOT
_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")


class PrivateOutputError(ValueError):
    """The requested private destination or exclusive artifact is unsafe."""


@dataclass(frozen=True)
class PrivateRunPaths:
    run_dir: Path
    fold_dir: Path
    metrics_path: Path


def _validate_base(output_dir: str | Path) -> Path:
    try:
        raw = Path(output_dir)
    except (TypeError, ValueError):
        raise PrivateOutputError("an absolute private output base is required") from None
    if not raw.is_absolute() or ".." in raw.parts:
        raise PrivateOutputError("an absolute private output base without parent traversal is required")
    if _APPROVED_PRIVATE_ROOT == _UNCONFIGURED_RUNTIME_ROOT:
        raise PrivateOutputError("private artifact output requires the authorized research archive")
    try:
        canonical = _TRUSTED_CANONICAL_ROOT.resolve(strict=True)
        approved = _APPROVED_PRIVATE_ROOT.resolve(strict=True)
        if canonical != _TRUSTED_CANONICAL_ROOT or approved != canonical:
            raise PrivateOutputError("approved runtime alias does not match the trusted canonical root")
        if any(component.is_symlink() for component in
               (_TRUSTED_CANONICAL_ROOT, *_TRUSTED_CANONICAL_ROOT.parents)):
            raise PrivateOutputError("trusted canonical runtime path must have no symlinks")
        if not canonical.is_dir():
            raise PrivateOutputError("trusted canonical runtime root must be an existing directory")
        if os.name == "posix":
            info = canonical.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise PrivateOutputError("trusted canonical runtime root must be owner-only")

        # Only the two exact approved spellings are accepted. The storage alias
        # may contain site-managed symlinks up to its runtime root; every
        # caller-supplied component below either root must be symlink-free.
        if raw != _APPROVED_PRIVATE_ROOT and raw.is_relative_to(_APPROVED_PRIVATE_ROOT):
            prefix = _APPROVED_PRIVATE_ROOT
        elif raw != _TRUSTED_CANONICAL_ROOT and raw.is_relative_to(_TRUSTED_CANONICAL_ROOT):
            prefix = _TRUSTED_CANONICAL_ROOT
        else:
            raise PrivateOutputError("private output base must be below an approved UiS runtime root")
        current = prefix
        for component in raw.relative_to(prefix).parts:
            current = current / component
            if current.is_symlink():
                raise PrivateOutputError("symlinks below the trusted runtime root are not allowed")
        base = raw.resolve(strict=True)
        if not base.is_dir():
            raise PrivateOutputError("private output base must be an existing directory")
        if base == canonical or not base.is_relative_to(canonical):
            raise PrivateOutputError("private output base must be below the approved UiS runtime root")
        if os.name == "posix":
            info = base.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise PrivateOutputError("private output base must be owner-only (mode 0700)")
    except PrivateOutputError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise PrivateOutputError("private output base cannot be safely resolved") from None
    return base


def create_private_run(output_dir: str | Path, run_id: str, outer_fold: int) -> PrivateRunPaths:
    """Create a new, owner-only run/fold directory; never reuse an existing run."""
    if type(run_id) is not str or not _RUN_ID.fullmatch(run_id):
        raise PrivateOutputError("run_id must be a 1–64 character non-identifying slug")
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise PrivateOutputError("outer_fold must be an integer from 0 to 4")
    base = _validate_base(output_dir)
    run_dir = base / run_id
    fold_dir = run_dir / f"outer_fold_{outer_fold}"
    try:
        run_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        raise PrivateOutputError("run directory already exists; choose a new run_id") from None
    except OSError:
        raise PrivateOutputError("private run directory could not be created") from None
    try:
        os.chmod(run_dir, 0o700)
        fold_dir.mkdir(mode=0o700, exist_ok=False)
        os.chmod(fold_dir, 0o700)
        if not fold_dir.resolve(strict=True).is_relative_to(base):
            raise PrivateOutputError("created run resolved outside the private base")
    except PrivateOutputError:
        raise
    except (OSError, RuntimeError):
        raise PrivateOutputError("private fold directory could not be secured") from None
    return PrivateRunPaths(run_dir, fold_dir, fold_dir / "aggregate_training.json")


def open_private_run(output_dir: str | Path, run_id: str, outer_fold: int) -> PrivateRunPaths:
    """Reopen only an existing, owner-only run within the approved private base."""
    if type(run_id) is not str or not _RUN_ID.fullmatch(run_id):
        raise PrivateOutputError("run_id must be a 1–64 character non-identifying slug")
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise PrivateOutputError("outer_fold must be an integer from 0 to 4")
    base = _validate_base(output_dir)
    run_dir = base / run_id
    fold_dir = run_dir / f"outer_fold_{outer_fold}"
    try:
        for directory in (run_dir, fold_dir):
            if directory.is_symlink() or not directory.is_dir():
                raise PrivateOutputError("private continuation directory is missing or unsafe")
            if not directory.resolve(strict=True).is_relative_to(base):
                raise PrivateOutputError("private continuation directory leaves its approved base")
            if os.name == "posix":
                info = directory.stat()
                if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                    raise PrivateOutputError("private continuation directory must be owner-only (mode 0700)")
    except PrivateOutputError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise PrivateOutputError("private continuation directory cannot be safely opened") from None
    return PrivateRunPaths(run_dir, fold_dir, fold_dir / "aggregate_training.json")


def open_private_binary(path: Path) -> BinaryIO:
    """Open a new mode-0600 artifact without following or replacing a file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            else:
                os.chmod(path, 0o600)
            return os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
    except OSError:
        raise PrivateOutputError("private artifact could not be created exclusively") from None
