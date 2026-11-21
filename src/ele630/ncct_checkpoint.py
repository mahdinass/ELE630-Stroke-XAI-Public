"""Configuration-bound, exclusive NCCT checkpoints. No access at import time.

Historical weights-only files remain untouched. They are deliberately rejected
by this new loader unless an independently registered manifest exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from .ncct_model_config import record_signature
from .ncct_private_output import open_private_binary


class NCCTCheckpointError(ValueError):
    """Missing, incomplete, unsafe or incompatible checkpoint; sanitized message."""


def manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")


def _check_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise NCCTCheckpointError("NCCT checkpoint or manifest is missing or unsafe")
    if os.name == "posix":
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise NCCTCheckpointError("NCCT checkpoint artifacts must be owner-only")


def save_ncct_checkpoint(model, checkpoint: Path, *, run_record: dict, epoch: int) -> dict:
    """Write weights then completion manifest; partial writes cannot be loaded."""
    import torch

    if type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    architecture = model.architecture_record()
    signature = record_signature(run_record)
    if run_record.get("architecture") != architecture:
        raise NCCTCheckpointError("NCCT run architecture disagrees with the model")
    # Never write weights if a completion manifest already occupies this name.
    sidecar = manifest_path(checkpoint)
    if sidecar.exists() or sidecar.is_symlink():
        raise NCCTCheckpointError("NCCT checkpoint manifest already exists")
    try:
        with open_private_binary(checkpoint) as stream:
            torch.save(model.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_digest(stream)
        record = {
            "schema_version": 1, "epoch": epoch, "checkpoint_name": checkpoint.name,
            "sha256": digest, "run_signature": signature, "run_record": run_record,
        }
        with open_private_binary(sidecar) as stream:
            stream.write(json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        raise NCCTCheckpointError("NCCT checkpoint could not be published completely") from None
    return record


def _stream_digest(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_ncct_checkpoint(model, checkpoint: Path, *, expected_run_record: dict,
                         expected_epoch: int, map_location="cpu") -> dict:
    """Validate full configuration, epoch and bytes before weights_only loading.

    The expected record must come from the caller's registered experiment,
    not be blindly accepted from the checkpoint sidecar itself. Digests do not
    defend against a malicious owner rewriting both artifacts.
    """
    import torch

    if type(expected_epoch) is not int or expected_epoch < 1:
        raise ValueError("expected_epoch must be a positive integer")
    try:
        _check_file(checkpoint)
        sidecar = manifest_path(checkpoint)
        _check_file(sidecar)
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_signature = record_signature(expected_run_record)
        if (type(saved) is not dict
                or set(saved) != {"schema_version", "epoch", "checkpoint_name", "sha256", "run_signature", "run_record"}
                or type(saved["schema_version"]) is not int or saved["schema_version"] != 1
                or type(saved["epoch"]) is not int or saved["epoch"] != expected_epoch
                or saved["checkpoint_name"] != checkpoint.name
                or saved["run_signature"] != expected_signature
                or record_signature(saved["run_record"]) != expected_signature
                or expected_run_record.get("architecture") != model.architecture_record()):
            raise NCCTCheckpointError("NCCT checkpoint configuration or epoch is incompatible")
        # Hash and deserialize through the same open file handle.
        with checkpoint.open("rb") as stream:
            if _stream_digest(stream) != saved["sha256"]:
                raise NCCTCheckpointError("NCCT checkpoint byte digest disagrees")
            stream.seek(0)
            state = torch.load(stream, map_location=map_location, weights_only=True)
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise NCCTCheckpointError("NCCT checkpoint weight keys disagree")
        if any(not isinstance(state[key], torch.Tensor)
               or state[key].shape != expected[key].shape
               or state[key].dtype != expected[key].dtype
               or not torch.isfinite(state[key]).all().item() for key in expected):
            raise NCCTCheckpointError("NCCT checkpoint weights are invalid")
        model.load_state_dict(state, strict=True)
        return saved
    except NCCTCheckpointError:
        raise
    except Exception:
        raise NCCTCheckpointError("NCCT checkpoint could not be safely loaded") from None
