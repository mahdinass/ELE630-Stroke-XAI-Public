"""Private epoch commits for bounded CTP development; no patient file access."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .ncct_private_output import PrivateOutputError, open_private_binary


class CTPContinuationError(ValueError):
    """A continuation artifact is absent, unsafe or incompatible."""


_COMMIT = re.compile(r"commit_epoch_(\d{3})_([0-9a-f]{16})\.json\Z")
_STATE_VERSION = 1


def _digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def signature(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise CTPContinuationError("private continuation artifact is missing or unsafe")
    if os.name == "posix":
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise CTPContinuationError("private continuation artifact must be owner-only")


def _write_json(path: Path, value: dict) -> None:
    with open_private_binary(path) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def publish_json(directory: Path, name: str, value: dict) -> Path:
    """Write, sync, then publish a uniquely named JSON file in one rename."""
    destination = directory / name
    if destination.exists() or destination.is_symlink():
        raise CTPContinuationError("private completion artifact already exists")
    pending = directory / f"pending_{secrets.token_hex(16)}.json"
    _write_json(pending, value)
    # A lock serializes cooperating jobs. The destination is new and never
    # replaced by this runner; an interrupted pending file is never consulted.
    if destination.exists() or destination.is_symlink():
        raise CTPContinuationError("private completion artifact already exists")
    os.rename(pending, destination)
    return destination


def save_torch(path: Path, value: object) -> str:
    with open_private_binary(path) as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return _digest(path)


@contextmanager
def run_lock(fold_dir: Path):
    """A kernel lock releases on process termination; the file stays private."""
    path = fold_dir / ".continuation.lock"
    if path.is_symlink():
        raise CTPContinuationError("private continuation lock is unsafe")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        raise CTPContinuationError("private continuation lock is unavailable") from None
    locked = False
    try:
        if os.name == "posix":
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or not stat.S_ISREG(info.st_mode):
                raise CTPContinuationError("private continuation lock must be owner-only")
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        locked = True
        yield
    except (BlockingIOError, OSError):
        raise CTPContinuationError("private continuation run is locked or unavailable") from None
    finally:
        if locked:
            if os.name == "posix":
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            else:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


def load_config(path: Path, expected: dict) -> str:
    try:
        _private_file(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise CTPContinuationError("private run configuration is incomplete") from None
    if signature(saved) != signature(expected):
        raise CTPContinuationError("private run configuration is incompatible")
    return signature(expected)


def _numpy_rng() -> tuple:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return name, keys.tolist(), position, has_gauss, cached


def capture_rng(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng(value: dict, device: torch.device) -> None:
    if type(value) is not dict or set(value) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise CTPContinuationError("private RNG state is incomplete")
    cpu = value["torch_cpu"]
    cuda = value["torch_cuda"]
    if (not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8
            or tuple(cpu.shape) != tuple(torch.get_rng_state().shape)):
        raise CTPContinuationError("private CPU RNG state is invalid")
    if device.type == "cuda":
        current = torch.cuda.get_rng_state_all()
        if (type(cuda) is not list or len(cuda) != len(current)
                or any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8
                       or tuple(item.shape) != tuple(reference.shape)
                       for item, reference in zip(cuda, current))):
            raise CTPContinuationError("private CUDA RNG state is incompatible")
    elif cuda is not None:
        raise CTPContinuationError("private CUDA RNG state disagrees with device")
    try:
        name, keys, position, has_gauss, cached = value["numpy"]
        if name != "MT19937" or type(keys) is not list or len(keys) != 624:
            raise ValueError("invalid NumPy RNG state")
        numpy_state = (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
        # Validate both language RNG records before setting the torch states.
        probe = random.Random()
        probe.setstate(value["python"])
        np.random.set_state(numpy_state)
        random.setstate(value["python"])
        torch.set_rng_state(cpu)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda)
    except (TypeError, ValueError, RuntimeError):
        raise CTPContinuationError("private RNG state cannot be restored") from None


def commit_epoch(
    fold_dir: Path, *, epoch: int, record: dict, selected: bool,
    state: dict, model: torch.nn.Module,
) -> dict:
    """Publish a complete epoch only after separate weights/state are synced."""
    token = secrets.token_hex(8)
    best = state["best_checkpoint"]
    if selected:
        canonical = fold_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        checkpoint = canonical if not canonical.exists() else fold_dir / f"checkpoint_epoch_{epoch:03d}_{token}.pt"
        best = {"name": checkpoint.name, "sha256": save_torch(checkpoint, model.state_dict())}
        state["best_checkpoint"] = best
    state_path = fold_dir / f"state_epoch_{epoch:03d}_{token}.pt"
    state_digest = save_torch(state_path, state)
    marker = {
        "version": _STATE_VERSION, "epoch": epoch, "record": record,
        "checkpoint_selected": selected, "state_file": state_path.name,
        "state_sha256": state_digest, "best_checkpoint": best,
    }
    publish_json(fold_dir, f"commit_epoch_{epoch:03d}_{token}.json", marker)
    return best


def read_committed_state(fold_dir: Path, *, selected_epoch: int | None = None) -> tuple:
    """Read a checksummed Stage 4R state without restoring training state.

    The resume loader and read-only evaluation share this weights-only decoder.
    An absent selected_epoch reads the latest complete commit.
    """
    commits = []
    for path in fold_dir.glob("commit_epoch_*.json"):
        match = _COMMIT.fullmatch(path.name)
        if match is None:
            raise CTPContinuationError("private epoch commit name is invalid")
        _private_file(path)
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            raise CTPContinuationError("private epoch commit is incomplete") from None
        epoch, token = int(match[1]), match[2]
        if (type(marker) is not dict or marker.get("version") != _STATE_VERSION
                or type(marker.get("epoch")) is not int or marker["epoch"] != epoch
                or marker.get("state_file") != f"state_epoch_{epoch:03d}_{token}.pt"
                or type(marker.get("state_sha256")) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", marker["state_sha256"])):
            raise CTPContinuationError("private epoch commit is invalid")
        state_file = fold_dir / marker["state_file"]
        _private_file(state_file)
        if _digest(state_file) != marker["state_sha256"]:
            raise CTPContinuationError("private epoch state checksum disagrees")
        commits.append((epoch, marker, state_file))
    commits.sort(key=lambda item: item[0])
    if not commits or [epoch for epoch, _, _ in commits] != list(range(1, len(commits) + 1)):
        raise CTPContinuationError("private epoch history is absent or incomplete")
    if selected_epoch is not None and (type(selected_epoch) is not int
                                       or not 1 <= selected_epoch <= len(commits)):
        raise CTPContinuationError("private selected epoch is invalid")
    epoch, marker, state_file = commits[-1 if selected_epoch is None else selected_epoch - 1]
    try:
        state = torch.load(state_file, map_location="cpu", weights_only=True)
    except Exception:
        raise CTPContinuationError("private epoch state cannot be decoded") from None
    return epoch, marker, state, commits


def load_latest(fold_dir: Path, experiment_signature: str, *,
                training_count: int, validation_count: int, max_epochs: int,
                patience: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device) -> dict:
    """Reject broken history before image I/O; ignore uncommitted attempt files."""
    epoch, marker, state, _ = read_committed_state(fold_dir)
    if (type(state) is not dict or state.get("version") != _STATE_VERSION
            or state.get("experiment_signature") != experiment_signature
            or type(state.get("completed_epoch")) is not int or state["completed_epoch"] != epoch
            or type(state.get("next_epoch")) is not int or state["next_epoch"] != epoch + 1
            or epoch > max_epochs or type(state.get("optimizer_steps")) is not int
            or state["optimizer_steps"] != epoch * training_count
            or type(state.get("history")) is not list or len(state["history"]) != epoch):
        raise CTPContinuationError("private epoch state is incompatible")
    history = state["history"]
    best_epoch, best_bce, stale = None, None, 0
    for index, item in enumerate(history, start=1):
        if (type(item) is not dict or type(item.get("epoch")) is not int or item["epoch"] != index
                or type(item.get("training_patients")) is not int or item["training_patients"] != training_count
                or type(item.get("validation_patients")) is not int or item["validation_patients"] != validation_count
                or any(type(item.get(key)) not in (int, float) or not math.isfinite(item[key])
                       for key in ("training_bce", "validation_bce"))):
            raise CTPContinuationError("private epoch history is invalid")
        if best_bce is None or item["validation_bce"] < best_bce:
            best_epoch, best_bce, stale = index, item["validation_bce"], 0
        else:
            stale += 1
        if stale >= patience and index != epoch:
            raise CTPContinuationError("private history continues after early stopping")
    if (state.get("best_epoch") != best_epoch or state.get("best_validation_bce") != best_bce
            or type(state.get("stale_epochs")) is not int or state["stale_epochs"] != stale
            or marker.get("record") != history[-1]
            or type(marker.get("checkpoint_selected")) is not bool
            or marker["checkpoint_selected"] != (best_epoch == epoch)
            or marker.get("best_checkpoint") != state.get("best_checkpoint")):
        raise CTPContinuationError("private epoch selection state is invalid")
    best = state["best_checkpoint"]
    if (type(best) is not dict or type(best.get("name")) is not str
            or not re.fullmatch(rf"checkpoint_epoch_{best_epoch:03d}(?:_[0-9a-f]{{16}})?\.pt", best["name"])
            or type(best.get("sha256")) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", best["sha256"])):
        raise CTPContinuationError("private best checkpoint reference is invalid")
    checkpoint = fold_dir / best["name"]
    _private_file(checkpoint)
    if _digest(checkpoint) != best["sha256"]:
        raise CTPContinuationError("private best checkpoint checksum disagrees")
    try:
        loaded_optimizer = state["optimizer_state"]
        if (type(loaded_optimizer) is not dict
                or loaded_optimizer.get("param_groups") != optimizer.state_dict()["param_groups"]):
            raise ValueError("optimizer configuration changed")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(loaded_optimizer)
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("invalid model parameter")
        if len(optimizer.state) != len(list(model.parameters())):
            raise ValueError("incomplete optimizer state")
        for parameter, values in optimizer.state.items():
            if (not isinstance(values, dict) or int(values.get("step", -1)) != state["optimizer_steps"]
                    or not {"step", "exp_avg", "exp_avg_sq"}.issubset(values)
                    or any(values[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq"))
                    or any(not torch.isfinite(value).all().item()
                           for value in values.values() if isinstance(value, torch.Tensor))):
                raise ValueError("invalid optimizer state")
        restore_rng(state["rng"], device)
    except (KeyError, TypeError, ValueError, RuntimeError):
        raise CTPContinuationError("private model, optimizer or RNG state is invalid") from None
    return state
