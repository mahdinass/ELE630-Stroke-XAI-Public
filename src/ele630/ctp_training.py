"""Bounded fold-0 CTP development and one-patient smoke runner; no import I/O."""

from __future__ import annotations

import io
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from .ctp_continuation import (
    capture_rng, commit_epoch, load_config, load_latest, publish_json,
    run_lock, signature,
)
from .ctp_development import CTPDevelopmentData
from .ctp_model import CompactCTPMean
from .data_contract import DatasetContract
from .ncct_private_output import create_private_run, open_private_binary, open_private_run
from .ncct_training import _seed_everything


class CTPTrainingError(RuntimeError):
    """A stopped CTP run; messages never include patient or artifact details."""


@dataclass(frozen=True)
class CTPTrainingConfig:
    max_epochs: int = 10
    device: str = "cpu"
    patience: int = field(default=3, init=False)
    learning_rate: float = field(default=1e-4, init=False)
    weight_decay: float = field(default=1e-4, init=False)
    seed: int = field(default=630, init=False)

    def __post_init__(self) -> None:
        if type(self.max_epochs) is not int or not 1 <= self.max_epochs <= 10:
            raise ValueError("max_epochs must be an integer from 1 through 10")
        if type(self.device) is not str or self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class CTPEpoch:
    epoch: int
    training_bce: float
    validation_bce: float
    training_patients: int
    validation_patients: int


@dataclass(frozen=True)
class CTPRunResult:
    mode: str
    optimizer_steps: int
    epochs: tuple[CTPEpoch, ...]
    best_epoch: int | None
    best_validation_bce: float | None
    checkpoint_path: Path = field(repr=False)
    metrics_path: Path = field(repr=False)


def _write_json(path: Path, record: dict) -> None:
    with io.TextIOWrapper(open_private_binary(path), encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def _save_weights(path: Path, model: torch.nn.Module) -> None:
    with open_private_binary(path) as stream:
        torch.save(model.state_dict(), stream)


def _validation_bce(model: torch.nn.Module, data: CTPDevelopmentData, device: torch.device) -> float:
    """Complete one fixed inner-validation pass, without gradients or updates."""
    model.zero_grad(set_to_none=True)
    model.eval()
    losses = []
    with torch.no_grad():
        for patient in data.validation_patients:
            sample = data.open_validation(patient)
            logit = model(sample.series)
            target = torch.tensor([sample.target], dtype=torch.float32, device=device)
            if tuple(logit.shape) != (1,) or not torch.isfinite(logit).all().item():
                raise ValueError("CTP validation requires one finite logit")
            loss = F.binary_cross_entropy_with_logits(logit, target)
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite CTP validation loss")
            losses.append(float(loss.item()))
            del sample, logit, target, loss
    if not losses:
        raise ValueError("CTP validation requires an authorized inner-validation partition")
    return math.fsum(losses) / len(losses)


def _split_digest(data: CTPDevelopmentData) -> str:
    """Bind the run to exact inner membership and labels without publishing IDs."""
    partitions = (
        data.plan.inner_train, data.plan.inner_validation, data.plan.outer_test,
    )
    records = [
        [(p.patient_id, p.cohort, p.outcome, p.has_gt, p.fold) for p in part]
        for part in partitions
    ]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode("utf-8")).hexdigest()


def _run_configuration(contract: DatasetContract, data: CTPDevelopmentData,
                       config: CTPTrainingConfig, model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer) -> dict:
    cuda_environment = None
    if config.device == "cuda":
        cuda_environment = {
            "runtime": torch.version.cuda,
            "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "capabilities": [list(torch.cuda.get_device_capability(index))
                             for index in range(torch.cuda.device_count())],
        }
    return {
        "continuation_format": 1,
        "mode": "development", "outer_fold": 0, "outer_fingerprint": contract.fingerprint,
        "inner_partition_sha256": _split_digest(data),
        "training": asdict(config), "model": model.configuration,
        "optimizer": {"name": "AdamW", **optimizer.defaults},
        "loss": "unweighted_binary_cross_entropy_with_logits",
        "target_mapping": {"good": 0, "poor": 1}, "batch_size": 1,
        "inner_split_seed": 630, "inner_validation_fraction": 0.20,
        "inner_training_count": len(data.plan.inner_train),
        "inner_validation_count": len(data.plan.inner_validation),
        "outer_test_count": len(data.plan.outer_test),
        "overall_epoch_limit": config.max_epochs, "patience": config.patience,
        "training_order": "local_Random(seed_plus_epoch)_shuffle",
        "checkpoint_selection": "strict_minimum_inner_validation_bce",
        "ties": "keep_earlier_epoch", "resume_supported": True,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": str(torch.__version__), "cuda_environment": cuda_environment,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def run_ctp_development(
    contract: DatasetContract,
    repository,
    *,
    mode: str,
    private_output_dir: str | Path,
    run_id: str,
    config: CTPTrainingConfig = CTPTrainingConfig(),
    epochs_this_job: int = 1,
    resume: bool = False,
    model_factory: Callable[..., torch.nn.Module] | None = None,
) -> CTPRunResult:
    """Run bounded fold-0 epochs, resume a committed epoch, or smoke once.

    Real callers must first use load_and_validate with the active profile and
    supply RegisteredCTPRepository. A resume never uses best-checkpoint weights.
    Only fully committed epochs are eligible for continuation. Omit
    model_factory for the unchanged mean model; explicitly supply a model
    class for a separate signed architecture experiment.
    """
    if type(config) is not CTPTrainingConfig:
        raise TypeError("config must be CTPTrainingConfig")
    if type(epochs_this_job) is not int or not 1 <= epochs_this_job <= config.max_epochs:
        raise ValueError("epochs_this_job must be an integer from 1 through max_epochs")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if model_factory is not None and not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if mode == "smoke" and (resume or epochs_this_job != 1):
        raise ValueError("smoke mode permits one new patient update only")
    data = CTPDevelopmentData(contract, repository, mode=mode)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    paths = (open_private_run(private_output_dir, run_id, 0) if resume
             else create_private_run(private_output_dir, run_id, 0))
    model = None
    steps = 0
    committed_epochs = 0
    history: list[CTPEpoch] = []
    best_epoch = None
    best_bce = None
    checkpoint = None
    started = time.perf_counter()
    with run_lock(paths.fold_dir):
        try:
            _seed_everything(config.seed)
            device = torch.device(config.device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            factory = CompactCTPMean if model_factory is None else model_factory
            model = factory(width=16).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                betas=(0.9, 0.999), eps=1e-8, amsgrad=False, foreach=False,
            )
            if mode == "smoke":
                _write_json(paths.fold_dir / "run_config.json", {
                    "mode": mode, "outer_fold": 0, "outer_fingerprint": contract.fingerprint,
                    "training": asdict(config), "model": model.configuration,
                    "optimizer": {"name": "AdamW", **optimizer.defaults},
                    "loss": "unweighted_binary_cross_entropy_with_logits",
                    "target_mapping": {"good": 0, "poor": 1}, "batch_size": 1,
                    "inner_split_seed": 630, "inner_validation_fraction": 0.20,
                    "inner_training_count": len(data.plan.inner_train),
                    "inner_validation_count": len(data.plan.inner_validation),
                    "outer_test_count": len(data.plan.outer_test),
                    "effective_epoch_limit": 1,
                    "smoke_selection": "first_sorted_inner_train",
                    "checkpoint_selection": "none_smoke",
                    "ties": "keep_earlier_epoch", "resume_supported": False,
                    "torch_version": str(torch.__version__),
                    "deterministic_algorithms": True, "cudnn_benchmark": False,
                })
                first_epoch = 1
                stale = 0
                experiment_signature = None
                best = None
            else:
                expected = _run_configuration(contract, data, config, model, optimizer)
                experiment_signature = signature(expected)
                if resume:
                    load_config(paths.fold_dir / "run_config.json", expected)
                    state = load_latest(
                        paths.fold_dir, experiment_signature,
                        training_count=len(data.training_patients),
                        validation_count=len(data.validation_patients),
                        max_epochs=config.max_epochs, patience=config.patience,
                        model=model, optimizer=optimizer, device=device,
                    )
                    history = [CTPEpoch(**item) for item in state["history"]]
                    committed_epochs = state["completed_epoch"]
                    steps = state["optimizer_steps"]
                    best_epoch, best_bce = state["best_epoch"], state["best_validation_bce"]
                    stale = state["stale_epochs"]
                    best = state["best_checkpoint"]
                    checkpoint = paths.fold_dir / best["name"]
                    first_epoch = state["next_epoch"]
                else:
                    publish_json(paths.fold_dir, "run_config.json", expected)
                    first_epoch, stale, best = 1, 0, None

            last_epoch = (first_epoch - 1 if stale >= config.patience else
                          min(config.max_epochs, first_epoch + epochs_this_job - 1))
            for epoch in range(first_epoch, (1 if mode == "smoke" else last_epoch) + 1):
                model.train()
                order = list(data.training_patients)
                random.Random(config.seed + epoch).shuffle(order)
                losses = []
                for patient in order:
                    sample = data.open_training(patient)
                    target = torch.tensor([sample.target], dtype=torch.float32, device=device)
                    result = model.backward_series(sample.series, target)
                    # backward_series completes all 40+40 frame reads before this
                    # sole update. There is no optimizer access within either pass.
                    if (tuple(result.logit.shape) != (1,) or result.loss.ndim != 0
                            or not torch.isfinite(result.logit).all().item()
                            or not torch.isfinite(result.loss).item()):
                        raise ValueError("nonfinite CTP training result")
                    if any(p.grad is None or not torch.isfinite(p.grad).all().item()
                           for p in model.parameters()):
                        raise ValueError("missing or nonfinite CTP parameter gradient")
                    optimizer.step()
                    steps += 1
                    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                        raise ValueError("nonfinite CTP parameter after optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    if mode == "development":
                        losses.append(float(result.loss.item()))
                    del sample, target, result

                if mode == "smoke":
                    checkpoint = paths.fold_dir / "smoke_completed.pt"
                    _save_weights(checkpoint, model)
                    break  # No inner-validation access, loss report, or selection.

                validation = _validation_bce(model, data, device)
                training = math.fsum(losses) / len(losses)
                record = CTPEpoch(epoch, training, validation, len(order), len(data.validation_patients))
                improved = best_bce is None or validation < best_bce
                if improved:
                    best_epoch, best_bce, stale = epoch, validation, 0
                else:
                    stale += 1
                history.append(record)
                state = {
                    "version": 1, "experiment_signature": experiment_signature,
                    "completed_epoch": epoch, "next_epoch": epoch + 1,
                    "history": [asdict(item) for item in history],
                    "optimizer_steps": steps, "best_epoch": best_epoch,
                    "best_validation_bce": best_bce, "stale_epochs": stale,
                    "best_checkpoint": best,
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "rng": capture_rng(device),
                }
                best = commit_epoch(paths.fold_dir, epoch=epoch, record=asdict(record),
                                    selected=improved, state=state, model=model)
                committed_epochs = epoch
                checkpoint = paths.fold_dir / best["name"]
                if stale >= config.patience:
                    break

            if checkpoint is None:
                raise RuntimeError("CTP run produced no completed checkpoint")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            complete = mode == "smoke" or len(history) >= config.max_epochs or stale >= config.patience
            metrics_path = (paths.metrics_path if complete else
                            paths.fold_dir / f"job_epoch_{len(history):03d}.json")
            metrics = {
                "status": "complete" if complete else "paused", "mode": mode, "outer_fold": 0,
                "job_epoch_limit": epochs_this_job, "job_start_epoch": first_epoch,
                "next_epoch": None if complete else len(history) + 1,
                "resumed": resume,
                "optimizer_steps": steps, "completed_epochs": len(history),
                "training_patient_visits": steps,
                "validation_patient_visits": len(history) * len(data.validation_patients),
                "best_epoch": best_epoch, "best_validation_bce": best_bce,
                "epochs": [asdict(item) for item in history],
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            }
            if mode == "smoke":
                _write_json(metrics_path, metrics)
            else:
                if metrics_path.exists():
                    if not (resume and complete):
                        raise RuntimeError("CTP completion artifact already exists")
                    if metrics_path.is_symlink():
                        raise RuntimeError("CTP completion artifact is unsafe")
                    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if (type(saved_metrics) is not dict or saved_metrics.get("status") != "complete"
                            or saved_metrics.get("completed_epochs") != len(history)
                            or saved_metrics.get("optimizer_steps") != steps
                            or saved_metrics.get("best_epoch") != best_epoch
                            or saved_metrics.get("best_validation_bce") != best_bce):
                        raise RuntimeError("CTP completion artifact disagrees with committed state")
                else:
                    publish_json(paths.fold_dir, metrics_path.name, metrics)
            return CTPRunResult(mode, steps, tuple(history), best_epoch, best_bce, checkpoint, metrics_path)
        except Exception:
            if model is not None:
                model.zero_grad(set_to_none=True)
            try:
                failure_name = "failed.json" if mode == "smoke" else f"failed_job_{time.time_ns()}.json"
                _write_json(paths.fold_dir / failure_name, {
                    "status": "failed", "optimizer_steps_attempted": steps,
                    "completed_epochs": committed_epochs,
                    "committed_optimizer_steps": committed_epochs * len(data.plan.inner_train),
                    "resume_from_last_commit_only": mode == "development",
                })
            except Exception:
                pass  # Disk failures may prevent even a failure marker.
            raise CTPTrainingError("CTP job stopped; inspect its private committed state before continuation") from None
