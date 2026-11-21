"""One-fold, NCCT-only, batch-one training; no file access at import time."""

from __future__ import annotations

import io
import json
import math
import os
import random
import hashlib
import sys
from importlib import metadata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from .ncct_samples import NCCTSample
from .data_contract import OutcomePatient
from .ncct_private_output import PrivateOutputError, create_private_run, open_private_binary
from .ncct_splits import (
    FoldPlan, binary_outcome, inner_train_positive_weight, validate_fold_plan,
)
from .resnet3d import CompactResNet10
from .ncct_model_config import record_signature, validate_stage4_stride
from .ncct_checkpoint import save_ncct_checkpoint


class SampleSource(Protocol):
    def load(self, patient: object) -> NCCTSample: ...


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 10
    patience: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    positive_weighting: bool = False
    seed: int = 630
    device: str = "cpu"
    pooling: str = "avg"
    stage4_stride: tuple[int, int, int] = (2, 2, 2)
    diagnostic_epoch: int | None = None

    def __post_init__(self) -> None:
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("patience must be a positive integer")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) for value in (self.learning_rate, self.weight_decay)):
            raise ValueError("optimizer parameters must be finite numbers")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if type(self.positive_weighting) is not bool:
            raise ValueError("positive_weighting must be boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if type(self.pooling) is not str or self.pooling not in {"avg", "avgmax"}:
            raise ValueError("pooling must be 'avg' or 'avgmax'")
        validate_stage4_stride(self.stage4_stride)
        if self.diagnostic_epoch is not None and (
                type(self.diagnostic_epoch) is not int
                or not 1 <= self.diagnostic_epoch <= self.max_epochs):
            raise ValueError("diagnostic_epoch must be None or an epoch within max_epochs")


@dataclass(frozen=True)
class EpochLoss:
    epoch: int
    training_loss: float
    training_loss_kind: str  # "positive_weighted_bce" or "unweighted_bce"
    validation_bce: float


@dataclass(frozen=True)
class FoldTrainingResult:
    outer_fold: int
    epochs: tuple[EpochLoss, ...]
    best_epoch: int
    best_validation_bce: float
    optimizer_steps: int
    positive_weight: float | None
    checkpoint_path: Path
    metrics_path: Path
    run_record: dict
    diagnostics: tuple[dict, ...]


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _membership_record(plan: FoldPlan) -> dict:
    # Private signatures only, never literal identities or individual labels.
    return {name: {
        "count": len(part),
        "signature": record_signature({"patients": sorted(
            [[p.patient_id, p.cohort, p.outcome, p.fold, p.has_gt] for p in part],
        )}),
    } for name, part in (("inner_train", plan.inner_train),
                        ("inner_validation", plan.inner_validation))}


def _run_record(model: nn.Module, plan: FoldPlan, samples: SampleSource,
                config: TrainingConfig, source_revision: str | None,
                protocol_sha256: str | None) -> dict:
    source_dir = Path(__file__).parent
    names = ("resnet3d.py", "ncct_model_config.py", "ncct_checkpoint.py",
             "ncct_training.py", "ncct_samples.py", "native_ncct_input.py",
             "ncct_intensity.py", "ncct_splits.py", "data_contract.py",
             "ncct_private_output.py", "ncct.py", "ncct_paths.py", "geometry.py",
             "ncct_development.py", "ctp_metrics.py")
    def package_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {
        "schema_version": 1, "architecture": model.architecture_record(),
        "training": asdict(config), "outer_fold": plan.outer_fold,
        "split_seed": plan.seed, "validation_fraction": plan.validation_fraction,
        "membership": _membership_record(plan),
        "input": getattr(samples, "input_record", {"status": "not_recorded"}),
        "source_revision": source_revision, "protocol_sha256": protocol_sha256,
        "source_file_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                               for name in names},
        "initial_parameters_sha256": _parameter_digest(model),
        "environment": {
            "python": sys.version.split()[0], "numpy": str(np.__version__),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "nibabel": package_version("nibabel"), "scipy": package_version("scipy"),
            "cudnn": torch.backends.cudnn.version(),
            "device": config.device,
            "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "selection": "strict-min-unweighted-inner-validation-BCE-earliest-tie",
    }


def _training_diagnostic(model, plan, samples, device, epoch: int) -> dict:
    """Fixed-state train-only diagnostics; never a selection/stopping signal."""
    logits, erased, losses, embeddings = [], [], [], []
    model.eval()
    with torch.no_grad():
        for patient in plan.inner_train:
            try:
                sample = samples.load(patient)
            except Exception:
                raise RuntimeError("NCCT diagnostic sample loading failed") from None
            x, y = _tensor_pair(sample, patient, device)
            embedding = model.forward_features(x)
            logit = model.classifier(embedding).squeeze(1)
            blank = model(torch.zeros_like(x))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            if not all(torch.isfinite(t).all().item() for t in (embedding, logit, blank, loss)):
                raise ValueError("nonfinite NCCT training diagnostic")
            logits.append(float(logit.item()))
            erased.append(float(blank.item()))
            losses.append(float(loss.item()))
            embeddings.append(embedding.detach().cpu().numpy()[0])
    values, blank_values = np.asarray(logits), np.asarray(erased)
    channel_sd = np.asarray(embeddings).std(axis=0)
    return {
        "epoch": epoch, "partition": "inner_train", "count": len(logits),
        "fixed_model_bce": math.fsum(losses) / len(losses),
        "logit_mean": float(values.mean()), "logit_sd": float(values.std()),
        "logit_range": [float(values.min()), float(values.max())],
        "erased_logit_sd": float(blank_values.std()),
        "normal_erased_mean_absolute_logit_change": float(np.abs(values - blank_values).mean()),
        "embedding_channel_sd_mean": float(channel_sd.mean()),
        "embedding_channel_sd_max": float(channel_sd.max()),
    }


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _tensor_pair(sample: NCCTSample, patient: OutcomePatient,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if type(sample) is not NCCTSample or not isinstance(sample.image, np.ndarray):
        raise ValueError("sample loader must return an NCCTSample")
    image = sample.image
    if (image.dtype != np.dtype(np.float32) or image.ndim != 4 or image.shape[0] != 1 or
            any(size < 1 for size in image.shape[1:]) or not np.isfinite(image).all()):
        raise ValueError("sample image must be finite float32 (1,D,H,W)")
    if type(sample.target) is not np.float32 or float(sample.target) not in {0.0, 1.0}:
        raise ValueError("sample target must be binary float32")
    if float(sample.target) != float(binary_outcome(patient)):
        raise ValueError("sample target disagrees with the validated patient outcome")
    # The loader has one channel; add only the batch dimension here.
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(device)
    target_tensor = torch.tensor([float(sample.target)], dtype=torch.float32, device=device)
    return image_tensor, target_tensor


def run_one_fold(
    plan: FoldPlan,
    samples: SampleSource,
    *,
    private_output_dir: str | Path,
    run_id: str,
    config: TrainingConfig = TrainingConfig(),
    source_revision: str | None = None,
    protocol_sha256: str | None = None,
    require_complete_provenance: bool = False,
    expected_initial_parameters_sha256: str | None = None,
) -> FoldTrainingResult:
    """Train from scratch and select one checkpoint using inner-validation BCE.

    The outer-test partition is checked for separation, then never loaded or
    iterated. This function does not evaluate test outcomes or print case data.
    """
    validate_fold_plan(plan)
    if not callable(getattr(samples, "load", None)):
        raise TypeError("samples must provide load(patient)")
    if type(config) is not TrainingConfig:
        raise TypeError("config must be TrainingConfig")
    for name, value, length in (("source_revision", source_revision, 40),
                                ("protocol_sha256", protocol_sha256, 64),
                                ("expected_initial_parameters_sha256", expected_initial_parameters_sha256, 64)):
        if value is not None and (type(value) is not str or len(value) != length
                                 or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    if type(require_complete_provenance) is not bool:
        raise TypeError("require_complete_provenance must be boolean")
    if require_complete_provenance and (
            source_revision is None or protocol_sha256 is None
            or getattr(samples, "input_record", {}).get("representation") != "native-NIfTI-declared-scaling-float32-zyx"):
        raise ValueError("NCCT execution needs registered source, protocol and input provenance")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    weight = inner_train_positive_weight(plan) if config.positive_weighting else None
    paths = create_private_run(private_output_dir, run_id, plan.outer_fold)
    # Seed before model construction and any CUDA tensor allocation.
    _seed_everything(config.seed + plan.outer_fold)
    # Each invocation creates a new model and optimizer. The outer fold is fixed.
    device = torch.device(config.device)
    model = CompactResNet10(width=16, pooling=config.pooling,
                           stage4_stride=config.stage4_stride).to(device)
    run_record = _run_record(model, plan, samples, config, source_revision, protocol_sha256)
    if (expected_initial_parameters_sha256 is not None
            and run_record["initial_parameters_sha256"] != expected_initial_parameters_sha256):
        raise ValueError("NCCT paired initialization disagrees before optimization")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    train_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=None if weight is None else torch.tensor(weight, dtype=torch.float32, device=device),
    )
    validation_loss_fn = nn.BCEWithLogitsLoss()
    metrics = paths.metrics_path
    checkpoint: Path | None = None
    training_loss_kind = "positive_weighted_bce" if weight is not None else "unweighted_bce"
    history: list[EpochLoss] = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    optimizer_steps = 0
    diagnostics: list[dict] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = list(plan.inner_train)
        random.Random(config.seed + plan.outer_fold + epoch).shuffle(order)
        train_losses: list[float] = []
        for patient in order:
            try:
                sample = samples.load(patient)
            except Exception:
                raise RuntimeError("NCCT sample loading failed") from None
            x, y = _tensor_pair(sample, patient, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if tuple(logits.shape) != (1,) or not torch.isfinite(logits).all().item():
                raise ValueError("model must return one finite logit per patient")
            loss = train_loss_fn(logits, y)
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(parameter.grad is not None and
                   not torch.isfinite(parameter.grad).all().item()
                   for parameter in model.parameters()):
                raise ValueError("nonfinite parameter gradient")
            optimizer.step()
            optimizer_steps += 1
            if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise ValueError("nonfinite model parameter after optimizer step")
            train_losses.append(float(loss.detach().item()))
            del x, y, logits, loss

        if epoch == config.diagnostic_epoch:
            diagnostics.append(_training_diagnostic(model, plan, samples, device, epoch))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for patient in plan.inner_validation:
                try:
                    sample = samples.load(patient)
                except Exception:
                    raise RuntimeError("NCCT sample loading failed") from None
                x, y = _tensor_pair(sample, patient, device)
                logits = model(x)
                if tuple(logits.shape) != (1,) or not torch.isfinite(logits).all().item():
                    raise ValueError("model must return one finite logit per patient")
                loss = validation_loss_fn(logits, y)
                if not torch.isfinite(loss).item():
                    raise ValueError("nonfinite validation loss")
                validation_losses.append(float(loss.item()))
                del x, y, logits, loss
        train_mean = math.fsum(train_losses) / len(train_losses)
        validation_mean = math.fsum(validation_losses) / len(validation_losses)
        if not math.isfinite(train_mean) or not math.isfinite(validation_mean):
            raise ValueError("nonfinite aggregate loss")
        history.append(EpochLoss(epoch, train_mean, training_loss_kind, validation_mean))
        # Exact ties are non-improvements; the first minimum wins.
        if validation_mean < best_loss:
            best_loss, best_epoch, stale_epochs = validation_mean, epoch, 0
            checkpoint = paths.fold_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            try:
                save_ncct_checkpoint(model, checkpoint, run_record=run_record, epoch=epoch)
            except Exception:
                raise PrivateOutputError("private checkpoint could not be written") from None
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if checkpoint is None:
        raise RuntimeError("no finite validation checkpoint was selected")
    result = FoldTrainingResult(
        plan.outer_fold, tuple(history), best_epoch, best_loss,
        optimizer_steps, weight, checkpoint, metrics,
        run_record, tuple(diagnostics),
    )
    # Aggregate losses only. No patient IDs, per-patient labels or paths.
    try:
        with io.TextIOWrapper(open_private_binary(metrics), encoding="utf-8") as stream:
            json.dump({
                "outer_fold": result.outer_fold,
                "pooling": config.pooling,
                "stage4_stride": list(config.stage4_stride),
                "run_record": run_record,
                "run_signature": record_signature(run_record),
                "diagnostics": diagnostics,
                "best_epoch": result.best_epoch,
                "best_validation_bce": result.best_validation_bce,
                "optimizer_steps": result.optimizer_steps,
                "positive_weight": result.positive_weight,
                "epochs": [vars(item) for item in result.epochs],
            }, stream, indent=2)
    except Exception:
        raise PrivateOutputError("private aggregate metrics could not be written") from None
    return result
