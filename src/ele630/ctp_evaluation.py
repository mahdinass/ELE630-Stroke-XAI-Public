"""Read-only, aggregate-only evaluation of the committed fold-0 CTP best model."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F

from .ctp_continuation import (
    _digest, _private_file, publish_json, read_committed_state, run_lock, signature,
)
from .ctp_development import CTPDevelopmentData
from .ctp_metrics import CTPAggregateMetrics, CTPMetricError, aggregate_ctp_metrics
from .ctp_model import CompactCTPMean
from .ctp_training import CTPTrainingConfig, _split_digest
from .data_contract import ACTIVE_PROFILE, DatasetContract
from .ncct_private_output import open_private_run
from .ncct_splits import binary_outcome


class CTPEvaluationError(RuntimeError):
    """A sanitized failure with no private path or patient information."""


_OUTPUT_NAME = "aggregate_inner_validation_evaluation.json"
_CHECKPOINT = re.compile(r"checkpoint_epoch_(\d{3})(?:_[0-9a-f]{16})?\.pt\Z")


def _read_json(path: Path) -> dict:
    _private_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("invalid private aggregate record")
    return value


def _check_configuration(saved: dict, contract: DatasetContract,
                         data: CTPDevelopmentData, model: torch.nn.Module) -> None:
    """Check the stored Stage 4R experiment without creating an optimizer."""
    training = saved.get("training")
    optimizer = saved.get("optimizer")
    if type(training) is not dict or type(optimizer) is not dict:
        raise ValueError("private experiment configuration is invalid")
    config = CTPTrainingConfig(max_epochs=training.get("max_epochs"),
                               device=training.get("device"))
    if (training != asdict(config) or saved.get("continuation_format") != 1
            or saved.get("mode") != "development" or saved.get("outer_fold") != 0
            or saved.get("outer_fingerprint") != contract.fingerprint
            or saved.get("inner_partition_sha256") != _split_digest(data)
            or saved.get("model") != model.configuration
            or saved.get("loss") != "unweighted_binary_cross_entropy_with_logits"
            or saved.get("target_mapping") != {"good": 0, "poor": 1}
            or saved.get("batch_size") != 1 or saved.get("inner_split_seed") != 630
            or saved.get("inner_validation_fraction") != 0.20
            or saved.get("inner_training_count") != len(data.plan.inner_train)
            or saved.get("inner_validation_count") != len(data.plan.inner_validation)
            or saved.get("outer_test_count") != len(data.plan.outer_test)
            or saved.get("overall_epoch_limit") != config.max_epochs
            or saved.get("patience") != 3
            or saved.get("checkpoint_selection") != "strict_minimum_inner_validation_bce"
            or saved.get("ties") != "keep_earlier_epoch"
            or saved.get("resume_supported") is not True
            or optimizer.get("name") != "AdamW"
            or optimizer.get("lr") != 1e-4 or optimizer.get("weight_decay") != 1e-4
            or optimizer.get("betas") != [0.9, 0.999]
            or optimizer.get("eps") != 1e-8
            or optimizer.get("amsgrad") is not False
            or optimizer.get("foreach") is not False):
        raise ValueError("private experiment configuration is incompatible")


def _best_state_dict(fold_dir: Path, config: dict, summary: dict,
                     training_count: int, validation_count: int) -> dict:
    """Trace the selected checkpoint through verified, contiguous epoch commits."""
    epoch, latest, state, commits = read_committed_state(fold_dir)
    history = state.get("history") if type(state) is dict else None
    if (type(state) is not dict or state.get("version") != 1
            or state.get("experiment_signature") != signature(config)
            or type(state.get("completed_epoch")) is not int or state["completed_epoch"] != epoch
            or type(state.get("next_epoch")) is not int or state["next_epoch"] != epoch + 1
            or type(history) is not list or len(history) != epoch
            or type(state.get("optimizer_steps")) is not int
            or state["optimizer_steps"] != epoch * training_count
            or epoch > config["training"]["max_epochs"]):
        raise ValueError("private selected state is incompatible")
    best_epoch, best_bce, stale = None, None, 0
    for index, record in enumerate(history, start=1):
        if (type(record) is not dict or type(record.get("epoch")) is not int
                or record["epoch"] != index
                or record.get("training_patients") != training_count
                or record.get("validation_patients") != validation_count
                or type(record.get("validation_bce")) not in (float, int)
                or not math.isfinite(record["validation_bce"])):
            raise ValueError("private validation history is invalid")
        if best_bce is None or record["validation_bce"] < best_bce:
            best_epoch, best_bce, stale = index, record["validation_bce"], 0
        else:
            stale += 1
        if stale >= 3 and index != epoch:
            raise ValueError("private epoch history exceeds patience")
    best = state.get("best_checkpoint")
    if (type(best) is not dict or type(best.get("name")) is not str
            or type(best.get("sha256")) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", best["sha256"])):
        raise ValueError("private selected checkpoint reference is invalid")
    matched = _CHECKPOINT.fullmatch(best["name"])
    if (matched is None or int(matched[1]) != best_epoch
            or state.get("best_epoch") != best_epoch
            or state.get("best_validation_bce") != best_bce
            or state.get("stale_epochs") != stale
            or latest.get("record") != history[-1]
            or latest.get("best_checkpoint") != best
            or latest.get("checkpoint_selected") is not (best_epoch == epoch)
            or summary.get("status") != "complete"
            or summary.get("mode") != "development"
            or summary.get("outer_fold") != 0
            or "next_epoch" not in summary
            or summary.get("next_epoch") is not None
            or summary.get("completed_epochs") != epoch
            or summary.get("optimizer_steps") != epoch * training_count
            or summary.get("training_patient_visits") != epoch * training_count
            or summary.get("best_epoch") != best_epoch
            or summary.get("best_validation_bce") != best_bce
            or summary.get("validation_patient_visits") != epoch * validation_count
            or summary.get("epochs") != history
            or not (epoch == config["training"]["max_epochs"] or stale >= 3)):
        raise ValueError("private best-checkpoint provenance is inconsistent")
    best_commit = commits[best_epoch - 1][1]
    if best_commit.get("checkpoint_selected") is not True or best_commit.get("best_checkpoint") != best:
        raise ValueError("private selected checkpoint was not committed")
    checkpoint_path = fold_dir / best["name"]
    _private_file(checkpoint_path)
    if _digest(checkpoint_path) != best["sha256"]:
        raise ValueError("private selected checkpoint checksum disagrees")
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict):
        raise ValueError("private selected checkpoint is invalid")
    # Verify the separate selected file matches the weights saved in its epoch.
    selected_state = (state if best_epoch == epoch else
                      read_committed_state(fold_dir, selected_epoch=best_epoch)[2])
    selected_weights = selected_state.get("model_state") if type(selected_state) is dict else None
    if (not isinstance(selected_weights, dict)
            or selected_state.get("experiment_signature") != signature(config)
            or type(selected_state.get("completed_epoch")) is not int
            or selected_state["completed_epoch"] != best_epoch
            or type(selected_state.get("next_epoch")) is not int
            or selected_state["next_epoch"] != best_epoch + 1
            or selected_state.get("best_checkpoint") != best
            or set(weights) != set(selected_weights)
            or any(not isinstance(weights[key], torch.Tensor)
                   or not torch.equal(weights[key], selected_weights[key]) for key in weights)):
        raise ValueError("private selected checkpoint disagrees with its epoch state")
    return weights


def evaluate_best_ctp_inner_validation(
    contract: DatasetContract,
    repository,
    *,
    private_output_dir: str | Path,
    run_id: str,
    outer_fold: int = 0,
    device: str = "cpu",
    model_factory: Callable[..., torch.nn.Module] | None = None,
) -> CTPAggregateMetrics:
    """Evaluate the selected checkpoint; None retains the Stage 4S mean model.

    Supply CompactCTPMeanStd explicitly for Stage 4U. The requested model's
    configuration must match the signed run before any image is opened.
    """
    try:
        if model_factory is not None and not callable(model_factory):
            raise TypeError("model_factory must be callable")
        if type(outer_fold) is not int or outer_fold != 0:
            raise ValueError("only the established fold-0 development run is supported")
        if type(device) is not str or device not in {"cpu", "cuda"}:
            raise ValueError("evaluation device must be cpu or cuda")
        if (type(contract) is not DatasetContract
                or len(contract.patients) != ACTIVE_PROFILE.patients
                or contract.fingerprint != ACTIVE_PROFILE.fingerprint):
            raise ValueError("the active outcome contract is required")
        data = CTPDevelopmentData(contract, repository, mode="development")
        validation_ids = {patient.patient_id for patient in data.validation_patients}
        outer_ids = {patient.patient_id for patient in data.plan.outer_test}
        if (not validation_ids or len(validation_ids) != len(data.validation_patients)
                or validation_ids & outer_ids
                or any(patient.fold == 0 for patient in data.validation_patients)):
            raise ValueError("inner-validation membership overlaps outer test")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA evaluation is unavailable")
        paths = open_private_run(private_output_dir, run_id, 0)
        with run_lock(paths.fold_dir):
            output = paths.fold_dir / _OUTPUT_NAME
            if output.exists() or output.is_symlink():
                raise ValueError("aggregate evaluation already exists")
            saved_config = _read_json(paths.fold_dir / "run_config.json")
            summary = _read_json(paths.metrics_path)
            with torch.random.fork_rng(devices=[]):
                factory = CompactCTPMean if model_factory is None else model_factory
                model = factory(width=16)
            _check_configuration(saved_config, contract, data, model)
            weights = _best_state_dict(paths.fold_dir, saved_config, summary,
                                       len(data.training_patients), len(data.validation_patients))
            model.load_state_dict(weights, strict=True)
            if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise ValueError("selected model contains nonfinite parameters")
            model.to(torch.device(device))
            model.eval()
            labels: list[int] = []
            probabilities: list[float] = []
            losses: list[float] = []
            with torch.no_grad():
                for patient in data.validation_patients:
                    sample = data.open_validation(patient)
                    label = binary_outcome(patient)
                    if sample.target != label:
                        raise ValueError("evaluation target disagrees with validated contract")
                    target = torch.tensor([label], dtype=torch.float32,
                                          device=torch.device(device))
                    logit = model(sample.series)
                    if tuple(logit.shape) != (1,) or not torch.isfinite(logit).all().item():
                        raise ValueError("evaluation requires one finite logit")
                    loss = F.binary_cross_entropy_with_logits(logit, target)
                    if not torch.isfinite(loss).item():
                        raise ValueError("evaluation BCE is nonfinite")
                    labels.append(label)
                    probabilities.append(float(torch.sigmoid(logit).item()))
                    losses.append(float(loss.item()))
                    del sample, target, logit, loss
            aggregate = aggregate_ctp_metrics(labels, probabilities, losses)
            publish_json(paths.fold_dir, _OUTPUT_NAME, aggregate.as_dict())
            return aggregate
    except CTPMetricError as exc:
        raise CTPEvaluationError(str(exc)) from None
    except Exception:
        raise CTPEvaluationError("CTP inner-validation evaluation stopped") from None
