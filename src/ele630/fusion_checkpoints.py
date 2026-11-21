"""Read-only selection of the two existing private development checkpoints.

Never register legacy weights, change old runs, or fall back to another model.
NCCT run name is discovered by its recorded source/job/configuration, not guessed.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json

from .fusion_runtime import RUNTIME, NCCT_REVISION, git, read_private_json, private_file
from .ncct_model_config import record_signature


def validate_ncct_record(summary, plan, model):
    from .ncct_training import TrainingConfig, _membership_record
    record = summary["run_record"]
    training = asdict(TrainingConfig(device="cuda", diagnostic_epoch=3))
    training["stage4_stride"] = [2, 2, 2]
    if (record.get("source_revision") != NCCT_REVISION
            or record.get("architecture") != model.architecture_record()
            or record.get("training") != training
            or record.get("membership") != _membership_record(plan)
            or record.get("split_seed") != 630 or record.get("outer_fold") != 0
            or record.get("validation_fraction") != .20
            or record.get("selection") != "strict-min-unweighted-inner-validation-BCE-earliest-tie"
            or record.get("environment", {}).get("slurm_job_id") != "486314"
            or summary.get("best_epoch") != 7
            or abs(summary.get("best_validation_bce", 0) - .614388) > 0.0000005
            or summary.get("run_signature") != record_signature(record)):
        raise ValueError("NCCT frozen run provenance disagrees")
    expected_input = {
        "representation": "native-NIfTI-declared-scaling-float32-zyx",
        "intensity": "clip-linear-scale", "lower": -20.0, "upper": 120.0,
        "spatial_resampling": False, "input_crop_pad": False,
        "background_replacement": False, "active_fingerprint": "3db8c4fd5fef0e4e",
    }
    if record.get("input") != expected_input:
        raise ValueError("NCCT input provenance disagrees")
    import hashlib
    names = {"resnet3d.py", "ncct_model_config.py", "ncct_checkpoint.py", "ncct_training.py",
             "ncct_samples.py", "native_ncct_input.py", "ncct_intensity.py", "ncct_splits.py",
             "data_contract.py", "ncct_private_output.py", "ncct.py", "ncct_paths.py",
             "geometry.py", "ncct_development.py", "ctp_metrics.py"}
    if record.get("source_file_sha256") != {
        name: hashlib.sha256(git("show", f"{NCCT_REVISION}:src/ele630/{name}")).hexdigest()
        for name in names
    }:
        raise ValueError("NCCT historical source digests disagree")
    history = summary.get("epochs", [])
    if not history or [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
        raise ValueError("NCCT epoch history invalid")
    best = min(history, key=lambda h: h["validation_bce"])
    if best["epoch"] != 7 or best["validation_bce"] != summary["best_validation_bce"]:
        raise ValueError("NCCT checkpoint is not the selected minimum")
    return record


def load_ncct(plan):
    from .resnet3d import CompactResNet10
    from .ncct_checkpoint import load_ncct_checkpoint
    from .ncct_private_output import _validate_base
    # The runtime root is documented; the Stage5C2 output-base spelling is not.
    # Validate its trusted alias using this workflow's established output base,
    # then inspect only the standard two-level run-summary layout. Never search
    # images, the source dataset, environment contents recursively or weight files.
    _validate_base(RUNTIME / "fusion_runs")
    candidates = []
    # Inspect only aggregate summaries. No images, clinical rows or legacy weights.
    for summary_path in sorted(RUNTIME.glob("*/*/outer_fold_0/aggregate_training.json")):
        summary = read_private_json(summary_path)
        record = summary.get("run_record", {})
        if (record.get("source_revision") == NCCT_REVISION
                and record.get("environment", {}).get("slurm_job_id") == "486314"
                and record.get("architecture", {}).get("stage4_stride") == [2, 2, 2]):
            candidates.append((summary_path.parent, summary))
    if len(candidates) != 1:
        raise ValueError("exactly one Stage 5C2 job486314 stride2 run is required")
    directory, summary = candidates[0]
    model = CompactResNet10(width=16, pooling="avg", stage4_stride=(2, 2, 2))
    record = validate_ncct_record(summary, plan, model)
    checkpoint = directory / "checkpoint_epoch_007.pt"
    private_file(checkpoint)
    manifest = load_ncct_checkpoint(model, checkpoint, expected_run_record=record, expected_epoch=7)
    return model, {"checkpoint_sha256": manifest["sha256"], "selected_epoch": 7,
                   "run_signature": manifest["run_signature"]}


def load_ctp(contract):
    from .ctp_model import CompactCTPMean
    from .ctp_development import CTPDevelopmentData
    from .ctp_evaluation import _check_configuration, _best_state_dict
    from .ctp_continuation import read_committed_state
    from .ncct_private_output import open_private_run
    # This precise run ID is recorded in the original Stage 4R/4S experiment log.
    paths = open_private_run(RUNTIME / "private_ctp_runs", "stage4r_ctp_dev_20260925_a", 0)
    config = read_private_json(paths.fold_dir / "run_config.json")
    summary = read_private_json(paths.metrics_path)
    model = CompactCTPMean(width=16)
    class NoImages:
        def open_series(self, patient):
            raise AssertionError("checkpoint preflight cannot open images")
    data = CTPDevelopmentData(contract, NoImages(), mode="development")
    _check_configuration(config, contract, data, model)
    if (config["training"] != {"max_epochs": 10, "device": "cuda", "patience": 3,
                               "learning_rate": 1e-4, "weight_decay": 1e-4, "seed": 630}
            or summary.get("best_epoch") != 2
            or abs(summary.get("best_validation_bce", 0) - .6143472169) > 5e-10):
        raise ValueError("frozen temporal-mean run identity disagrees")
    weights = _best_state_dict(paths.fold_dir, config, summary, 92, 23)
    expected = model.state_dict()
    import torch
    if set(weights) != set(expected) or any(
        not isinstance(weights[k], torch.Tensor) or weights[k].shape != expected[k].shape
        or weights[k].dtype != expected[k].dtype or not torch.isfinite(weights[k]).all() for k in expected
    ):
        raise ValueError("CTP checkpoint architecture or values disagree")
    model.load_state_dict(weights, strict=True)
    selected = read_committed_state(paths.fold_dir)[2]["best_checkpoint"]
    return model, {"checkpoint_sha256": selected["sha256"],
                   "selected_epoch": 2, "run_signature": record_signature(config)}
