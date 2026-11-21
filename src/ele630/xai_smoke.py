"""Actual frozen-topology synthetic device smoke, including private clinical IPC."""
from __future__ import annotations

import io
import sys
import unittest
from dataclasses import asdict
from contextlib import nullcontext

import numpy as np
import torch


def gpu_test_suite(project):
    sys.path.insert(0, str(project/"tests"))
    from test_xai_foundation import TorchFoundationTests
    from test_xai_quantitative import QuantitativeNumpyTests, QuantitativeTorchTests
    from test_xai_infrastructure import InfrastructureTests, PipelineTests
    from test_xai_review import PatchReviewTests, ResourceReviewTests, MaskPreflightReviewTests
    from test_xai_ctp_cache import CacheTests, CacheTorchTests
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (TorchFoundationTests, QuantitativeNumpyTests,
                                           QuantitativeTorchTests, InfrastructureTests, PipelineTests,
                                           PatchReviewTests, ResourceReviewTests, MaskPreflightReviewTests,
                                           CacheTests, CacheTorchTests))
    tested = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    if not tested.wasSuccessful() or tested.skipped or tested.testsRun < 20:
        raise ValueError("Stage7 GPU-stack synthetic tests failed or skipped")
    return {"tests": tested.testsRun, "skipped": 0}


def device_smoke(device, clinical_bridge, revision):
    from .fusion_synthetic import fictional_plan, FictionalCTP
    from .fusion_features import DevelopmentFeatures
    from .resnet3d import CompactResNet10
    from .ctp_model import CompactCTPMean
    from .xai_fused_score import torch_state, fused_logit_torch
    from .xai_analysis import analyse_case
    from .xai_store import validate_case
    from .ncct_training import _parameter_digest
    from .xai_data import cached_development_features
    from .xai_ctp_cache import cached_series
    results = {}
    for modality in ("ncct", "ctp"):
        torch.manual_seed(630)
        model = (CompactResNet10() if modality == "ncct" else CompactCTPMean()).to(device)
        plan = fictional_plan()
        indices = {p.patient_id: i for i, p in enumerate(plan.inner_train+plan.inner_validation)}
        def loader(patient):
            seed = 630+100*indices[patient.patient_id]
            if modality == "ncct":
                return np.random.default_rng(seed).random((1, 8, 32, 32), dtype=np.float32)
            return FictionalCTP(seed, shape=(8, 32, 32))
        source = (cached_development_features(plan, loader, model) if modality == "ctp"
                  else DevelopmentFeatures(plan, loader, model, modality))
        before = _parameter_digest(model)
        train, _ = source.partition(plan.inner_train)
        validation, probs = source.partition(plan.inner_validation)
        request = {"mode": "synthetic", "plan": asdict(plan), "train": train.tolist(),
                   "validation": validation.tolist(), "imaging_probabilities": probs.tolist(),
                   "source_revision": revision}
        private = clinical_bridge(request)
        state = torch_state(private["state"], device)
        for i in range(len(validation)):
            e = torch.as_tensor(validation[i:i+1], device=device)
            c = torch.as_tensor(private["clinical_validation"][i:i+1], device=device)
            z = fused_logit_torch(e, c, state)
            if abs(z.item()-private["logits"][i]) > 1e-8 or abs(z.sigmoid().item()-private["probabilities"][i]) > 1e-10:
                raise ValueError("device/sklearn reconstruction equivalence failed")
        value = loader(plan.inner_validation[0])
        if modality == "ncct":
            value = torch.from_numpy(value).unsqueeze(0).to(device)
        labels = np.ones((32, 32, 8), dtype=np.uint8)
        labels[16:24, 16:24, 4:] = 2
        labels[20:24, 20:24, 4:] = 3
        labels[:3] = 0
        with cached_series(value) if modality == "ctp" else nullcontext(value) as prepared:
            record = analyse_case(model, prepared, torch.as_tensor(private["clinical_validation"][0:1], device=device),
                                  state, modality, np.eye(4), labels_xyz=labels,
                                  expected_logit=private["logits"][0], clinical_logit=private["clinical_logits"][0])
        validate_case(record)
        if _parameter_digest(model) != before or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise ValueError("device smoke changed original frozen model")
        results[modality] = {"sklearn_equivalence_cases": len(validation), "complete_quantitative_cases": 1,
                             "frozen_weights_unchanged": True}
        del private, source, record, model, state, value
    return results
