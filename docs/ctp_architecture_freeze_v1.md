# ELE630 — Dynamic CTP Architecture and Development Recipe Freeze v1

**Status:** The CTP temporal-mean architecture was selected and frozen by the team on 7 October 2026 after the prespecified three-arm comparison. This document records the existing decision for the next Strategy V4 clinical-integration phase; it does not claim independent test performance or freeze a universal set of weights.

**Freeze decision:** 7 October 2026
**Freeze record prepared:** 18 October 2026
**Source/provenance:** `src/ele630/ctp_model.py`, `src/ele630/ctp_training.py`, `docs/stage4p_ctp_mean_baseline.md`, `docs/stage4q_ctp_training_protocol.md`, `docs/stage4r_ctp_continuation_protocol.md`, Stage 4S and Stage 4V experiment records; evaluated TCN source revision `3daf230a3205e04e74982ce6d4cd4032fd8faa67` (not necessarily the original temporal-mean training revision). Current repository review revision `ec886423cc33382553eb2bf9128b1e28a782ce65`.

## 1. Frozen CTP imaging architecture

| Component | Frozen decision |
| --- | --- |
| Modality | Registered admission dynamic CTP, all 40 frames |
| Model | `CompactCTPMean(width=16)` |
| Spatial encoder | Scratch-initialized shared `CompactResNet10` width 16; global average pooling and GroupNorm |
| Frame representation | 128-dimensional spatial embedding per frame |
| Temporal representation | Arithmetic mean of all 40 frame embeddings; no temporal-order model |
| Classifier | One `Linear(128,1)` head, producing one poor-outcome logit |
| Initialization | Shared encoder initialized from scratch; head normal weights (SD 0.01), zero bias |
| Temporal alternatives rejected | `CompactCTPMeanStd` and `CompactCTPShallowTCN` for the primary branch |

Every frame follows the provided registration and native spatial grid. Use the curated processed 0–255 voxel values converted to float32, then divide by 255 without clipping or applying the NCCT HU window. Do not silently select frames, generate quantitative perfusion maps, introduce extra resampling, crop/pad the inputs, or add acute lesion masks to the primary predictor.

The existing two-pass frame-recomputation backward method is part of the established CTP implementation: collect a mean embedding without retaining 40 spatial graphs; differentiate the head; recompute one frame at a time to backpropagate encoder gradients, with one optimizer update per patient. The existing recorded input, frame-order, and memory/continuation contracts remain in effect.

## 2. Frozen CTP development training and selection recipe

| Component | Frozen rule |
| --- | --- |
| Model initialization | Scratch, seed 630 |
| Patient batch | One patient per optimizer step |
| Optimizer | AdamW, LR `1e-4`, weight decay `1e-4` (original fixed defaults) |
| Objective | Unweighted binary BCE with logits, good=0 and poor=1 |
| Training budget | Up to 10 epochs, patience 3 |
| Selection | Strict lowest mean inner-validation BCE, earliest exact tie |
| Training partition | Existing immutable fold contract, training-only parameter fitting |
| Final weights | Retrain fold-specific models using the frozen recipe; do not copy fold-0 development weights across folds |

The selected temporal-mean **fold-0 development checkpoint was epoch 2**, not an instruction to run exactly two epochs in future folds. Source-controlled continuation and best-checkpoint rules remain authoritative for individual runs.

## 3. Evidence for the 7 October selection

All values below are *development-only* aggregates on the same 23-person fold-0 inner-validation set. Higher ranking metrics for the mean+std challenger do not establish an independent improvement.

| Metric | Temporal mean (selected) | Mean + std | Shallow TCN |
| --- | ---: | ---: | ---: |
| Validation BCE (primary; lower better) | **0.6143472169** | 0.6144164593 | 0.6147012685 |
| AUROC | 0.6785714286 | 0.6964285714 | 0.5714285714 |
| Average precision | 0.4163308913 | 0.4294784580 | 0.3681175967 |
| Brier score | **0.2116544701** | 0.2116840143 | 0.2118049933 |
| Balanced accuracy at 0.5 | 0.500000 | 0.500000 | 0.500000 |

The temporal mean had the lowest descriptive BCE and Brier score. Differences were small, and the more complex temporal representations did not earn promotion on the prespecified criterion. This is not proof that temporal order cannot contain prognostic information. The selected CTP model also predicts probabilities close to the training prior, so clinical benefit is **not** established.

**Execution references:** temporal-mean selected-checkpoint evaluation Slurm `482687`, 27 September, epoch-2 checkpoint; final shallow-TCN aggregate evaluation Slurm `484527`, 7 October, 23 inner-validation patients. The private weights/metrics/checkpoint manifests remain on UiS and are not reproduced here.

## 4. Frozen boundaries and next experiment

The 143-outcome-patient cohort (98 good/45 poor), original patient-level five-fold assignments, seed 630 and fingerprint `3db8c4fd5fef0e4e` remain fixed. Fold-0 development was 92 train / 23 validation / 28 outer-test; this exploratory comparison did not evaluate outer-test predictions. Future fold-specific fitting must avoid using that fold's outer-test patients in fitted preprocessing, training or checkpoint selection.

**Next required Stage 5C3 comparison:** CTP-only versus CTP + the identical locked admission-time clinical feature set, with a clinical-only baseline. Freeze a separate, leakage-safe clinical-integration method *before* the combined-model experiment. Strategy V4 does **not** prescribe 50/50 probability averaging. NCCT+CTP imaging fusion is optional and separate.

Do not resume CTP architecture search. Document any later exception as a new versioned scientific decision. Repeated selection exposure to the fold-0 validation subset must be disclosed in later five-fold OOF interpretation; neither this comparison nor future scores alone establish unbiased independent generalization.
