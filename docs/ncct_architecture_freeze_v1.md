# ELE630 — NCCT Architecture and Development Recipe Freeze v1

**Status:** Architecture and development recipe selected for downstream Strategy V4 clinical integration. This is **not** an independent performance validation, final cross-validation result, or a freeze of one universal set of trained weights.

**Decision date:** 18 October 2026
**Source revision:** `714ce2cbbbe1e235f9cc5e95bb5540747d20d991`
**Supporting experiment:** Stage 5C2, UiS Slurm job `486314`, `COMPLETED 0:0`, 19 min 22 sec
**Development cohort:** n=143, fixed patient-level five-fold split, seed 630, fingerprint `3db8c4fd5fef0e4e`. Fold 0 development: train n=92, validation n=23, outer test n=28 **not evaluated**.

## 1. Frozen imaging architecture

| Property | Locked value |
| --- | --- |
| Imaging modality | Admission-time, skull-stripped NCCT |
| Network | `CompactResNet10` (compact 3D ResNet-10) |
| Input channels / head | One NCCT channel / one binary-outcome logit |
| Width | 16; stage widths 16, 32, 64, 128 |
| Normalization | GroupNorm, 8 groups at default widths |
| Pooling | Global average (`pooling="avg"`), **not** avgmax |
| Stage-4 stride | `(2, 2, 2)`, **not** `(1, 1, 1)` |
| Initialization | From scratch (no pretrained MedicalNet or foundation-model initialization) |
| Source/topology identity | Keep architecture implementation and semantic configuration tied to the above source revision; no silent architecture changes |

## 2. Frozen model-input policy

- Use the supplied, already preprocessed and curated NCCT. Respect original NIfTI-declared scaling.
- Convert to contiguous float32 native array in `(1,z,y,x)` channel-depth-height-width order (plus batch dimension for PyTorch).
- Clip Hounsfield units to `[-20,120]`, then scale linearly to `[0,1]`.
- No additional common-grid resampling, cropping, padding, orientation modification, augmentation, image-fitted normalization, or label-dependent patient exclusions in this NCCT experiment recipe.
- Do not feed clinical variables, registered CTP, acute core/penumbra masks, or lesion ground truth into the imaging-only network. Clinical integration is a **separate frozen-architecture extension**.

## 3. Frozen development training and selection policy

| Property | Locked value |
| --- | --- |
| Batch size | 1, one optimizer update per training patient |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Objective | Unweighted `BCEWithLogitsLoss` |
| Epoch budget | Maximum 10 epochs |
| Early stopping | Patience 3 |
| Selection | Strict minimum mean unweighted inner-validation BCE; earliest exact tie |
| Seed | 630; deterministic seeding and deterministic data shuffling as in source |
| Other | No learning-rate scheduler, class weighting, AMP, or gradient accumulation |

The **epoch-7 checkpoint** was selected in the completed **fold-0 development run**, but **epoch 7 is not a universal number of epochs to impose on all folds**. In any future outer-fold training, fit new fold-specific weights with the locked recipe and legitimate fold-specific training-only/inner-validation boundaries. Preserve private checkpoint provenance, hashes, and actual training artifacts **on the approved UiS private runtime only**, not in this code repository or public documents.

## 4. Supporting controlled experiment

Single-seed, same-input, same-development-split Stage 5C2 comparison. Reported aggregates were read from the private UiS comparison file; they were not independently inspected by the assistant.

| Development metric | Original stride `(2,2,2)` | Challenger stride `(1,1,1)` |
| --- | ---: | ---: |
| Selected epoch | 7 | 5 |
| Validation BCE (primary, lower better) | **0.614388** | 0.614564 |
| Training-prior BCE | 0.614779 | 0.614779 |
| AUROC | **0.616071** | 0.446429 |
| Average precision | **0.409890** | 0.343504 |
| Brier score | **0.211671** | 0.211746 |
| Balanced accuracy at 0.5 | 0.500000 | 0.500000 |
| Predicted-probability SD | 0.000780 | 0.000209 |
| Epoch-3 train BCE | 0.623567 | 0.623206 |
| Epoch-3 logit SD | 0.003561 | 0.001551 |
| Epoch-3 mean embedding-channel SD | 0.009643 | 0.006987 |

**Paired primary difference (stride1 minus stride2):** validation BCE `+0.000176` (worse). Original is only `0.000391` BCE lower than the training-prior reference. Both models predict nearly constant probabilities and neither establishes clinically useful NCCT discrimination. There is **no demonstrated benefit** of reducing stage-4 stride. This supports freezing the original lower-complexity architecture as a **weak but reproducible imaging reference**, not claiming it is a strong prognostic model.

## 5. Decision and boundaries

1. **Freeze the above NCCT architecture, input, training, and checkpoint-selection rules. Stop open-ended NCCT architecture tuning** on the repeatedly reused 23-patient fold-0 development validation set.
2. **Keep the experiment's private fold-0 epoch-7 checkpoint as a development artifact**, not as the only weight file used to obtain all future predictions.
3. **Next required Strategy V4 comparison:** NCCT-only vs NCCT + admission/pre-treatment clinical variables; CTP temporal-mean-only vs CTP + the same clinical variables. Keep the clinical-only comparator. Prespecify one appropriately low-capacity, fold-leakage-safe clinical-integration mechanism and selection rule **before fitting or scoring**, rather than assuming an arbitrary 50/50 probability mix.
4. Preserve the already frozen **CTP temporal-mean architecture**. NCCT+CTP imaging fusion remains optional, not a prerequisite.
5. Outer-fold predictions and any separate completely unseen cohort remain unavailable for new architecture or clinical-feature tuning. Refit final models only under the agreed evaluation contract.
6. This development-selected design was influenced by repeatedly inspected fold-0 validation patients, who appear in other outer folds. Subsequent five-fold out-of-fold scores must be reported with this design-selection exposure limitation; **do not label them fully unbiased nested selection estimates**. Separate external evaluation, if compatible, provides a stronger independent check.
7. The existing project-created imaging-review QC clause and proposed `5C2D-QC-v1` technical clarification are historically documented separately. Do not imply radiologist certification or supervisor sign-off simply because architecture development is frozen.
8. This source-only decision document contains no patient rows, IDs, images, logits, predictions, or trained weight bytes.

## 6. Provenance and updates

- Governing scope: `ELE630_Strategy_V4_2026-10-06` sections 4, 6, 11, 12 and the post-meeting way-forward plan.
- Model source: `src/ele630/resnet3d.py`, `src/ele630/ncct_training.py`, `src/ele630/ncct_samples.py`, `src/ele630/ncct_intensity.py` at Git commit above.
- Executed two-arm runner: `scripts/experiments/run_ncct_stride_comparison.py`.
- Protocol and caveats: `docs/stage5c2_ncct_v4_decision_pack.md`, especially sections 6.3, 6.5.1 and 6.7.
- Internal project experiment records (not distributed): Stage 5C2 stride comparison and dated status notes.
- Changing any locked architecture or training parameter later requires a new versioned, justified decision. A separate, versioned fusion method does **not** constitute changing the frozen NCCT imaging architecture.
