# Controlled fold-0 held-out evaluation protocol v1

Historical evaluation protocol dated 9 November 2026. The final evaluation was
conducted on 10 November 2026. Original-artifact discovery preceded a separately
authorized deterministic train92 reconstruction.

This document fixes the rules before outcomes are scored. The concrete protocol
instance is frozen on UiS as `frozen.json` plus `seal.json`, after readiness and
development reproduction pass. A failing gate must not be relaxed after seeing
test results. The original operational records are retained in the restricted archive.

## Population and models

Original active contract: fingerprint `3db8c4fd5fef0e4e`, seed630, outer fold0,
validation fraction0.20. Original manifest and folds must validate under the
unchanged data contract. Check inventory150 separately. Eligible143 comprises
train92 (63 good/29 poor), repeatedly inspected dev23 (16/7), reserved test28
(19/9). No new partition, train+dev115 refit, model selection or tuning.
The private original manifests remain authoritative; source-level counts alone
do not establish server membership. Canonical plan, original file SHA256s and
private membership signatures enter the frozen instance. Known outcome metadata
are read for contract validation/stratification; preflight does not predict or
calculate test performance. It reads image headers and the clinical CSV header,
and streams file bytes for integrity hashes without interpreting test features.

Five models, all evaluated on the same outer28 in canonical order:

- NCCT CompactResNet10 width16, AVG, stride(2,2,2), selected epoch7 from job486314,
  source `714ce2cbbbe1e235f9cc5e95bb5540747d20d991`. Existing provenance loader
  uniquely discovers the original run and verifies `checkpoint_epoch_007.pt`.
- CTP CompactCTPMean width16, all40 registered frames, selected epoch2 in
  `stage4r_ctp_dev_20260925_a`; committed-state loader verifies selected bytes.
- Original clinical11 L2 logistic regression.
- NCCT embedding128 + training StandardScaler + PCA8 + clinical11 + logistic19.
- Corresponding CTP embedding128 + training StandardScaler + PCA8 + clinical11
  + logistic19.

No mean/std, TCN, AVGMAX, new geometry control, calibration, ablated clinical
features or new seed is eligible. Prior train92 exploratory analyses do not
promote another model into this final comparison.

## Artifact policy and acceptance criteria

First use `discover`, a read-only bounded walk of the documented private runtime
root, excluding environments, symlinks, caches and deep directories. It reports
file identity/size/hash, never deserializes pickle or runs patient inference.
Keep its detailed inventory on UiS. Absence from this bounded search alone is
not proof that no artifacts exist elsewhere; check any custodian-known approved
locations before attesting absence. No undisclosed server location is guessed.

**Original path:** explicitly select trusted fitted objects and their original
provenance using an owner-private original spec. File checksums are verified
before joblib deserialization. Require the original development source,
original train/dev signatures and the two exact imaging checkpoint provenances;
validate estimator types/hyperparameters, feature dimensions and PCA/scaler
training counts. Reuse fitted objects; no `.fit` is called on this path. Convert
to a private numerical state, then verify development predictions/metrics before
sealing. An unsupported object format, absent provenance or unknown feature
order is a blocker for reuse, not permission to claim it is an original model.
The defined importer supports separately serialized existing
ClinicalPreprocessor/StandardScaler/PCA/LogisticRegression objects. Other native
formats require an inspected adapter before reuse; do not rename incompatible
files or invent provenance. The original specification remains in the restricted archive.

**Reconstruction path:** the user must first inspect discovery and explicitly
attest originals are unavailable. Fit unchanged ClinicalPreprocessor, ImagePCA
and `_fit` constructors using train92 only, preserving original fit order and
including the no-image control construction. Use seed630, no warnings or
nonconvergence, no iteration/hyperparameter retries. Fit once per reconstruction
instance; the unchanged Stage7 reconstruction runs separately as a numerical
oracle. Validation values never enter a fit and outer clinical cells/images are
not requested. Complete reconstruction before the separate final command.

Fixed gates for both paths:

1. Original private Stage5C3 records must match source
   `54f289161a35206c00a517d7f12e2fdf1e7a4127`, modality, canonical train92/dev23
   signatures and exact imaging checkpoint provenance, using existing
   `xai_runtime.historical_gate`.
2. Reproduced fusion, clinical and imaging BCE/Brier/AUROC/BA must match the
   pre-existing fixed historical reference values within absolute `5e-7`.
   BCE/Brier/AUROC/BA/mean probability must also match the original private
   aggregate within `5e-7`. No-image/clinical BCE/Brier/mean agree within `1e-8`.
3. Reconstructed fusion development probabilities must match the unchanged
   Stage7 implementation within absolute `1e-10`, relative tolerance0. The
   exported numerical predictor in the imaging interpreter must match private
   clinical-worker probabilities for both fusion and clinical within `1e-10`.
4. If actual original patient-wise dev probabilities exist, supply the optional
   private reference, including canonical ordering and original provenance;
   require absolute `1e-10` against all three clinical/fusion probability arrays.
   Original Stage5B/5C3 code did not save these arrays. If absent, record this
   explicitly: validation then establishes original aggregate reproduction plus
   probability equivalence to Stage7, **not equality to unavailable original
   per-patient bytes**. Do not reconstruct a reference and label it original.

Reconstruction validation cannot establish binary equality with unavailable
historical objects. Its permitted scientific basis is the previously established
Stage7 deterministic recipe, original private aggregate/provenance gates and
strict probability equivalence. Any discrepancy stops preparation. No tolerance
is selected or widened from observed performance. Keep the limitation in Methods.

## Preprocessing and environment

NCCT: unchanged native NIfTI-declared scaling, float32ZYX, clip[-20,120], linear
scale[0,1]; no spatial resampling/crop/pad/augmentation/background replacement.
CTP: unchanged registered frame loader, float32/255, exactly all40 frames,
streamed mean embedding; no HU reinterpretation or temporal selection.
Clinical feature names/order and duration/wake-up rules are imported from
ClinicalPreprocessor. Numeric train medians/population mean/SD and categorical
train modes are frozen; outputs remain float32. Image StandardScaler is fitted
on train92 embedding128, followed by PCA8 full SVD, no whitening or post-scaling.
Logistic L2 C=1, lbfgs, intercept, no class weights, max_iter1000, tol1e-8,
random_state630, no warm start. No new estimator settings.

Preserve separate environments: imaging Python3.10.11, NumPy1.26.4,
Torch2.3.1+cu118, nibabel5.4.2; clinical worker Python3.10.11, NumPy1.24.3,
SciPy1.10.1, sklearn1.2.2, joblib1.1.1, threadpoolctl2.2.0. Existing environment
guards enforce them. Never unify the environments to bypass a failing guard.
The source-matched synthetic GPU smoke exercises both imaging paths, pinned
clinical IPC, reconstruction and numerical prediction before any preparation.

## Freeze, execution and retries

The frozen private bundle contains all preprocessing arrays, clinical/fusion
coefficients, development aggregates, path used, acceptance record, source and
protocol digests, checkpoint identities/hashes, dependency versions, canonical
plan digest and source manifest/fold/clinical file SHA256s. It contains no patient
identifiers, raw features or individual predictions. Fitted model information
remains confidential. The seal hashes canonical JSON, verified before final
execution. SHA256 seals detect changes, not malicious forgery by a file owner.

Preflight loads and validates the frozen bundle and original checkpoints on CPU,
checks original manifests, disjointness, feature header, all test image header
interfaces, required environment, smoke and output directory. It does not decode
image voxels, fit any object or generate predictions/performance. The private
runtime base must already exist owner-only; setup is a separate explicit command.
Source must be committed and unchanged, including untracked executable-source
guards. Input images must remain the original curated derivatives; the bundle
does not compute full byte digests of every medical image. This is an explicit
data-immutability assumption, not verified voxel-level identity.

Final requires an explicit execution flag and the exact reviewed seal digest.
It calls no preparation/optimizer/fitting routine. It restores numerical state,
transforms outer28 only and performs frozen inference under Torch inference_mode.
Check all image model state tensors before/after and verify no gradients;
hash numerical state before/after. Recheck source/data/checkpoint bindings before
publication. There is no tuning step and no choice based on test scores.

A single exclusive global claim prevents new final comparisons after a first
attempt. A process lock prevents concurrent technical retries. Completed runs
cannot repeat; failed attempts can retry only with the same run ID, source,
protocol, data/model hashes and explicit technical-retry flag. No preparation
after any final claim. Failure does not justify a new model or altered protocol.

## Metrics and outputs

Positive class1=mRS3–6. Threshold p>=0.5, fixed. Primary metric: unweighted BCE,
consistent with original model selection. Secondary: tied-midrank AUROC,
non-interpolated AP, Brier, balanced accuracy, confusion matrix rows actual and
columns predicted in [good,poor] order, n/good/poor counts.
Reuse original aggregate functions. Image final BCE uses float32 Torch
BCEWithLogits; clinical/fusion BCE uses original float64 probability clipping
only for BCE at [eps64,1-eps64]. Original Stage5C3 image reference used its
probability-based metric path; tiny numerical BCE differences between that
development reference and standalone logit evaluation are not a new loss.
No confidence intervals or significance claims are added; n28 is small.
No bootstrapping/evaluation resampling is performed, hence no resampling seed.

Private outputs: source-matched synthetic smoke, frozen.json, seal.json,
global final_claim.json, execution.lock, aggregate_final.json, sanitized Slurm
stdout/stderr. No patient-level prediction file is necessary or written.
`results` validates completeness, counts, metric ranges, matrix consistency and
frozen provenance, then projects exactly ten model/cohort rows into a TSV table.
Metric range validation allows1e-12 floating-point summation roundoff at [0,1]
boundaries (e.g. perfect AP with nine positives); it never clips/changes metrics.
Only that numeric table, after local review, is suitable for report transfer.
Raw private bundles, discovery inventories, hashes and model coefficients stay
on UiS. No output is automatically uploaded anywhere.

Label dev23 as repeatedly inspected development validation and outer28 as an
independent internal held-out test, never external/clinical validation. Report
all five results regardless of which performs best. Final test findings may
revise scientific conclusions, never select a new model or threshold.
