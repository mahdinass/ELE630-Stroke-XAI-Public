# Stage 5C3 clinical–imaging fusion protocol

This is the methodological portion of the 18 October 2026 protocol for the
two subsequently executed development comparisons. The frozen imaging
checkpoints, source identity and historical execution records remain in the
restricted research archive. See the [results summary](../README.md#results-development-versus-final-internal-test)
and [held-out protocol](final_heldout_protocol.md) for the later test.

## 1. Question and fixed experiments

Does a fixed, low-capacity image representation contribute useful information
beyond the already frozen eleven admission-time clinical predictors, measured by
development validation BCE? Exactly two new patient experiments are defined:

1. NCCT + Clinical: frozen NCCT encoder followed by the fusion procedure below.
2. CTP + Clinical: frozen all-40-frame temporal-mean encoder followed by the
   identical fusion procedure.

Lower fusion BCE than imaging alone is insufficient evidence of incremental
image information. Compare fusion with clinical-only and both no-image controls.
All comparisons are descriptive on reused development data. No effect is assumed.
There is no third new clinical research experiment: the clinical classifier is
refitted transiently using its exact frozen recipe to obtain aligned comparators.

## 2. Existing decisions and evidence

The three committed freeze documents are authoritative for architecture/input
and clinical specifications: `ncct_architecture_freeze_v1.md`,
`ctp_architecture_freeze_v1.md`, `clinical_feature_and_baseline_freeze_v1.md`.
Cross-checked source materials included the frozen local architecture and clinical
specifications, Strategy V4 decisions, Stage 5B and CTP experiments, and Stage 5C2.
Internal working notes are not part of the submitted code package.

Later completed experiments and commits supersede older pending-plan paragraphs.
GitHub connector independently confirmed main at the expected revision. Existing
descriptive references on the same validation n=23:

| Frozen reference | Provenance | BCE | AUROC |
|---|---|---:|---:|
| NCCT compact ResNet10 width16 AVG, stride2 | Stage 5C2 job486314, source714ce2c, epoch7 | 0.614388 | 0.616071 |
| CTP CompactCTPMean width16, 40 frames | Stage 4R/4S, evaluation job482687, epoch2 | 0.6143472169 | 0.6785714286 |
| Clinical L2 logistic | Stage 5B `stage5b_clinical_dev_01`, frozen12Oct | 0.2313358984 | 0.9821428571 |

These weights are development artifacts, not universal weights for future folds.
Architecture freezes do not authorize using fold0 weights across all outer folds.

## 3. Exact proposed mechanism and hyperparameters

1. Load the existing selected imaging checkpoint, verify its provenance and
   architecture, set the entire model to eval mode and `requires_grad_(False)`;
   clear gradients. Use float32 inference, no AMP and no optimizer. Check all
   model weights before/after extraction for equality by SHA256.
2. NCCT: existing native NIfTI scaling, float32 `(1,z,y,x)`, HU window
   `[-20,120]` linearly to `[0,1]`; no crop, pad, resample or augmentation.
   Use `CompactResNet10.forward_features` before its one-logit classifier:
   exactly 128 features.
3. CTP: use `CompactCTPMean._mean_embedding` with existing registered-series
   proxy/adapter, exactly frames0..39 once per patient, float32 values /255,
   no clipping, frame selection or NCCT window. It accumulates one 128-feature
   vector while releasing each spatial forward computation. No full 4D image
   is stacked; no temporal training/recomputation is necessary during inference.
4. Fit scikit-learn `StandardScaler(with_mean=True, with_std=True)` to the
   **92 training embeddings only**: population mean/SD, zero SD uses scale1.
5. Fit `PCA(n_components=8, svd_solver="full", whiten=False,
   random_state=630)` to those standardized training embeddings only.
   No post-PCA scaling or whitening. Preserve component variances. Reject fewer
   than nine training observations, nonfinite values, warnings, or training rank
   below8 at tolerance `eps64 * max(n,128) * largest_singular_value`.
   No automatic PCA-dimension reduction or alternative representation.
6. Fit a fresh existing `ClinicalPreprocessor` on training patients only. Use
   its exact frozen eleven-feature order and semantics, including wake-up masking,
   numeric median imputation/population scaling and categorical mode imputation.
   Do not fit a pooled143 or validation preprocessor.
7. Concatenate `[eight image PC scores, eleven clinical values]` (19 columns).
   Fit the existing frozen logistic constructor: L2 penalty, `C=1.0`, `lbfgs`,
   `max_iter=1000`, `tol=1e-8`, intercept enabled, `class_weight=None`,
   `random_state=630`, `warm_start=False`, `verbose=0`. Treat all fitting warnings,
   nonconvergence, invalid classes/coefficients or nonfinite inputs as failures.
   No retries with different settings, feature selection or capacity search.
8. After every fit is complete, transform validation features with the fitted
   training transformers and score once. Validation embeddings can be extracted
   beforehand, but never enter any `.fit`. All clinical validation rows are
   requested only after the three fixed logistic fits finish.

No thresholds, mixture weights, PCA dimensions, calibration or clinical features
are tuned. Poor outcome is class1; the fixed threshold is p>=0.5.

## 4. Boundaries, references and negative controls

Use the original143 cohort, original five folds, seed630, fingerprint
`3db8c4fd5fef0e4e`. Reconstruct the existing fold0 plan: training92,
validation23, outer-test28. Validate exact membership against both imaging
checkpoint records. Contract metadata necessarily contains cohort/fold/outcome
assignments, including excluded patients, to validate the immutable split; no
outer-test image or clinical feature is opened, transformed, predicted or fitted.
The clinical CSV adapter lexically tokenizes the source file but discards
excluded rows before indexing/interpreting clinical cells, as in Stage 5B.

Matched scores within each experiment:

- **Imaging-only:** original classifier applied to the very same frozen128-vector
  used for fusion; no logistic refit of the imaging baseline.
- **Clinical-only:** exact Stage 5B preprocessing/classifier fit on training92,
  transient predictions on validation23.
- **Zero-image control:** same19-column fitting procedure with eight zero image
  columns on both partitions and identical clinical values; this must reproduce
  the clinical reference to numerical tolerance. It checks dimensional/pipeline
  effects without any patient-specific image information.
- **Fixed shrinkage control:** `0.5*p_clinical + 0.5*training_poor_prevalence`.
  The coefficient0.5 is predeclared, never fitted or selected from validation.
  It checks whether generic movement toward the training prior can account for
  apparent probabilistic improvement. It is a diagnostic control, not proof that
  every possible shrinkage explanation has been excluded.
- **Training-prior control:** constant training poor-outcome prevalence.

Before declaring a patient run successful, recomputed imaging-only BCE must
match its recorded frozen reference within5e-7, and clinical BCE must match
Stage5B within1e-7. The real-run pre-publication gate also requires zero-image
versus clinical-only agreement for **BCE, Brier score and mean predicted
probability**, each with fixed **absolute tolerance1e-8 and relative tolerance0**
(`math.isclose`; equality at the tolerance is accepted). Nonfinite values fail.
This tolerance is prespecified for numerical equivalence of the two fixed
logistic fits, without inspecting real validation outcomes. These are
provenance/reproduction checks, not model-selection
rules. A mismatch stops publication; investigate without changing the model or
tuning on validation. Both successful experiments must have identical private
membership signatures and clinical aggregates agreeing within a fixed absolute
1e-10 numerical tolerance before joint reporting (counts must therefore agree
exactly). This allows floating-point differences across allocated CPU hardware.

## 5. Metrics and intended tables

Compute every model's BCE with the same float64 probability calculation:
clip only BCE inputs to `[eps64,1-eps64]`; retain original probabilities for all
other metrics. Minor differences from historical float32 logits BCE are expected
within the fixed reproduction tolerance. Use existing aggregate AUROC/AP/Brier/
balanced-accuracy routines. Dispersion is population probability SD, min, max
and mean. No patient-level probabilities, labels or group-specific rows escape.

| Model | n validation | BCE primary | AUROC | AP | Brier | BA@0.5 | Probability mean/SD/min/max |
|---|---:|---:|---:|---:|---:|---:|---|
| NCCT-only, matched recomputation | 23 | pending | pending | pending | pending | pending | pending |
| NCCT+Clinical | 23 | pending | pending | pending | pending | pending | pending |
| CTP-only, matched recomputation | 23 | pending | pending | pending | pending | pending | pending |
| CTP+Clinical | 23 | pending | pending | pending | pending | pending | pending |
| Clinical-only, matched computation | 23 | pending | pending | pending | pending | pending | pending |
| Zero-image control (per experiment) | 23 | pending | pending | pending | pending | pending | pending |
| Fixed clinical-prior shrinkage | 23 | pending | pending | pending | pending | pending | pending |
| Training-prior control | 23 | pending | pending | pending | pending | pending | pending |

For each fused model report fusion-minus-reference differences in BCE, AUROC,
AP, Brier, balanced accuracy and probability SD against imaging-only,
clinical-only, zero-image and shrinkage controls. Negative BCE/Brier deltas favor
fusion. Do not present p-values, confidence of clinical benefit, or independent
generalization claims from these reused23 cases.

## 6. Scientific limitations

The n23 validation partition has repeatedly influenced imaging architecture and
checkpoint selection. Neither these comparisons nor future ordinary five-fold
OOF estimates erase that design exposure. Image models have very small prediction
dispersion; PCA may amplify nuisance/cohort variation rather than prognosis.
Clinical performance is unusually high in seven validation poor outcomes and
requires cautious interpretation. The frozen clinical feature semantics/timing,
onset-to-CT codebook, wake-up masking and possible cohort shortcuts retain the
limitations in the clinical freeze. The fusion proposal changes none of them.

Eight image components and C=1 are explicit design choices, not evidence-backed
optimal settings. Failure to beat clinical-only is an acceptable outcome. The
controls are diagnostic, not sufficient to establish causal image utility.
No model is clinically certified. Final outer/external evaluation, calibration,
XAI and any alternative fusion design require separately versioned protocols;
this runner exposes no outer-test evaluation path.
