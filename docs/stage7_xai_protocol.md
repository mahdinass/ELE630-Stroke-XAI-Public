# Stage 7 frozen source protocol v1

Governing plan: `ELE630_Strategy_V4_XAI_Full_Implementation_Plan.md`, read in
full. Baseline source `54f289161a35206c00a517d7f12e2fdf1e7a4127`.
User authorizes local synthetic engineering through 7D, overriding Section 11.
Human review of the resulting protocol, scoped commit, source-matched UiS GPU
smoke and explicit real-execution authorization remain mandatory.

## Scope and reconstruction

Original fingerprint `3db8c4fd5fef0e4e`, seed 630, fold0 train92/validation23,
outer28 locked. Metadata is read only to verify membership; outer images,
clinical cells and predictions are forbidden. Use original checkpoint guards
unchanged: NCCT job486314/source714ce2c/stride2/epoch7 and Stage4R CTP mean/epoch2.
Preserve original native NCCT [-20,120] and processed CTP float32/255 loaders.
No CNN training. All original model/training/fusion files remain unchanged.

Refit exactly the existing ClinicalPreprocessor, ImagePCA and fixed _fit
constructors using train92 only, in original fit order. PCA8 full SVD after
training population standardization; no whitening/post-scaling; clinical11.
Clinical fitting remains in the existing isolated clinical environment. Only
owner-private pipes/memory carry fitted state and transformed validation rows.
No fitted state, embeddings or patient probabilities are exported.

Float64 scaler/PCA/logistic computations in Torch; gradient casts through the
original float32 CNN embedding. Sklearn logit/probability equivalence: absolute
1e-8/1e-10, relative zero. Frame recomputation: atol=rtol=1e-6. Historical
fusion/clinical/imaging BCE and Brier: absolute 5e-7; AUROC and BA: 5e-7.
Mean probability: compare original private Stage5C3 aggregate at atol 5e-7,
because no exact historical mean is supplied in the plan. Also require clinical
and zero-image BCE/Brier/mean agreement 1e-8. Fixed thresholds never widen after
patient inspection. A mismatch fails before CAMs. Historical artifacts must
match original source, modality, checkpoint and membership, not just metrics.

## Scores, maps and resolution

Primary target: predicted-class support, +z if baseline z>=0, otherwise -z.
Freeze sign before any perturbation/randomization. Secondary raw-poor signed
maps are retained by multiplying primary signed maps by this baseline sign.
Grad-CAM = ReLU(sum(A * spatial_mean(G))); HiResCAM = ReLU(sum(A * G)).
Same forward output/layer/target for each pair; no image-only classifier CAM
is labeled a fused CAM. The exact intercept, image-PC and clinical terms are
computed exactly privately. Aggregate image/clinical terms are reported; the
patient-constant intercept is withheld from export because it is itself a
fitted coefficient. Image gradients do not depend on clinical values in this
additive model. These terms are not causal or SHAP contributions.

Layer: stage4.conv2 before GroupNorm/residual/ReLU, unchanged for both models.
Native tensor D/H/W feature stride is 4/16/16, centre offset zero. No automatic
earlier-layer fallback. Coarse/singleton dimensions and degenerate maps are
reported; interpolated maps add no spatial information. CAMs at this internal
residual/normalized layer carry no theorem of complete or exact explanation.
CTP: original sequential float32 mean baseline, then gradient/40 through one
frame graph at a time. Retain detached CPU per-frame feature maps and arithmetic
mean of their positive maps. Temporal mean maps differ from temporal occlusion.

## Quantitative decisions (review before any patient execution)

Primary anatomical metrics use feature-grid centres, sampled from verified
native reference masks. Native-grid projection for perturbation uses separable
linear interpolation at native_index/stride, clamped to the boundary feature
centre. Images are never resampled. Mask grids must have identical native shape,
RAS+ affine within 1e-5 mm and valid nonsingular geometry; do not repair failed
registration. Original audited label schema is background0/normal brain1/
penumbra2/core3. Brain support is labels1|2|3, used only for XAI analysis and
interventions, never baseline prediction. This reference-derived support is not
an independently validated whole-brain segmentation. Missing/invalid masks
produce explicit NA and denominators; no inferred brain mask is substituted.

Map diagnostics: nonfinite, all-zero/maximum<=1e-12 or range<=1e-12 is undefined.
No min-max normalization; use unnormalized positive maps for sums and ranking.
Regions: core, penumbra excluding core, union and brain outside union. Empty
ROI/brain/map denominator => NA with reason. Nuisance: outside brain, native
outermost one-voxel border and one-voxel six-connected brain boundary. Feature
grid ROI mass denominator is within-brain mass; nuisance denominator is whole
map mass. Label masks are nearest-neighbour centre-sampled only.

Spatial fidelity: disjoint native 4x16x16 D/H/W patches anchored at index0,
score by mean attribution over eligible brain voxels. Select ceil(.10*Nbrain)
brain voxels by descending patch score, ties lexicographic patch then voxel
order. Last patch is clipped lexicographically to meet the exact budget. Random
controls use the same patch partition with 10 seeded random permutations and
random within-patch order for the final partial patch, independent of attribution
and labels. All methods share the same controls per case. Budget masks have
exactly equal eligible volume; shapes need not be identical.

Replacement: separable Gaussian blur of the already normalized input, sigma
2 native voxels per axis, radius6, replicate boundaries; replace selected voxels
only. Blur is computed from the unperturbed frame, never iteratively. Outside
mask unchanged. Native spacing, kernel voxel/mm sizes, selected count and actual
changed count are private audit fields. A hard intervention boundary remains;
no claim of a physiological counterfactual. CTP applies one spatial mask to all
40 frames. Same fixed target sign; report baseline support minus perturbed
support, 10-control median, difference and targeted>random-median frequency.
Boundary support gets a separate fixed intervention where nonempty.

CTP temporal windows: size5, stride2 starts0,2,...,34 plus35. Replace only the
five frames by linear interpolation between nearest unaffected left/right
frames; edge windows repeat the sole nearest unaffected boundary frame. Keep
40 indices. Cache only 40 detached embeddings; recompute five replacement
embeddings/window, sum deltas into original mean, compare with brute-force
synthetic oracle. Float32 accumulation differences use logit atol1e-6/rtol1e-6.
Frame profile = arithmetic mean of scores of covering windows, all40 covered.
This measures frame-content sensitivity; no learned temporal-order/phase claim.

Sanity controls use independent deep model/state copies. Image-score-head
randomization replaces the eight PC coefficients with seeded normal coefficients
having the original coefficient RMS (zero RMS => explicit undefined); clinical
coefficients, PCA/scaler and intercept fixed. Cumulative upper-block control
also reinitializes stage4 convolutions with Kaiming fan_out/relu and GroupNorm
ones/zeros on a separate copy. Keep original prediction sign. Report rank
correlation (average tied ranks; constant => NA) and equal-budget top-mask Dice.
Unchanged original checkpoint digest is mandatory before and after analysis.
Between-seed explanation stability is NOT ASSESSED: no independent fits exist
in this source work. Reusable matched-array comparisons are not independent fits.

## Aggregation, privacy and execution

Patient is the unit. Patient-first reductions, paired-method differences only
on joint valid patients, total/valid/missing counts and fixed reason counts.
Seed630 percentile bootstrap of patient means, 2000 resamples, 95% intervals,
n<2 => interval NA. Descriptive uncertainty on reused n23, not independent
predictive inference. Temporal bands are pointwise, not simultaneous.

Execution modes are separate: read-only preflight, synthetic CUDA smoke,
NCCT development, CTP development, aggregate reader, report writer. New exclusive
run slugs under approved owner-only UiS runtime; no overwrite or automatic
submission. Resume only validated completed private per-case records with exact
source/protocol/checkpoint/dependency/membership identity; incomplete files fail.
Completed records contain quantitative metrics only (no input/embedding/map/
coefficient export); approved private per-frame maps may remain memory-only.
Public reader accepts an exact numeric aggregate schema, rejects unknown fields
and small-denominator summaries, never dumps raw run records or exception text.
Server verification and real runs are outstanding, not synthetic success claims.

XAI-first changes V4 evaluation order. Full five-fold OOF, outer/external
evaluation, independent-seed stability and clinical validation remain deferred.

## Independent-review engineering corrections (local, uncommitted)

See `stage7_review_followup.md` for exact equivalence, forward-pass accounting,
bounded instrumentation, native synthetic resource calibration, anatomical
header preflight and explicitly missing private-overlay support. All scientific
budgets, controls, windows, targets and layers remain unchanged. Reuse identical
scalar mask scores and detached baseline embeddings within an unchanged case
only; no cross-case cache/model fitting is added.

Before real CTP require a separate same-source CUDA/gzip-proxy resource artifact
covering all115 authorized development image-header dimensions. CPU/tiny smoke
results cannot satisfy it. Unresolved declared anatomical paths block execution;
invalid geometry is counted separately and retains NA semantics. No alternative
mask is silently selected. Optional private overlays remain unimplemented/off;
aggregate reporting contains no patient image.

## Stage7 CPU CTP frame cache (pending review)

See `stage7_ctp_cache_review.md`. Only input access changes: one read-only
sequential NIfTI stream per patient visit prepares40 bounded CPU frames, reused
through the unchanged model adapter and then released. No cohort or disk cache.
The fixed prepared-frame limit is8GiB; over-limit shapes fail before decoding.
Historical pre-CAM gates require115 feature cache builds plus23 separate XAI
builds. Resource schema2 charges all138, retains23,345 encoder forwards, the2x
margin and19.2h gate. Prior resource artifacts cannot authorize this path. All
methods, targets, controls, windows, cases and historical tolerances are unchanged.
