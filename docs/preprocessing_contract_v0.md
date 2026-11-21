# ELE630 NCCT preprocessing contract v0 — proposal, not a training freeze

## Verified inputs and boundaries

- The active outcome benchmark is 143 matched patients with one fixed, patient-level five-fold CV (seed 630; fingerprint `3db8c4fd5fef0e4e`). Use the validated `DatasetContract` for patient selection. GT availability is not an NCCT input requirement.
- Admission skull-stripped NCCT is 3D in patient-specific physical grids; observed orientation is LPS. Registered CTP shares each patient's NCCT grid, but grids vary between patients. Reported active-cohort NCCT physical spans reach about 242.525 mm in-plane and 174 mm axially.
- `NCCTRepository.load()` exposes the original NIfTI affine, header and voxel data after only NIfTI-declared slope/intercept scaling. It checks finite voxels and does not reorient, resample, crop, pad, window or normalize.
- The prior **0–80 HU** display window was used for visual QC. It is not an approved training intensity policy.
- Slurm `cpu20` is available for CPU work, and its separate CPU environment is verified. Job 481950 passed the 143/143 NCCT header-only inventory; job 481954 passed one real NCCT voxel-loader test. A new full 143-volume voxel inventory has not been run.
- Patient images and patient-level tables must never be tracked in Git, committed, pushed, copied off UiS, or supplied to Codex. Private, Git-ignored server-side `outputs/tables/` are permitted.

## Initial geometry candidate to evaluate

| Item | Candidate | Interpretation |
| --- | --- | --- |
| Target voxel spacing | **1.0 × 1.0 × 3.0 mm** | Explicit candidate, not a finalized training grid. |
| Output array | **256 × 256 × 72** | Explicit candidate, not a finalized training grid. |
| Voxel-center span | **255 × 255 × 213 mm** | `(shape − 1) × spacing`, between the outermost voxel centers. |
| Voxel-edge field of view | **256 × 256 × 216 mm** | `shape × spacing`; distinct from center-to-center span. |

For an axis-aligned, geometrically centered scan whose spans are exactly 242.525 × 242.525 × 174 mm, the candidate leaves about **6.24 mm per side in-plane** and **19.5 mm per side axially** between the source and output *outer voxel centers*. This arithmetic does **not** prove brain coverage. Obliquity, translation of anatomy within a source array, and brain/support shape can cause cropping even if native-axis spans are smaller than the proposed field of view.

Resampling to 1 mm in-plane may downsample scans acquired below 1 mm and discard detail; it may upsample coarser scans without adding information. The 3 mm axial proposal must be checked against actual header geometry and the final interpolation/anti-aliasing policy. Model tensor size alone is not evidence that anatomy is preserved.

## Geometry-planning convention implemented in v0

`ele630.geometry.plan_output_grid(input_shape, input_affine, target_spacing_mm=..., target_shape=...)` requires both candidate parameters explicitly and returns only a grid plan. It does **not** read or interpolate voxel data.

- Input affine is the nibabel/NIfTI voxel-to-world **RAS+ millimetre** matrix; the source voxel-center index is `(shape − 1) / 2`, including half indices for even dimensions.
- The output voxel axes are fixed **LPS** directions in RAS+ world coordinates: negative world X, negative world Y, positive world Z. The output affine's diagonal is therefore `(-sx, -sy, +sz)`.
- The output voxel-center index `(target_shape − 1) / 2` maps to the same world point as the source geometric center. This is a geometric-center proposal, not a measured brain-center alignment.
- `input_corner_output_bounds()` maps the eight source voxel-center corners into continuous output indices. `contains_input_voxel_centers()` checks whether the *entire source center lattice* fits. It is a conservative geometric diagnostic; it does not identify brain tissue, detect interpolation artefacts, or certify anatomical coverage.

## Voxel-resampling prototype implemented

`ele630.resampling.resample_to_grid(data, input_affine, plan, outside_value=...)` accepts a 3D NumPy array and a matching `GridPlan`, then returns a `ResampledVolume` with a float64 `data` array and the plan's `output_affine` in its `affine` field. It composes the output affine with the inverse input affine, so `scipy.ndimage.affine_transform` samples at the correct physical coordinates for translated, flipped, and oblique sources. The prototype uses linear interpolation and requires an explicit finite constant for out-of-image samples. That constant is a caller choice, **not** a default NCCT background or padding policy. This function is separate from `NCCTRepository.load()` and does not window, normalize, mask, or write images.

## Open decisions before training

1. Confirm the **candidate** 1 × 1 × 3 mm, 256 × 256 × 72 target grid, fixed-axis convention and center choice using the authorized full-cohort geometry summary and representative visual QC. If image support is off-center, compare geometric centering with a brain-support center derived from admission NCCT alone.
2. Measure clipping of admission-NCCT brain support after proposed resampling/crop/pad. Check every output face, the fraction of support lost, and representative high-risk scans. Define acceptance thresholds *before* final model evaluation. Do not use GT or outcome labels to position the grid.
3. Decide anti-aliasing for downsampling, final padding value and out-of-field semantics beyond this prototype's constant fill. All **40 registered CTP frames** must later receive spatial processing consistent with admission NCCT; any masks used for evaluation need nearest-neighbour interpolation.
4. Define and freeze a brain-relevant NCCT HU window/intensity policy from the real dataset audit and scientific rationale. The QC display window is not automatically suitable. Any fitted normalization must use only outer-training data in each fold.
5. Decide whether scan-level anomalies require exclusion or a documented handling rule, without changing the fixed active folds silently.

The repository currently has **no dependency manifest**. The loader uses NumPy and nibabel; this prototype additionally needs `scipy.ndimage`. The verified UiS `cpu20` `.venv_cpu` has NumPy 1.26.4, nibabel 5.4.2 and SciPy 1.11.4. The available local Windows test runtime has NumPy but no SciPy or nibabel. A versioned dependency decision remains open; no packages were installed for this prototype.

## Acceptance tests and next gate

- Synthetic tests must verify output-world coordinates, exact even-grid centering, deterministic affines, axis-aligned and oblique coverage checks, and rejection of invalid geometry. Resampling tests must also verify voxel values under identity, translation, LPS/RAS axis flips and oblique transforms, plus output affine, explicit outside fill and input mismatch rejection.
- Before adopting candidate parameters, compare input/output physical bounds and brain-support retention across all 143 eligible cases on an **authorized** compute host. Review aggregate failures and selected private server-side QC images; keep patient-level results out of tracked source.
- Only after geometry and intensity decisions are documented should a fold-safe voxel-resampling pipeline and NCCT baseline be implemented. This document does not authorize full-cohort processing or training.
