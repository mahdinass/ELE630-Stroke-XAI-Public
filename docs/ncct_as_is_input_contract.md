# Native-grid NCCT model-input adapter: as-is baseline proposal

## Operational direction

Sazidur's latest instruction means: **“Try the already preprocessed images as-is first.”** The primary baseline should initially retain each curated, registered NCCT volume's native voxel grid and intensity values. Patient-to-patient slice counts may differ. This adapter is only an in-memory axis and dtype conversion for one 3D volume; it is not a model, loader, training pipeline or approved final preprocessing method.

The active outcome cohort is **143 patients** with fixed five-fold patient assignments and fingerprint `3db8c4fd5fef0e4e`. The archived 142-patient split must not be used. The adapter does not open or validate either private manifest or fold file; dataset selection remains a separate contract.

## Array and coordinate contract

`ele630.native_ncct_input.prepare_native_ncct_input(native_data, native_affine)` accepts a real, finite 3D NumPy array with native NIfTI shape `(x, y, z)` and a valid voxel-to-world affine. It returns `NativeNCCTInput` with:

| Field | Meaning |
| --- | --- |
| `tensor` | Independent, contiguous float32 array of shape `(C, D, H, W) = (1, z, y, x)` |
| `native_shape` | Original `(x, y, z)` shape, without padding or cropping |
| `native_affine` | Separately preserved, read-only float64 copy of the original native voxel-to-world matrix; matrix values are unchanged |
| `native_axis_order`, `tensor_axis_order` | Explicit `("x", "y", "z")` and `("C", "z", "y", "x")` axis metadata |
| `tensor_nbytes` | Tensor storage bytes only; excludes source array and runtime overhead |

**Native voxel index `(x, y, z)` ↔ tensor index `(C=0, D=z, H=y, W=x)`.** To map a tensor location into physical RAS+ space, first restore native index order `(W, H, D) = (x, y, z)`, then apply `native_affine` to `(x, y, z, 1)`. Applying the native affine directly to `(D, H, W)` is incorrect. The transformation is reversible by taking channel 0 and transposing spatial axes back from `(z, y, x)` to `(x, y, z)`.

The only value change is declared conversion to float32, which may round higher-precision values. Inputs that would become nonfinite in float32 are rejected. Every spatial voxel is retained. There is no resampling, interpolation, crop, pad, resize, reorientation, new registration, HU window, clipping, intensity normalization, brain-mask derivation or background replacement. Native physical voxel spacing and anatomy remain represented by the preserved affine and native grid.

## Proposed later integration and open decisions

Processing **one volume at a time with batch size 1** and using **shape-independent model pooling** are proposed engineering choices; GPU memory use and architecture have not been validated. This adapter does not assemble batches or implement a PyTorch DataLoader.

The former 1 × 1 × 3 mm / 256 × 256 × 72 physical-grid candidate and optional resampling QC prototype are **parked, not deleted**. They are not prerequisites for this as-is baseline and do not define its preprocessing policy.

The final NCCT training HU window, any background or padding policy, GPU memory feasibility, model normalization architecture, and whether later physical resampling is needed remain **unresolved**. None is chosen by this adapter.

Synthetic local tests check exact voxel and coordinate mapping, reversible axes, variable shapes, float32 behavior and invalid inputs. The prior **58/58** UiS cpu20 suite result predates this adapter and does not verify it. No patient images, identifiers, patient-level tables or derived artifacts belong in this local work or Git; real-data work stays on UiS under the established private workflow.
