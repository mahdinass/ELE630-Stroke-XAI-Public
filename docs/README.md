# Implementation and reproducibility

## Input configuration

The CPU example and tests create their inputs in memory or in temporary
directories. The clinical study used 3D NCCT volumes, 40-frame registered CTP,
clinical records, case-level fold assignments and reference masks. The hospital
data, patient-level outputs and trained weights have restricted access.

The read-only data contract checker accepts explicit paths. Its process
environment equivalents are in [the example configuration](../.env.example).
The shell must export them before they will be used.

| Checker option | Environment variable | Input |
| --- | --- | --- |
| `--project-root` | `ELE630_PROJECT` | Local workspace |
| `--dataset-root` | `DATA_ROOT` | Authorized read-only dataset root |
| `--runtime-root` | `ELE630_RUNTIME` | Local runtime location |
| `--manifest` | `ELE630_MANIFEST` | Cohort inventory CSV |
| `--folds` | `ELE630_FOLDS` | Private fold-assignment CSV |

Use `python scripts/preflight/validate_active_dataset.py --help` for the
checker options. The active-study validator expects the original study
inventory. The synthetic tests exercise the same schema paths with fabricated
membership. Missing inputs cause a preflight failure.

## Methods

- [NCCT architecture](ncct_architecture_freeze_v1.md) and [input contract](ncct_as_is_input_contract.md)
- [CTP architecture and comparison rationale](ctp_architecture_freeze_v1.md)
- [Clinical feature and baseline freeze](clinical_feature_and_baseline_freeze_v1.md)
- [Training-only PCA and separate clinical–imaging fusion](stage5c3_clinical_imaging_fusion_protocol.md)
- [Quantitative Stage 7 XAI protocol](stage7_xai_protocol.md)
- [Frozen internal held-out evaluation protocol](final_heldout_protocol.md)

Spatial faithfulness selects exactly the prescribed voxel budget within
reference brain. The ten matched controls are draws from one fixed-seed
generator, with the same controls across CAM methods for a case.
Label-0 attribution and the brain-constrained perturbation answer different
questions. The latter cannot determine whether an image model relies on
label-0 voxels.

## Study provenance

The code is a sanitized snapshot of the implementation. The executed
historical source identifiers below refer to the restricted research archive.

| Evidence | Historical revision |
| --- | --- |
| Matched NCCT and CTP development fusion | `54f2891` |
| NCCT quantitative Stage 7 XAI | `abe39b4` |
| CTP quantitative Stage 7 XAI | `1f306e5` |
| Frozen internal test evaluated on 10 November 2026 | `9685bf0` |
| Private development-case figure workflow | `7f39cbc` |

Site-specific scheduler wrappers, exploratory audit entry points that write
case-level files, and private provenance records are kept in that archive.
Source-identity checks in the archived runners require the original source
history. The regression suite uses a hash-verified source-only fixture for
the equivalent Stage 7 comparison.

Historical model-artifact code expects trusted, provenance-verified objects.
The research GPU and clinical stacks were used under separate historical
contracts. The documented CPU environment is for fabricated inputs and does not
require any existing model artifact.
