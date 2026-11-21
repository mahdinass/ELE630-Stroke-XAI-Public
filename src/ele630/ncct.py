"""Admission skull-stripped NCCT access for validated outcome patients.

Loading applies only NIfTI's declared slope/intercept through nibabel. It does
not reorient, resample, crop, window or otherwise normalize the image.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .data_contract import DatasetContract, OutcomePatient
from .ncct_paths import NCCTError, NCCTResolver


@dataclass(frozen=True)
class NCCTMetadata:
    affine: np.ndarray
    header: nib.nifti1.Nifti1Header
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: str
    stored_dtype: str
    slope: float
    intercept: float
    sform_code: int
    qform_code: int


@dataclass(frozen=True)
class NCCTVolume(NCCTMetadata):
    data: np.ndarray
    data_dtype: str


class NCCTRepository(NCCTResolver):
    """Resolve and load only patients present in a validated DatasetContract."""

    def _open(self, patient: OutcomePatient) -> nib.spatialimages.SpatialImage:
        path = self.resolve(patient)
        try:
            return nib.load(str(path))
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise NCCTError("unreadable", "nibabel could not open the NCCT NIfTI") from exc

    @staticmethod
    def _metadata(image: nib.spatialimages.SpatialImage) -> NCCTMetadata:
        if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
            raise NCCTError("dimension", "admission NCCT must have exactly three spatial dimensions")

        affine = np.array(image.affine, dtype=np.float64, copy=True)
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
        if (
            affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or not np.isfinite(spacing).all()
            or any(x <= 0 for x in spacing)
            or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12
        ):
            raise NCCTError("metadata", "NCCT affine or voxel spacing is invalid")
        codes = nib.aff2axcodes(affine)
        if any(code is None for code in codes):
            raise NCCTError("metadata", "NCCT orientation cannot be determined")

        proxy = image.dataobj
        slope = float(getattr(proxy, "slope", 1.0) or 1.0)
        intercept = float(getattr(proxy, "inter", 0.0) or 0.0)
        return NCCTMetadata(
            affine=affine,
            header=image.header.copy(),
            shape=tuple(int(x) for x in image.shape),
            spacing_mm=spacing,
            orientation="".join(codes),
            stored_dtype=str(image.get_data_dtype()),
            slope=slope,
            intercept=intercept,
            sform_code=int(image.header["sform_code"]),
            qform_code=int(image.header["qform_code"]),
        )

    def read_header(self, patient: OutcomePatient) -> NCCTMetadata:
        """Inspect geometry without decoding the voxel array."""
        return self._metadata(self._open(patient))

    def load(self, patient: OutcomePatient) -> NCCTVolume:
        image = self._open(patient)
        metadata = self._metadata(image)
        try:
            # The proxy applies the NIfTI-declared scaling. No further intensity
            # transform or spatial operation occurs here.
            data = np.asanyarray(image.dataobj)
            if not (
                np.issubdtype(data.dtype, np.integer)
                or np.issubdtype(data.dtype, np.floating)
            ):
                raise NCCTError("metadata", "NCCT voxel dtype must be real numeric")
            finite = bool(np.isfinite(data).all())
        except NCCTError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise NCCTError("unreadable", "nibabel could not decode NCCT voxels") from exc
        if not finite:
            raise NCCTError("nonfinite", "NCCT contains NaN or infinite voxels")
        return NCCTVolume(**vars(metadata), data=data, data_dtype=str(data.dtype))


@dataclass(frozen=True)
class NCCTCaseResult:
    patient_id: str  # Private: write only to an ignored server-side table.
    status: str
    shape: tuple[int, int, int] | None = None
    orientation: str | None = None


@dataclass
class NCCTInventory:
    total: int
    headers_only: bool = False
    counts: Counter[str] = field(default_factory=Counter)
    shapes: Counter[tuple[int, int, int]] = field(default_factory=Counter)
    orientations: Counter[str] = field(default_factory=Counter)
    spacings: list[tuple[float, float, float]] = field(default_factory=list)
    extents: list[tuple[float, float, float]] = field(default_factory=list)
    minima: list[float] = field(default_factory=list)
    maxima: list[float] = field(default_factory=list)
    cases: list[NCCTCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.counts["ok"] == self.total


def scan_ncct(contract: DatasetContract, dataset_root: str | Path) -> NCCTInventory:
    """Inspect one NCCT at a time; keep only aggregate measurements in memory."""
    repository = NCCTRepository(contract, dataset_root)
    inventory = NCCTInventory(total=len(contract.patients))
    for patient in contract.patients:
        try:
            image = repository.load(patient)
        except NCCTError as exc:
            inventory.counts[exc.kind] += 1
            inventory.cases.append(NCCTCaseResult(patient.patient_id, exc.kind))
            continue
        inventory.counts["ok"] += 1
        inventory.cases.append(
            NCCTCaseResult(patient.patient_id, "ok", image.shape, image.orientation)
        )
        inventory.shapes[image.shape] += 1
        inventory.orientations[image.orientation] += 1
        inventory.spacings.append(image.spacing_mm)
        inventory.extents.append(
            tuple((size - 1) * spacing for size, spacing in zip(image.shape, image.spacing_mm))
        )
        inventory.minima.append(float(np.min(image.data)))
        inventory.maxima.append(float(np.max(image.data)))
        del image
    return inventory
