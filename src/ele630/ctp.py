"""Read-only, per-series access to registered acute CTP for outcome patients.

Header grid agreement with NCCT does not prove visually correct registration.
Frame indices are acquisition indices, not aligned bolus phases across patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .ctp_input import CTPFrameInput, CTPInputError, prepare_ctp_frame_input, validate_frame_index
from .ctp_paths import CTPError, CTPResolver
from .data_contract import DatasetContract, OutcomePatient
from .geometry import GeometryError, _affine
from .ncct import NCCTError, NCCTRepository


@dataclass(frozen=True)
class CTPMetadata:
    spatial_shape: tuple[int, int, int]
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    frame_count: int
    temporal_spacing: float
    temporal_unit: str
    temporal_offset: float
    stored_dtype: str


@dataclass(frozen=True)
class RegisteredCTPFrame:
    input: CTPFrameInput
    metadata: CTPMetadata


@dataclass(frozen=True)
class RegisteredCTPSeries:
    """One validated CTP series; frame reads reuse its NIfTI image proxy."""

    metadata: CTPMetadata
    _proxy: nib.arrayproxy.ArrayProxy = field(repr=False)

    def load_frame(self, frame_index: int) -> RegisteredCTPFrame:
        """Decode one frame from this series, without another patient lookup."""
        try:
            frame_index = validate_frame_index(frame_index)
        except CTPInputError as exc:
            raise CTPError("frame", str(exc)) from exc
        try:
            frame = np.asanyarray(self._proxy[:, :, :, frame_index])
        except (OSError, ValueError, TypeError, IndexError) as exc:
            raise CTPError("unreadable", "nibabel could not decode the requested CTP frame") from exc
        try:
            prepared = prepare_ctp_frame_input(frame, self.metadata.affine, frame_index)
        except CTPInputError as exc:
            raise CTPError("voxels", str(exc)) from exc
        return RegisteredCTPFrame(prepared, self.metadata)


class RegisteredCTPRepository(CTPResolver):
    """Open matching native-grid CTP/NCCT headers and read requested frames."""

    def __init__(self, contract: DatasetContract, dataset_root: str | Path):
        super().__init__(contract, dataset_root)
        self._ncct = NCCTRepository(contract, dataset_root)

    def _open(self, patient: OutcomePatient) -> nib.spatialimages.SpatialImage:
        path = self.resolve(patient)  # Patient membership is checked before file access.
        try:
            return nib.load(str(path))
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise CTPError("unreadable", "nibabel could not open registered CTP NIfTI") from exc

    def _metadata(self, image: nib.spatialimages.SpatialImage, patient: OutcomePatient) -> CTPMetadata:
        if len(image.shape) != 4 or any(size <= 0 for size in image.shape):
            raise CTPError("dimension", "registered CTP must have exactly four nonempty dimensions")
        if image.shape[3] != 40:
            raise CTPError("dimension", "registered CTP must have exactly 40 frames")
        if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
            raise CTPError("metadata", "registered CTP storage dtype must be float32")
        try:
            affine = np.array(_affine(image.affine), dtype=np.float64, copy=True)
        except GeometryError as exc:
            raise CTPError("metadata", "registered CTP spatial affine is invalid") from exc
        zooms = tuple(float(z) for z in image.header.get_zooms()[:4])
        if len(zooms) != 4 or not np.isfinite(zooms).all() or any(z <= 0 for z in zooms):
            raise CTPError("metadata", "registered CTP spatial or temporal spacing is invalid")
        if any(code is None for code in nib.aff2axcodes(affine)):
            raise CTPError("metadata", "registered CTP orientation cannot be determined")
        temporal_offset = float(image.header["toffset"])
        if not np.isfinite(temporal_offset):
            raise CTPError("metadata", "registered CTP temporal offset is invalid")
        try:
            ncct = self._ncct.read_header(patient)
        except NCCTError as exc:
            raise CTPError("ncct", "matching NCCT header could not be validated") from exc
        if tuple(image.shape[:3]) != ncct.shape:
            raise CTPError("grid", "registered CTP and NCCT spatial shapes disagree")
        # Match the pre-existing header audit's absolute tolerances.
        if not np.allclose(zooms[:3], ncct.spacing_mm, rtol=0, atol=1e-5):
            raise CTPError("grid", "registered CTP and NCCT spatial spacings disagree")
        if not np.allclose(affine, ncct.affine, rtol=0, atol=1e-4):
            raise CTPError("grid", "registered CTP and NCCT spatial affines disagree")
        affine.setflags(write=False)
        return CTPMetadata(
            tuple(int(s) for s in image.shape[:3]), affine, zooms[:3], 40,
            zooms[3], image.header.get_xyzt_units()[1], temporal_offset,
            str(image.get_data_dtype()),
        )

    def read_header(self, patient: OutcomePatient) -> CTPMetadata:
        """Validate CTP/NCCT grid metadata without decoding CTP voxels."""
        image = self._open(patient)
        return self._metadata(image, patient)

    def open_series(self, patient: OutcomePatient) -> RegisteredCTPSeries:
        """Resolve and validate one patient once, then retain its image proxy."""
        image = self._open(patient)
        metadata = self._metadata(image, patient)
        return RegisteredCTPSeries(metadata, image.dataobj)

    def load_frame(self, patient: OutcomePatient, frame_index: int) -> RegisteredCTPFrame:
        """Backward-compatible single-frame access with index-first validation."""
        try:
            frame_index = validate_frame_index(frame_index)
        except CTPInputError as exc:
            raise CTPError("frame", str(exc)) from exc
        return self.open_series(patient).load_frame(frame_index)
