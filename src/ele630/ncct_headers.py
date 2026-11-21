"""Lightweight NIfTI-1 NCCT header inventory (no voxel reads)."""

from __future__ import annotations

import gzip
import itertools
import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .data_contract import DatasetContract
from .ncct_paths import NCCTError, NCCTResolver


@dataclass(frozen=True)
class NCCTHeader:
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: str


def read_ncct_header(path: Path) -> NCCTHeader:
    """Read only the first 348 uncompressed bytes; require an explicit sform.

    The existing audit reports sform as the affine source for all available
    NCCTs. Cases without an sform are flagged for full nibabel inspection.
    """
    try:
        with gzip.open(path, "rb") as stream:
            header = stream.read(348)
        if len(header) != 348:
            raise NCCTError("unreadable", "NCCT NIfTI-1 header is truncated")
        if struct.unpack_from("<i", header)[0] == 348:
            endian = "<"
        elif struct.unpack_from(">i", header)[0] == 348:
            endian = ">"
        else:
            raise NCCTError("unreadable", "NCCT does not have a NIfTI-1 header")
        if header[344:348] != b"n+1\x00":
            raise NCCTError("unreadable", "NCCT has invalid NIfTI-1 magic")
        dims = struct.unpack_from(endian + "8h", header, 40)
        if dims[0] != 3 or any(size <= 0 for size in dims[1:4]):
            raise NCCTError("dimension", "admission NCCT must have exactly three spatial dimensions")
        shape = tuple(int(size) for size in dims[1:4])
        pixdim = struct.unpack_from(endian + "8f", header, 76)
        spacing = tuple(abs(float(value)) for value in pixdim[1:4])
        sform_code = struct.unpack_from(endian + "h", header, 254)[0]
        if sform_code <= 0:
            raise NCCTError("metadata", "NCCT has no sform; use full nibabel inspection")
        rows = [struct.unpack_from(endian + "4f", header, offset) for offset in (280, 296, 312)]
        matrix = [[float(rows[i][j]) for j in range(3)] for i in range(3)]
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, k = matrix[2]
        determinant = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
        if (
            not all(math.isfinite(value) and value > 0 for value in spacing)
            or not all(math.isfinite(value) for row in rows for value in row)
            or not math.isfinite(determinant)
            or abs(determinant) < 1e-12
        ):
            raise NCCTError("metadata", "NCCT sform or voxel spacing is invalid")
        # Same best-axis-permutation convention as 05_nifti_header_audit.py.
        permutation = max(
            itertools.permutations(range(3)),
            key=lambda p: sum(abs(matrix[p[axis]][axis]) for axis in range(3)),
        )
        positive = "RAS"
        negative = "LPI"
        orientation = "".join(
            positive[world] if matrix[world][axis] >= 0 else negative[world]
            for axis, world in enumerate(permutation)
        )
        return NCCTHeader(shape, spacing, orientation)
    except (OSError, EOFError, struct.error) as exc:
        raise NCCTError("unreadable", "NCCT NIfTI-1 header cannot be read") from exc


@dataclass(frozen=True)
class HeaderCaseResult:
    patient_id: str
    status: str
    shape: tuple[int, int, int] | None = None
    orientation: str | None = None


@dataclass
class HeaderInventory:
    total: int
    headers_only: bool = True
    counts: Counter[str] = field(default_factory=Counter)
    shapes: Counter[tuple[int, int, int]] = field(default_factory=Counter)
    orientations: Counter[str] = field(default_factory=Counter)
    spacings: list[tuple[float, float, float]] = field(default_factory=list)
    extents: list[tuple[float, float, float]] = field(default_factory=list)
    minima: list[float] = field(default_factory=list)
    maxima: list[float] = field(default_factory=list)
    cases: list[HeaderCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.counts["ok"] == self.total


def scan_ncct_headers(contract: DatasetContract, dataset_root: str | Path) -> HeaderInventory:
    resolver = NCCTResolver(contract, dataset_root)
    inventory = HeaderInventory(len(contract.patients))
    for patient in contract.patients:
        try:
            path = resolver.resolve(patient)
            image = read_ncct_header(path)
        except NCCTError as exc:
            inventory.counts[exc.kind] += 1
            inventory.cases.append(HeaderCaseResult(patient.patient_id, exc.kind))
            continue
        inventory.counts["ok"] += 1
        inventory.cases.append(HeaderCaseResult(patient.patient_id, "ok", image.shape, image.orientation))
        inventory.shapes[image.shape] += 1
        inventory.orientations[image.orientation] += 1
        inventory.spacings.append(image.spacing_mm)
        inventory.extents.append(
            tuple((size - 1) * spacing for size, spacing in zip(image.shape, image.spacing_mm))
        )
    return inventory
