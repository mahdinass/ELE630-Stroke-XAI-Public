"""Contract-restricted admission NCCT path resolution (standard library only)."""

from __future__ import annotations

from pathlib import Path

from .data_contract import DatasetContract, OutcomePatient


NCCT_PATTERN = "*_ses-01_ncct_skull_stripped.nii.gz"


class NCCTError(ValueError):
    """An NCCT path or image failed validation; messages omit patient IDs."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class NCCTResolver:
    """Resolve files only for patients present in a validated outcome contract."""

    def __init__(self, contract: DatasetContract, dataset_root: str | Path):
        self._patients = {patient.patient_id: patient for patient in contract.patients}
        self._raw_root = Path(dataset_root) / "raw_data_nifti"

    def resolve(self, patient: OutcomePatient) -> Path:
        if not isinstance(patient, OutcomePatient) or self._patients.get(patient.patient_id) != patient:
            raise NCCTError("ineligible", "patient is not in the validated outcome contract")
        pid = patient.patient_id
        if pid in {".", ".."} or "/" in pid or "\\" in pid:
            raise NCCTError("ineligible", "patient_id is not a safe directory name")
        patient_dir = self._raw_root / pid
        if not patient_dir.is_dir():
            raise NCCTError("missing", "admission NCCT patient directory is missing")
        if not patient_dir.resolve().is_relative_to(self._raw_root.resolve()):
            raise NCCTError("ineligible", "patient directory resolves outside the dataset root")
        matches = sorted(patient_dir.rglob(NCCT_PATTERN))
        if not matches:
            raise NCCTError("missing", "admission skull-stripped NCCT is missing")
        if len(matches) != 1:
            raise NCCTError("ambiguous", "multiple admission skull-stripped NCCT candidates")
        if not matches[0].is_file():
            raise NCCTError("missing", "admission NCCT candidate is not a file")
        if not matches[0].resolve().is_relative_to(patient_dir.resolve()):
            raise NCCTError("ineligible", "NCCT candidate resolves outside the patient directory")
        return matches[0]
