"""Contract-restricted registered CTP path resolution (standard library only).

The derivatives directory and filename pattern are established by
scripts/audit/01_build_manifest.py and 05_nifti_header_audit.py.
"""

from __future__ import annotations

from pathlib import Path

from .data_contract import DatasetContract, OutcomePatient


CTP_PATTERN = "*_ses-01_space-ncct_ctp.nii.gz"


class CTPError(ValueError):
    """A registered CTP path, header or frame failed validation."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class CTPResolver:
    """Resolve one registered acute CTP for a validated outcome patient."""

    def __init__(self, contract: DatasetContract, dataset_root: str | Path):
        self._patients = {patient.patient_id: patient for patient in contract.patients}
        self._derivatives_root = Path(dataset_root) / "derivatives"

    def resolve(self, patient: OutcomePatient) -> Path:
        if not isinstance(patient, OutcomePatient) or self._patients.get(patient.patient_id) != patient:
            raise CTPError("ineligible", "patient is not in the validated outcome contract")
        pid = patient.patient_id
        if pid in {"", ".", ".."} or "/" in pid or "\\" in pid:
            raise CTPError("ineligible", "patient_id is not a safe directory name")
        patient_dir = self._derivatives_root / pid
        if patient_dir.is_symlink():
            raise CTPError("ineligible", "registered CTP patient directory is a symlink")
        if not patient_dir.is_dir():
            raise CTPError("missing", "registered CTP patient directory is missing")
        if not patient_dir.resolve().is_relative_to(self._derivatives_root.resolve()):
            raise CTPError("ineligible", "patient directory resolves outside the dataset root")
        matches = sorted(patient_dir.rglob(CTP_PATTERN))
        if not matches:
            raise CTPError("missing", "registered acute CTP is missing")
        if len(matches) != 1:
            raise CTPError("ambiguous", "multiple registered acute CTP candidates")
        candidate = matches[0]
        relative = candidate.relative_to(patient_dir)
        component = patient_dir
        for part in relative.parts:
            component = component / part
            if component.is_symlink():
                raise CTPError("ineligible", "registered CTP path contains a symlink")
        if not candidate.is_file():
            raise CTPError("missing", "registered CTP candidate is not a file")
        if not candidate.resolve().is_relative_to(patient_dir.resolve()):
            raise CTPError("ineligible", "CTP candidate resolves outside the patient directory")
        return candidate
