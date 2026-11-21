"""Validate the private manifest and fixed outcome-CV assignment before modelling.

The input columns and fingerprint algorithm follow scripts/audit/01_build_manifest.py
and scripts/audit/24_outcome_split_strategy_audit.py. No split is generated here.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


class ContractError(ValueError):
    """Input data does not satisfy the outcome-modelling contract."""


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    patients: int
    good: int
    poor: int
    with_gt: int
    fold_sizes: tuple[int, ...]
    poor_per_fold: tuple[int, ...]
    fingerprint: str
    seed: int = 630
    archived_fingerprints: tuple[str, ...] = ("6a6a203ef27bab5b",)

    def __post_init__(self) -> None:
        if len(self.fold_sizes) != 5 or len(self.poor_per_fold) != 5:
            raise ValueError("Outcome-CV profile must define exactly five folds")


ACTIVE_PROFILE = ValidationProfile(
    name="post-restoration 2026-09-23",
    patients=143,
    good=98,
    poor=45,
    with_gt=142,
    fold_sizes=(28, 29, 29, 29, 28),
    poor_per_fold=(9, 9, 9, 9, 9),
    fingerprint="3db8c4fd5fef0e4e",
)


@dataclass(frozen=True)
class OutcomePatient:
    patient_id: str
    cohort: str
    outcome: str
    has_gt: bool
    fold: int


@dataclass(frozen=True)
class FoldSummary:
    fold: int
    patients: int
    good: int
    poor: int
    cohorts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DatasetContract:
    patients: tuple[OutcomePatient, ...]
    fold_summaries: tuple[FoldSummary, ...]
    good: int
    poor: int
    with_gt: int
    fingerprint: str
    seed: int


MANIFEST_COLUMNS = frozenset(
    {
        "patient_id", "cohort", "has_ncct_skull_stripped", "has_ctp_registered",
        "has_ctp_registered_mask", "mrs_3months", "mrs_binary", "eligible_fusion_mrs",
    }
)
FOLD_COLUMNS = frozenset(
    {"patient_id", "fold", "cohort", "mrs_binary", "stratum", "has_gt"}
)


def _read_csv(path: Path, required: frozenset[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            if not headers:
                raise ContractError(f"{label}: missing CSV header")
            if len(headers) != len(set(headers)) or any(not h for h in headers):
                raise ContractError(f"{label}: duplicate or blank CSV column name")
            missing = required - set(headers)
            if missing:
                raise ContractError(
                    f"{label}: missing required columns: {', '.join(sorted(missing))}"
                )
            rows = []
            for number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ContractError(f"{label} row {number}: malformed CSV field count")
                rows.append(row)
    except OSError as exc:
        raise ContractError(f"{label}: cannot read {path}: {exc.strerror or exc}") from exc
    except UnicodeError as exc:
        raise ContractError(f"{label}: file is not valid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise ContractError(f"{label}: malformed CSV: {exc}") from exc
    if not rows:
        raise ContractError(f"{label}: CSV contains no data rows")
    return rows


def _flag(value: str, label: str) -> bool:
    if value not in {"0", "1"}:
        raise ContractError(f"{label}: expected 0 or 1")
    return value == "1"


def _mrs(value: str, label: str) -> int | None:
    if value == "":
        return None
    if value.strip().lower() == "n/a":
        raise ContractError(
            f"{label}: raw clinical 'N/A' must be exported as a blank mRS cell "
            "by 01_build_manifest.py; regenerate the patient manifest"
        )
    try:
        number = int(value)
    except ValueError as exc:
        raise ContractError(f"{label}: expected integer 0–6 or blank") from exc
    if not 0 <= number <= 6:
        raise ContractError(f"{label}: expected integer 0–6 or blank")
    return number


def _fingerprint(assignments: dict[str, int]) -> str:
    # Exact algorithm used by 24_outcome_split_strategy_audit.py.
    text = "\n".join(sorted(f"{patient_id}:{fold}" for patient_id, fold in assignments.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_and_validate(
    manifest_path: str | Path,
    folds_path: str | Path,
    profile: ValidationProfile = ACTIVE_PROFILE,
) -> DatasetContract:
    """Load existing assignments; reject stale or inconsistent inputs.

    Errors identify file/row/field without printing private patient identifiers.
    """
    manifest_rows = _read_csv(Path(manifest_path), MANIFEST_COLUMNS, "manifest")
    fold_rows = _read_csv(Path(folds_path), FOLD_COLUMNS, "folds")

    eligible: dict[str, tuple[str, str, bool]] = {}
    seen_manifest: set[str] = set()
    for number, row in enumerate(manifest_rows, start=2):
        location = f"manifest row {number}"
        pid = row["patient_id"].strip()
        if not pid:
            raise ContractError(f"{location}: patient_id is blank")
        if pid in seen_manifest:
            raise ContractError(f"{location}: duplicate patient_id")
        seen_manifest.add(pid)
        cohort = row["cohort"].strip()
        if not cohort:
            raise ContractError(f"{location}: cohort is blank")
        ncct = _flag(row["has_ncct_skull_stripped"], f"{location} has_ncct_skull_stripped")
        ctp = _flag(row["has_ctp_registered"], f"{location} has_ctp_registered")
        gt = _flag(row["has_ctp_registered_mask"], f"{location} has_ctp_registered_mask")
        mrs = _mrs(row["mrs_3months"], f"{location} mrs_3months")
        outcome = row["mrs_binary"].strip()
        expected_outcome = "" if mrs is None else ("good" if mrs <= 2 else "poor")
        if outcome != expected_outcome:
            raise ContractError(
                f"{location}: mrs_binary must be {expected_outcome!r} for mrs_3months"
            )
        calculated_eligible = ncct and ctp and mrs is not None
        recorded_eligible = _flag(row["eligible_fusion_mrs"], f"{location} eligible_fusion_mrs")
        if recorded_eligible != calculated_eligible:
            raise ContractError(
                f"{location}: eligible_fusion_mrs disagrees with NCCT, registered CTP and mRS"
            )
        if calculated_eligible:
            eligible[pid] = (cohort, outcome, gt)

    assignments: dict[str, int] = {}
    seen_folds: set[int] = set()
    for number, row in enumerate(fold_rows, start=2):
        location = f"folds row {number}"
        pid = row["patient_id"].strip()
        if not pid:
            raise ContractError(f"{location}: patient_id is blank")
        try:
            fold = int(row["fold"])
        except ValueError as exc:
            raise ContractError(f"{location}: fold must be an integer 0–4") from exc
        if not 0 <= fold < 5:
            raise ContractError(f"{location}: fold must be an integer 0–4")
        if pid in assignments:
            reason = "duplicate patient/fold assignment" if assignments[pid] == fold else "patient assigned to multiple folds"
            raise ContractError(f"{location}: {reason}")
        if pid not in eligible:
            raise ContractError(f"{location}: unknown or ineligible patient in outcome folds")
        cohort, outcome, gt = eligible[pid]
        if row["cohort"] != cohort or row["mrs_binary"] != outcome:
            raise ContractError(f"{location}: cohort or outcome disagrees with manifest")
        if _flag(row["has_gt"], f"{location} has_gt") != gt:
            raise ContractError(f"{location}: has_gt disagrees with manifest")
        if row["stratum"] != f"c{cohort}_{outcome}":
            raise ContractError(f"{location}: stratum disagrees with cohort and outcome")
        assignments[pid] = fold
        seen_folds.add(fold)

    fingerprint = _fingerprint(assignments)
    if fingerprint in profile.archived_fingerprints:
        raise ContractError(
            f"folds: archived outcome-CV fingerprint {fingerprint}; use active {profile.fingerprint}"
        )
    missing = eligible.keys() - assignments.keys()
    if missing:
        raise ContractError(f"folds: {len(missing)} eligible patient(s) have no fold")
    if seen_folds != set(range(5)):
        raise ContractError(f"folds: expected exactly five folds numbered 0–4; found {sorted(seen_folds)}")
    if fingerprint != profile.fingerprint:
        raise ContractError(
            f"folds: fingerprint {fingerprint} does not match active {profile.fingerprint}; "
            "verify the server-side fold file"
        )

    patients = tuple(
        OutcomePatient(pid, *eligible[pid], assignments[pid])
        for pid in sorted(eligible)
    )
    good = sum(patient.outcome == "good" for patient in patients)
    poor = len(patients) - good
    with_gt = sum(patient.has_gt for patient in patients)
    for label, actual, expected in (
        ("eligible outcome patients", len(patients), profile.patients),
        ("good outcomes", good, profile.good),
        ("poor outcomes", poor, profile.poor),
        ("patients with GT", with_gt, profile.with_gt),
    ):
        if actual != expected:
            raise ContractError(f"active profile: {label} = {actual}; expected {expected}")

    summaries = []
    observed_strata = {(patient.cohort, patient.outcome) for patient in patients}
    for fold in range(5):
        subset = [patient for patient in patients if patient.fold == fold]
        counts = Counter(patient.cohort for patient in subset)
        fold_good = sum(patient.outcome == "good" for patient in subset)
        fold_poor = len(subset) - fold_good
        if len(subset) != profile.fold_sizes[fold]:
            raise ContractError(
                f"active profile: fold {fold} size = {len(subset)}; expected {profile.fold_sizes[fold]}"
            )
        if fold_poor != profile.poor_per_fold[fold]:
            raise ContractError(
                f"active profile: fold {fold} poor outcomes = {fold_poor}; "
                f"expected {profile.poor_per_fold[fold]}"
            )
        fold_strata = {(patient.cohort, patient.outcome) for patient in subset}
        if fold_strata != observed_strata:
            raise ContractError(
                f"active profile: fold {fold} does not represent every cohort × outcome stratum"
            )
        summaries.append(FoldSummary(fold, len(subset), fold_good, fold_poor, tuple(sorted(counts.items()))))
    return DatasetContract(patients, tuple(summaries), good, poor, with_gt, fingerprint, profile.seed)
