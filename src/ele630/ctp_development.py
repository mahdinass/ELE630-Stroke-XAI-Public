"""Fold-0 CTP access restricted to the existing inner development partitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .data_contract import ACTIVE_PROFILE, DatasetContract, OutcomePatient
from .ncct_splits import FoldPlan, SplitError, binary_outcome, plan_outer_fold


class _Repository(Protocol):
    def open_series(self, patient: OutcomePatient): ...


@dataclass(frozen=True)
class CTPSeriesSample:
    series: object = field(repr=False)
    target: int = field(repr=False)


class CTPDevelopmentData:
    """Build the immutable fold-0 plan and authorize access before image I/O.

    Supply RegisteredCTPRepository on UiS; synthetic tests supply a fake.
    Smoke selects the first sorted inner-training patient, without image or
    outcome-based selection, and prohibits every other patient image.
    """

    def __init__(self, contract: DatasetContract, repository: _Repository, *, mode: str):
        if type(mode) is not str or mode not in {"development", "smoke"}:
            raise ValueError("mode must be explicitly 'development' or 'smoke'")
        self._plan = plan_outer_fold(contract, 0, seed=630, validation_fraction=0.20)
        if len(contract.patients) == ACTIVE_PROFILE.patients and (
            len(self._plan.inner_train), len(self._plan.inner_validation), len(self._plan.outer_test)
        ) != (92, 23, 28):
            raise SplitError("active fold-0 development counts must be 92/23/28")
        if not callable(getattr(repository, "open_series", None)):
            raise TypeError("repository must provide open_series(patient)")
        self._repository = repository
        self._training = self._plan.inner_train[:1] if mode == "smoke" else self._plan.inner_train
        self._validation = () if mode == "smoke" else self._plan.inner_validation
        self._training_by_id = {p.patient_id: p for p in self._training}
        self._validation_by_id = {p.patient_id: p for p in self._validation}

    @property
    def plan(self) -> FoldPlan:
        return self._plan

    @property
    def training_patients(self) -> tuple[OutcomePatient, ...]:
        return self._training

    @property
    def validation_patients(self) -> tuple[OutcomePatient, ...]:
        return self._validation

    def _open(self, patient: OutcomePatient, allowed: dict[str, OutcomePatient]) -> CTPSeriesSample:
        if type(patient) is not OutcomePatient or allowed.get(patient.patient_id) != patient:
            raise SplitError("CTP image request is outside the authorized inner partition")
        canonical = allowed[patient.patient_id]
        target = binary_outcome(canonical)
        try:
            series = self._repository.open_series(canonical)
        except Exception:
            raise RuntimeError("CTP series could not be opened") from None
        return CTPSeriesSample(series, target)

    def open_training(self, patient: OutcomePatient) -> CTPSeriesSample:
        return self._open(patient, self._training_by_id)

    def open_validation(self, patient: OutcomePatient) -> CTPSeriesSample:
        return self._open(patient, self._validation_by_id)
