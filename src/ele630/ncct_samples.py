"""Lazy one-patient NCCT model inputs; repository access occurs only on load()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data_contract import DatasetContract, OutcomePatient
from .native_ncct_input import prepare_native_ncct_input
from .ncct_intensity import clip_scale_ncct
from .ncct_splits import SplitError, _check_contract, binary_outcome


class _Volume(Protocol):
    data: np.ndarray
    affine: np.ndarray


class _Repository(Protocol):
    def load(self, patient: OutcomePatient) -> _Volume: ...


@dataclass(frozen=True)
class NCCTSample:
    image: np.ndarray  # float32 (1, D, H, W), without the batch axis
    target: np.float32  # good=0, poor=1


class LazyNCCTSamples:
    """Compose the existing loader, native adapter and explicit intensity bounds.

    Pass ``NCCTRepository(contract, dataset_root)`` on UiS, or a fake repository
    for synthetic tests. This object retains no decoded image after load returns.
    """

    def __init__(self, contract: DatasetContract, repository: _Repository, *,
                 lower: float, upper: float):
        _check_contract(contract)
        if not callable(getattr(repository, "load", None)):
            raise TypeError("repository must provide load(patient)")
        # Validate the explicit configuration before accessing any patient image.
        clip_scale_ncct(np.zeros((1, 1, 1, 1), dtype=np.float32), lower=lower, upper=upper)
        self._patients = {patient.patient_id: patient for patient in contract.patients}
        self._repository = repository
        self._lower = lower
        self._upper = upper
        self._fingerprint = contract.fingerprint

    @property
    def input_record(self) -> dict:
        """Explicit, label-free recipe for private experiment provenance."""
        return {
            "representation": "native-NIfTI-declared-scaling-float32-zyx",
            "intensity": "clip-linear-scale",
            "lower": float(self._lower), "upper": float(self._upper),
            "spatial_resampling": False, "input_crop_pad": False,
            "background_replacement": False,
            "active_fingerprint": self._fingerprint,
        }

    def load(self, patient: OutcomePatient) -> NCCTSample:
        if type(patient) is not OutcomePatient:
            raise SplitError("sample request must be an OutcomePatient from the contract")
        canonical = self._patients.get(patient.patient_id)
        if canonical is None or canonical != patient:
            raise SplitError("sample request is not a patient in the validated contract")
        volume = self._repository.load(canonical)
        native = prepare_native_ncct_input(volume.data, volume.affine)
        image = clip_scale_ncct(native.tensor, lower=self._lower, upper=self._upper)
        if image.dtype != np.dtype(np.float32) or not np.isfinite(image).all():
            raise ValueError("NCCT sample must be finite float32")
        return NCCTSample(image=image, target=np.float32(binary_outcome(canonical)))
