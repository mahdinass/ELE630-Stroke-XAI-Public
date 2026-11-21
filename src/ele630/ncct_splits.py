"""Pure inner splits inside the immutable, validated five-fold outcome contract."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .data_contract import ACTIVE_PROFILE, DatasetContract, OutcomePatient, _fingerprint


class SplitError(ValueError):
    """The supplied contract or requested inner split is unusable."""


def binary_outcome(patient: OutcomePatient) -> int:
    """mRS 0–2 (good) = 0; mRS 3–6 (poor) = 1."""
    if type(patient) is not OutcomePatient or patient.outcome not in {"good", "poor"}:
        raise SplitError("patient outcome must be 'good' or 'poor'")
    return int(patient.outcome == "poor")


@dataclass(frozen=True)
class FoldPlan:
    outer_fold: int
    seed: int
    validation_fraction: float
    inner_train: tuple[OutcomePatient, ...]
    inner_validation: tuple[OutcomePatient, ...]
    outer_test: tuple[OutcomePatient, ...]
    cohort_stratified_outcomes: tuple[str, ...]


def validate_fold_plan(plan: FoldPlan) -> None:
    """Reject partition overlap or a changed outer-test fold before training."""
    if type(plan) is not FoldPlan or type(plan.outer_fold) is not int or plan.outer_fold not in range(5):
        raise SplitError("a valid FoldPlan with outer fold 0–4 is required")
    partitions = (plan.inner_train, plan.inner_validation, plan.outer_test)
    if any(not part or any(type(p) is not OutcomePatient for p in part) for part in partitions):
        raise SplitError("all three plan partitions must contain outcome patients")
    ids = [[p.patient_id for p in part] for part in partitions]
    if sum(map(len, ids)) != len(set().union(*map(set, ids))):
        raise SplitError("plan partitions contain duplicate or overlapping patients")
    if any(p.fold == plan.outer_fold for p in plan.inner_train + plan.inner_validation):
        raise SplitError("outer-test patient appears in inner development")
    if any(p.fold != plan.outer_fold for p in plan.outer_test):
        raise SplitError("outer-test partition disagrees with original folds")
    if any({p.outcome for p in part} != {"good", "poor"} for part in partitions):
        raise SplitError("each partition must contain both outcome classes")
    for part in partitions:
        for patient in part:
            binary_outcome(patient)


def inner_train_positive_weight(plan: FoldPlan) -> float:
    """Negative/positive ratio from inner train only; never inspect test labels."""
    validate_fold_plan(plan)
    positives = sum(binary_outcome(p) for p in plan.inner_train)
    negatives = len(plan.inner_train) - positives
    if not positives or not negatives:
        raise SplitError("inner train must have both classes for weighting")
    return negatives / positives


def _check_contract(contract: DatasetContract) -> None:
    if type(contract) is not DatasetContract or not isinstance(contract.patients, tuple):
        raise SplitError("a validated DatasetContract is required")
    patients = contract.patients
    if not patients or any(type(p) is not OutcomePatient for p in patients):
        raise SplitError("contract must contain outcome patients")
    if any(not isinstance(p.patient_id, str) or not p.patient_id or
           not isinstance(p.cohort, str) or not p.cohort or
           type(p.fold) is not int or p.fold not in range(5) or
           type(p.has_gt) is not bool for p in patients):
        raise SplitError("contract contains an invalid patient field or outer fold")
    ids = [p.patient_id for p in patients]
    if len(set(ids)) != len(ids):
        raise SplitError("contract contains duplicate patients")
    for patient in patients:
        binary_outcome(patient)
    if {p.fold for p in patients} != set(range(5)):
        raise SplitError("contract must contain all five original outer folds")
    counts = Counter(p.outcome for p in patients)
    if (contract.good, contract.poor, contract.with_gt) != (
        counts["good"], counts["poor"], sum(p.has_gt for p in patients)
    ):
        raise SplitError("contract aggregate counts disagree with patients")
    if type(contract.seed) is not int or contract.seed != ACTIVE_PROFILE.seed:
        raise SplitError("contract outer-fold seed must be 630")
    if contract.fingerprint in ACTIVE_PROFILE.archived_fingerprints:
        raise SplitError("archived outer-fold fingerprint is prohibited")
    if contract.fingerprint != _fingerprint({p.patient_id: p.fold for p in patients}):
        raise SplitError("contract fingerprint disagrees with original outer assignments")
    if len(patients) == ACTIVE_PROFILE.patients and (
        contract.fingerprint != ACTIVE_PROFILE.fingerprint or
        (contract.good, contract.poor, contract.with_gt) !=
        (ACTIVE_PROFILE.good, ACTIVE_PROFILE.poor, ACTIVE_PROFILE.with_gt) or
        tuple(sum(p.fold == fold for p in patients) for fold in range(5)) != ACTIVE_PROFILE.fold_sizes or
        tuple(sum(p.fold == fold and p.outcome == "poor" for p in patients)
              for fold in range(5)) != ACTIVE_PROFILE.poor_per_fold
    ):
        raise SplitError("143-patient contract disagrees with the active benchmark")
    if len(contract.fold_summaries) != 5 or {s.fold for s in contract.fold_summaries} != set(range(5)):
        raise SplitError("contract must have five fold summaries")
    for summary in contract.fold_summaries:
        subset = [p for p in patients if p.fold == summary.fold]
        if (summary.patients, summary.good, summary.poor, summary.cohorts) != (
            len(subset),
            sum(p.outcome == "good" for p in subset),
            sum(p.outcome == "poor" for p in subset),
            tuple(sorted(Counter(p.cohort for p in subset).items())),
        ):
            raise SplitError("fold summary disagrees with original outer assignments")


def _validation_count(size: int, fraction: float) -> int:
    return max(1, min(size - 1, round(size * fraction)))


def _cohort_allocations(groups: dict[str, list[OutcomePatient]], desired: int,
                        fraction: float) -> dict[str, int] | None:
    # Preserve each represented cohort in both partitions only when feasible.
    if len(groups) < 2 or any(len(group) < 2 for group in groups.values()):
        return None
    if not len(groups) <= desired <= sum(len(group) - 1 for group in groups.values()):
        return None
    allocation = {cohort: 1 for cohort in groups}
    for _ in range(desired - len(groups)):
        choices = [cohort for cohort in groups if allocation[cohort] < len(groups[cohort]) - 1]
        selected = max(sorted(choices), key=lambda c: len(groups[c]) * fraction - allocation[c])
        allocation[selected] += 1
    return allocation


def plan_outer_fold(
    contract: DatasetContract,
    outer_fold: int,
    *,
    seed: int = 630,
    validation_fraction: float = 0.20,
) -> FoldPlan:
    """Select the existing test fold; stratify only the other four folds.

    Each outcome receives its rounded validation share, clamped to leave both
    classes in train and validation. Cohort-within-outcome allocation is used
    only if it can meet that target while leaving each cohort in both sets;
    otherwise that outcome is stratified by outcome alone. Sorting before a
    local seeded shuffle makes membership independent of manifest row order.
    """
    _check_contract(contract)
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise SplitError("outer_fold must be an integer from 0 to 4")
    if type(seed) is not int or seed < 0:
        raise SplitError("inner seed must be a nonnegative integer")
    if (isinstance(validation_fraction, bool) or
            not isinstance(validation_fraction, (int, float)) or
            not math.isfinite(validation_fraction) or
            not 0 < validation_fraction < 1):
        raise SplitError("validation_fraction must be finite and between 0 and 1")

    test = tuple(p for p in contract.patients if p.fold == outer_fold)
    development = tuple(p for p in contract.patients if p.fold != outer_fold)
    if not test or {p.outcome for p in test} != {"good", "poor"}:
        raise SplitError("outer test fold must contain both outcome classes")
    rng = random.Random(seed + outer_fold)
    train: list[OutcomePatient] = []
    validation: list[OutcomePatient] = []
    cohort_stratified: list[str] = []
    for outcome in ("good", "poor"):
        members = [p for p in development if p.outcome == outcome]
        if len(members) < 2:
            raise SplitError("inner development needs at least two patients per outcome")
        desired = _validation_count(len(members), validation_fraction)
        groups: dict[str, list[OutcomePatient]] = defaultdict(list)
        for patient in members:
            groups[patient.cohort].append(patient)
        allocation = _cohort_allocations(groups, desired, validation_fraction)
        if allocation is None:
            groups = {"all": members}
            allocation = {"all": desired}
        else:
            cohort_stratified.append(outcome)
        for cohort in sorted(groups):
            shuffled = sorted(groups[cohort], key=lambda p: p.patient_id)
            rng.shuffle(shuffled)
            validation.extend(shuffled[:allocation[cohort]])
            train.extend(shuffled[allocation[cohort]:])

    train_ids = {p.patient_id for p in train}
    validation_ids = {p.patient_id for p in validation}
    test_ids = {p.patient_id for p in test}
    if (len(train_ids) != len(train) or len(validation_ids) != len(validation) or
            len(test_ids) != len(test) or train_ids & validation_ids or
            train_ids & test_ids or validation_ids & test_ids or
            train_ids | validation_ids | test_ids != {p.patient_id for p in contract.patients} or
            any(p.fold == outer_fold for p in train + validation) or
            any(p.fold != outer_fold for p in test)):
        raise SplitError("split coverage, disjointness or original fold assignment failed")
    if {p.outcome for p in train} != {"good", "poor"} or {p.outcome for p in validation} != {"good", "poor"}:
        raise SplitError("both inner partitions must contain both outcome classes")
    return FoldPlan(
        outer_fold, seed, float(validation_fraction),
        tuple(sorted(train, key=lambda p: p.patient_id)),
        tuple(sorted(validation, key=lambda p: p.patient_id)),
        tuple(sorted(test, key=lambda p: p.patient_id)),
        tuple(cohort_stratified),
    )
