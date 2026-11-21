"""Fictional clinical rows only; no file, image or private dataset access."""

from __future__ import annotations

import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.clinical_preprocessing import (
    FEATURE_NAMES, ClinicalPreprocessingError, ClinicalPreprocessor,
    clinical_matrix, parse_onset_to_ct_minutes,
)
from ele630.data_contract import OutcomePatient
from ele630.ncct_splits import FoldPlan, SplitError


FORBIDDEN = (
    "patient_id", "mrs_3months", "onset_time", "ct_time", "door_time",
    "nihss_discharge", "thrombectomy", "recanalization", "mTICI",
    "needle_time", "dnt", "followup_mr", "fold", "cohort",
)


def row(**changes):
    result = dict(zip(FEATURE_NAMES, (20, "F", 20, 1, "2:00", 0, 0, 0, 0, 0, 0)))
    result.update(changes)
    return result


def plan():
    def pair(partition, fold):
        return tuple(OutcomePatient(f"fictional-{partition}-{i}", "synthetic",
                                    "good" if i == 0 else "poor", False, fold)
                     for i in range(2))
    return FoldPlan(0, 630, 0.20, pair("train", 1), pair("validation", 2),
                    pair("test", 0), ())


def training_rows(fold_plan):
    return {
        fold_plan.inner_train[0].patient_id: row(age=10, sex="F", nihss_admission=10,
                                               mrs_baseline=0, onset_to_ct_time="1:00"),
        fold_plan.inner_train[1].patient_id: row(age=30, sex="M", nihss_admission=30,
                                               mrs_baseline=2, onset_to_ct_time="3:00",
                                               af_paf=1, ht=1, dm=1, smoking=3,
                                               anticoagulation=1),
    }


class DurationTests(unittest.TestCase):
    def test_hours_minutes_seconds_and_long_durations(self):
        for raw, minutes in (("0:00", 0), ("1:05", 65), ("01:05:30", 65.5),
                             (" 23:59:59 ", 1439 + 59 / 60), ("25:00:00", 1500)):
            with self.subTest(raw=raw):
                self.assertAlmostEqual(parse_onset_to_ct_minutes(raw), minutes)

    def test_explicit_missing_values_remain_nan(self):
        for raw in (None, "", " ", "N/A", "na", "NaN", "None", "NULL", math.nan,
                    np.float32("nan")):
            with self.subTest(raw=raw):
                self.assertTrue(math.isnan(parse_onset_to_ct_minutes(raw)))

    def test_malformed_values_are_not_silently_missing(self):
        for raw in ("private-sentinel", "1:5", "1:60", "00:00:60", "-1:00",
                    "1:00:00:00", "1:00:00.5", "1 days 00:00:00", "2026-10-11",
                    "inf", float("inf"), 60, True, [], "9" * 400 + ":00"):
            with self.subTest(raw=raw), self.assertRaises(ClinicalPreprocessingError) as caught:
                parse_onset_to_ct_minutes(raw)
            self.assertNotIn("private-sentinel", str(caught.exception))


class ClinicalSchemaTests(unittest.TestCase):
    def test_exact_approved_schema_and_forbidden_columns_never_read(self):
        self.assertEqual(FEATURE_NAMES, (
            "age", "sex", "nihss_admission", "mrs_baseline", "onset_to_ct_time",
            "af_paf", "ht", "dm", "smoking", "anticoagulation", "wake_up_stroke",
        ))

        class GuardedRow(dict):
            def __getitem__(self, key):
                if key not in FEATURE_NAMES:
                    raise AssertionError("forbidden predictor accessed")
                return super().__getitem__(key)

        source = GuardedRow(row(**dict.fromkeys(FORBIDDEN, object())))
        np.testing.assert_array_equal(clinical_matrix([source]), clinical_matrix([row()]))
        self.assertFalse(set(FEATURE_NAMES) & set(FORBIDDEN))

    def test_smoking_binary_recode_and_missing(self):
        result = clinical_matrix([row(smoking=v) for v in (0, "0", 1, "2", 3, None)])
        np.testing.assert_array_equal(result[:5, 8], [0, 0, 1, 1, 1])
        self.assertTrue(np.isnan(result[5, 8]))

    def test_wake_up_masking_precedes_parsing_and_leaves_source_unchanged(self):
        rows = [row(wake_up_stroke="1", onset_to_ct_time="4:15"),
                row(wake_up_stroke=1, onset_to_ct_time="unknown-onset"),
                row(wake_up_stroke=0, onset_to_ct_time="4:15"),
                row(wake_up_stroke=None, onset_to_ct_time="4:15")]
        original = copy.deepcopy(rows)
        result = clinical_matrix(rows)
        self.assertTrue(np.isnan(result[:2, 4]).all())
        np.testing.assert_array_equal(result[2:, 4], [255, 255])
        np.testing.assert_array_equal(result[:3, 10], [1, 1, 0])
        self.assertTrue(np.isnan(result[3, 10]))
        self.assertEqual(rows, original)
        with self.assertRaises(ClinicalPreprocessingError):
            clinical_matrix([row(wake_up_stroke=0, onset_to_ct_time="unknown-onset")])

    def test_rejects_invalid_schema_codes_and_numeric_values_without_echo(self):
        bad = row()
        del bad["age"]
        cases = [bad, {name: None for name in FORBIDDEN},
                 row(age=-1), row(age=float("inf")), row(age="private-value"),
                 row(sex="unknown"), row(sex=1), row(nihss_admission=43),
                 row(nihss_admission=0.5), row(mrs_baseline=7), row(mrs_baseline=1.5),
                 row(smoking=-1), row(wake_up_stroke=2)]
        cases.extend(row(**{name: 2}) for name in ("af_paf", "ht", "dm", "anticoagulation"))
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ClinicalPreprocessingError) as caught:
                clinical_matrix([candidate])
            self.assertNotIn("private-value", str(caught.exception))


class ClinicalFittingTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan()
        self.rows = training_rows(self.plan)

    def fitted(self):
        return ClinicalPreprocessor().fit_training(self.plan, self.rows)

    def test_hand_calculated_train_statistics_and_fixed_output_order(self):
        transformer = self.fitted()
        output = transformer.transform(list(self.rows.values()))
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.shape, (2, 11))
        np.testing.assert_array_equal(output[:, [0, 2, 3, 4]], [[-1] * 4, [1] * 4])
        np.testing.assert_array_equal(output[:, [1, 5, 6, 7, 8, 9]], [[0] * 6, [1] * 6])
        missing = transformer.transform([dict.fromkeys(FEATURE_NAMES)])
        np.testing.assert_array_equal(missing, np.zeros((1, 11)))
        # Validation age 100 uses training mean 20 and scale 10, never its own mean.
        self.assertEqual(transformer.transform([row(age=100)])[0, 0], 8)

    def test_only_training_rows_accessed_and_transform_never_updates_fit(self):
        accesses = []

        class TrainingOnly(dict):
            def __getitem__(inner, key):
                accesses.append(key)
                if key not in self.rows:
                    raise AssertionError("nontraining clinical row accessed")
                return super().__getitem__(key)

        transformer = ClinicalPreprocessor().fit_training(self.plan, TrainingOnly(self.rows))
        self.assertEqual(accesses, [p.patient_id for p in self.plan.inner_train])
        before = transformer.transform([row(age=None)])
        transformer.transform([row(age=10000), row(age=0, sex="M", smoking=3)])
        np.testing.assert_array_equal(before, transformer.transform([row(age=None)]))
        with self.assertRaisesRegex(ClinicalPreprocessingError, "already fitted"):
            transformer.fit_training(self.plan, self.rows)

    def test_wake_up_interval_excluded_from_training_statistics(self):
        self.rows[self.plan.inner_train[1].patient_id] = row(wake_up_stroke=1,
                                                            onset_to_ct_time="999:00")
        output = self.fitted().transform([row(wake_up_stroke=1, onset_to_ct_time="invalid"),
                                          row(onset_to_ct_time="2:00")])
        # Only the first training duration (60 min) is observed; scale is one.
        np.testing.assert_array_equal(output[:, 4], [0, 60])
        np.testing.assert_array_equal(output[:, 10], [1, 0])

    def test_imputation_precedes_scaling_and_positive_mode_is_learned(self):
        first = self.plan.inner_train[0].patient_id
        self.rows[first] = row(age=None, sex=None, smoking=None)
        transformer = self.fitted()
        output = transformer.transform([row(age=None, sex=None, smoking=None)])
        self.assertEqual(output[0, 0], 0)
        self.assertEqual(output[0, 1], 1)
        self.assertEqual(output[0, 8], 1)

    def test_median_imputation_and_population_scaling_with_partial_missingness(self):
        extra = tuple(OutcomePatient(f"fictional-extra-{i}", "synthetic", "good", False, 1)
                      for i in range(2))
        expanded = replace(self.plan, inner_train=self.plan.inner_train + extra)
        rows = training_rows(expanded)
        rows[extra[0].patient_id] = row(age=100)
        rows[extra[1].patient_id] = row(age=None)
        transformer = ClinicalPreprocessor().fit_training(expanded, rows)
        # Observed ages 10,30,100 -> median 30, imputed ages 10,30,100,30.
        # Population center 42.5; variance 1168.75.
        result = transformer.transform([row(age=None), row(age=10)])[:, 0]
        np.testing.assert_allclose(result, [-12.5 / math.sqrt(1168.75),
                                          -32.5 / math.sqrt(1168.75)], rtol=1e-6)

    def test_entirely_missing_training_columns_fail_even_with_validation_values(self):
        for name in FEATURE_NAMES:
            with self.subTest(name=name):
                rows = copy.deepcopy(self.rows)
                for source in rows.values():
                    source[name] = None
                rows.update({p.patient_id: row() for p in self.plan.inner_validation})
                with self.assertRaisesRegex(ClinicalPreprocessingError, "entirely missing in training"):
                    ClinicalPreprocessor().fit_training(self.plan, rows)

    def test_labels_extra_columns_and_row_order_cannot_change_rules(self):
        first = self.fitted().transform([row(age=None, sex=None, smoking=None)])
        changed_plan = replace(self.plan, inner_train=tuple(
            replace(p, outcome="poor" if p.outcome == "good" else "good")
            for p in reversed(self.plan.inner_train)))
        rows = {key: dict(source, **dict.fromkeys(FORBIDDEN, "arbitrary"))
                for key, source in self.rows.items()}
        second = ClinicalPreprocessor().fit_training(changed_plan, rows).transform(
            [row(age=None, sex=None, smoking=None)])
        np.testing.assert_array_equal(first, second)

    def test_repeated_fits_source_integrity_empty_transform_and_zero_variance(self):
        original = copy.deepcopy(self.rows)
        a, b = self.fitted(), self.fitted()
        np.testing.assert_array_equal(a.transform([row()]), b.transform([row()]))
        self.assertEqual(self.rows, original)
        self.assertEqual(a.transform([]).shape, (0, 11))
        constant = {p.patient_id: row() for p in self.plan.inner_train}
        output = ClinicalPreprocessor().fit_training(self.plan, constant).transform([row()])
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[:, [0, 2, 3, 4]], [[0] * 4])

    def test_unfitted_invalid_plan_missing_row_and_float32_overflow_fail(self):
        with self.assertRaisesRegex(ClinicalPreprocessingError, "fit on inner training"):
            ClinicalPreprocessor().transform([row()])
        with self.assertRaises(SplitError):
            ClinicalPreprocessor().fit_training(replace(self.plan, inner_train=self.plan.outer_test), self.rows)
        with self.assertRaisesRegex(ClinicalPreprocessingError, "missing inner-training") as caught:
            ClinicalPreprocessor().fit_training(self.plan, {})
        self.assertNotIn("fictional", str(caught.exception))
        with self.assertRaisesRegex(ClinicalPreprocessingError, "nonfinite transformed"):
            self.fitted().transform([row(age=1e100)])


if __name__ == "__main__":
    unittest.main()
