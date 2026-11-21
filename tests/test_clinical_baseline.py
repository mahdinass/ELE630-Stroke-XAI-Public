"""Stage 5A synthetic access/metric gates and optional real-estimator tests."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import sys
import types
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630 import clinical_baseline as baseline
from ele630.clinical_preprocessing import ClinicalPreprocessor, FEATURE_NAMES
from ele630.data_contract import OutcomePatient
from ele630.ncct_splits import FoldPlan
from test_clinical_preprocessing import FORBIDDEN, row


def fictional_plan():
    def patients(part, outcomes, fold):
        return tuple(OutcomePatient(f"fictional-{part}-{i}", "synthetic", outcome, False, fold)
                     for i, outcome in enumerate(outcomes))
    return FoldPlan(0, 630, 0.20,
                    patients("train", ("good", "good", "good", "poor"), 1),
                    patients("validation", ("good", "good", "poor", "poor"), 2),
                    patients("test", ("good", "poor"), 0), ())


def clinical_rows(plan):
    return {p.patient_id: row(age=10 * (i + 1), nihss_admission=2 * (i + 1))
            for i, p in enumerate(plan.inner_train + plan.inner_validation)}


class FixedEstimator:
    """Test double only: records calls and returns fixed scores; never optimizes."""
    def __init__(self):
        self.calls = []
        self.classes_ = np.array([0, 1])
        self.coef_ = np.zeros((1, 11))
        self.intercept_ = np.zeros(1)

    def fit(self, inputs, labels):
        self.calls.append(("fit", inputs.copy(), labels.copy()))
        return self

    def predict_proba(self, inputs):
        self.calls.append(("predict", inputs.copy()))
        p = np.array([0.1, 0.4, 0.35, 0.8])
        return np.column_stack((1 - p, p))


class ProbabilityMetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics_match_imaging_definitions(self):
        result = baseline.aggregate_probability_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(result.bce, -math.log(0.9 * 0.6 * 0.35 * 0.8) / 4)
        self.assertAlmostEqual(result.auroc, 0.75)
        self.assertAlmostEqual(result.average_precision, 5 / 6)
        self.assertAlmostEqual(result.brier_score, (0.01 + 0.16 + 0.65**2 + 0.2**2) / 4)
        self.assertAlmostEqual(result.balanced_accuracy_0_5, 0.75)
        self.assertAlmostEqual(result.mean_predicted_probability, 0.4125)
        self.assertEqual((result.validation_count, result.validation_good_count,
                          result.validation_poor_count), (4, 2, 2))

    def test_probability_endpoints_are_finite_and_only_bce_is_clipped(self):
        eps = np.finfo(np.float64).eps
        correct = baseline.aggregate_probability_metrics([0, 1], [0.0, 1.0])
        wrong = baseline.aggregate_probability_metrics([0, 1], [1.0, 0.0])
        self.assertAlmostEqual(correct.bce, -math.log1p(-eps))
        self.assertAlmostEqual(wrong.bce, -math.log(eps))
        self.assertEqual(correct.brier_score, 0)
        self.assertEqual(wrong.brier_score, 1)
        tiny = baseline.aggregate_probability_metrics([0, 1], [1e-300, 1 - eps])
        self.assertTrue(math.isfinite(tiny.bce))

    def test_fixed_threshold_ties_and_no_threshold_optimization(self):
        tied = baseline.aggregate_probability_metrics([0, 1], [0.5, 0.5])
        self.assertEqual(tied.auroc, 0.5)
        self.assertEqual(tied.average_precision, 0.5)
        self.assertEqual(tied.balanced_accuracy_0_5, 0.5)
        below = baseline.aggregate_probability_metrics([0, 1], [0.1, 0.4])
        self.assertEqual(below.balanced_accuracy_0_5, 0.5)

    def test_invalid_arrays_and_labels_have_safe_errors(self):
        for labels, probabilities in (([], []), ([0], [0.5]), ([0, 2], [0.1, 0.2]),
                                     ([0., 1.], [0.1, 0.2]), ([0, 1], [0.5]),
                                     ([[0, 1]], [[0.1, 0.2]]), ([0, 1], [math.nan, 0.2]),
                                     ([0, 1], [math.inf, 0.2]), ([0, 1], [-0.1, 0.2]),
                                     ([0, 1], [1.01, 0.2]), ([0, 1], ["private-value", 0.2])):
            with self.subTest(labels=labels), self.assertRaises(baseline.ClinicalBaselineError) as caught:
                baseline.aggregate_probability_metrics(labels, probabilities)
            self.assertNotIn("private-value", str(caught.exception))

    def test_prior_uses_training_prevalence_not_validation_prevalence(self):
        plan = fictional_plan()
        result = baseline.evaluate_training_prior(plan)
        self.assertEqual(result.mean_predicted_probability, 0.25)
        self.assertEqual(result.auroc, 0.5)
        self.assertEqual(result.average_precision, 0.5)
        self.assertEqual(result.balanced_accuracy_0_5, 0.5)
        self.assertAlmostEqual(result.bce, -(math.log(0.75) + math.log(0.25)) / 2)
        self.assertAlmostEqual(result.brier_score, (0.25**2 + 0.75**2) / 2)
        changed = replace(plan, inner_validation=tuple(
            replace(p, outcome="poor") if i == 1 else p for i, p in enumerate(plan.inner_validation)))
        self.assertEqual(baseline.evaluate_training_prior(changed).mean_predicted_probability, 0.25)


class ClinicalDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.plan = fictional_plan()
        self.rows = clinical_rows(self.plan)
        self.estimator = FixedEstimator()

    def run_baseline(self, plan=None, rows=None):
        with patch.object(baseline, "_make_logistic_regression", return_value=self.estimator):
            return baseline.run_clinical_development(plan or self.plan, self.rows if rows is None else rows)

    def test_train_only_fit_transform_then_validation_only_prediction(self):
        accesses, preprocessing_calls = [], []
        train_ids = [p.patient_id for p in self.plan.inner_train]
        validation_ids = [p.patient_id for p in self.plan.inner_validation]

        class GuardedRows(dict):
            def __getitem__(inner, key):
                if key not in train_ids + validation_ids:
                    raise AssertionError("outer-test or unknown source access")
                if key in validation_ids:
                    self.assertEqual(self.estimator.calls[0][0], "fit")
                accesses.append(key)
                return super().__getitem__(key)

            def __iter__(inner):
                raise AssertionError("source mapping must not be scanned")

        class SpyPreprocessor(ClinicalPreprocessor):
            def fit_training(inner, plan, rows):
                self.assertEqual(set(rows), set(train_ids))
                preprocessing_calls.append("fit")
                return super().fit_training(plan, rows)

            def transform(inner, rows):
                preprocessing_calls.append(len(rows))
                return super().transform(rows)

        with patch.object(baseline, "ClinicalPreprocessor", SpyPreprocessor):
            result = self.run_baseline(rows=GuardedRows(self.rows))
        self.assertEqual(accesses, train_ids + validation_ids)
        self.assertEqual(preprocessing_calls, ["fit", 4, 4])
        self.assertEqual([call[0] for call in self.estimator.calls], ["fit", "predict"])
        np.testing.assert_array_equal(self.estimator.calls[0][2], [0, 0, 0, 1])
        scale = math.sqrt(125)
        np.testing.assert_allclose(self.estimator.calls[0][1][:, 0], [-15 / scale, -5 / scale, 5 / scale, 15 / scale])
        np.testing.assert_allclose(self.estimator.calls[1][1][:, 0], [25 / scale, 35 / scale, 45 / scale, 55 / scale])
        self.assertEqual(result.training_count, 4)
        self.assertEqual(result.training_poor_prevalence, 0.25)
        self.assertEqual(result.logistic_regression.auroc, 0.75)

    def test_validation_and_outer_label_changes_cannot_change_fit(self):
        self.run_baseline()
        original_x, original_y = self.estimator.calls[0][1:]
        changed = replace(self.plan, inner_validation=tuple(replace(p, outcome="poor" if p.outcome == "good" else "good")
                                                           for p in self.plan.inner_validation),
                          outer_test=tuple(replace(p, outcome="poor" if p.outcome == "good" else "good")
                                           for p in self.plan.outer_test))
        self.estimator = FixedEstimator()
        self.run_baseline(plan=changed)
        np.testing.assert_array_equal(self.estimator.calls[0][1], original_x)
        np.testing.assert_array_equal(self.estimator.calls[0][2], original_y)

    def test_forbidden_columns_never_read_source_unchanged_and_results_aggregate(self):
        class GuardedRow(dict):
            def __getitem__(inner, key):
                if key not in FEATURE_NAMES:
                    raise AssertionError("forbidden feature read")
                return super().__getitem__(key)

        rows = {key: GuardedRow(dict(source, **dict.fromkeys(FORBIDDEN, "private-value")))
                for key, source in self.rows.items()}
        original = copy.deepcopy(rows)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.run_baseline(rows=rows)
        self.assertEqual(rows, original)
        self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")
        record = result.as_dict()
        self.assertEqual(set(record), {"training_count", "training_poor_prevalence", "logistic_regression", "training_prior"})
        self.assertEqual(len(record["logistic_regression"]), 9)
        for secret in ("fictional", "private-value", "coef_", "patient_id", '"age"'):
            self.assertNotIn(secret, json.dumps(record) + repr(result))

    def test_validation_feature_changes_do_not_change_training_parameters(self):
        self.run_baseline()
        original_x, original_y = self.estimator.calls[0][1:]
        rows = copy.deepcopy(self.rows)
        for p in self.plan.inner_validation:
            rows[p.patient_id] = row(age=10000, smoking=3, nihss_admission=42)
        self.estimator = FixedEstimator()
        self.run_baseline(rows=rows)
        np.testing.assert_array_equal(self.estimator.calls[0][1], original_x)
        np.testing.assert_array_equal(self.estimator.calls[0][2], original_y)

    def test_existing_wake_up_and_smoking_rules_are_preserved(self):
        rows = copy.deepcopy(self.rows)
        rows[self.plan.inner_validation[0].patient_id] = row(
            wake_up_stroke=1, onset_to_ct_time="unknown-onset", smoking=3)
        self.run_baseline(rows=rows)
        transformed = self.estimator.calls[1][1][0]
        self.assertEqual(transformed[4], 0)  # Constant training interval, masked/imputed.
        self.assertEqual(transformed[8], 1)
        self.assertEqual(transformed[10], 1)

    def test_invalid_plan_rejected_before_dependency_loading_or_row_access(self):
        bad_plans = [None, replace(self.plan, outer_fold=1), replace(self.plan, seed=631),
                     replace(self.plan, validation_fraction=0.3),
                     replace(self.plan, inner_train=self.plan.outer_test),
                     replace(self.plan, inner_validation=self.plan.inner_train),
                     replace(self.plan, inner_train=self.plan.inner_train[:3])]
        with patch.object(baseline, "_make_logistic_regression") as factory:
            for candidate in bad_plans:
                with self.subTest(plan=candidate), self.assertRaises(baseline.ClinicalBaselineError):
                    baseline.run_clinical_development(candidate, Mock())
            factory.assert_not_called()

    def test_fictional_143_patient_plan_cannot_impersonate_active_contract(self):
        extra = tuple(OutcomePatient(f"fictional-extra-{i}", "synthetic", "good", False, 1) for i in range(133))
        expanded = replace(self.plan, inner_train=self.plan.inner_train + extra)
        with self.assertRaises(baseline.ClinicalBaselineError):
            baseline.run_clinical_development(expanded, Mock())

    def test_missing_dependency_fails_before_any_rows_are_read(self):
        with patch.dict(sys.modules, {"sklearn.linear_model": None}):
            with self.assertRaisesRegex(baseline.ClinicalBaselineError, "working scikit-learn/SciPy"):
                baseline.run_clinical_development(self.plan, Mock())

    def test_fixed_l2_configuration_uses_established_estimator_constructor(self):
        module = types.ModuleType("sklearn.linear_model")
        constructor = Mock(return_value=self.estimator)
        module.LogisticRegression = constructor
        with patch.dict(sys.modules, {"sklearn.linear_model": module}):
            baseline.run_clinical_development(self.plan, self.rows)
        constructor.assert_called_once_with(penalty="l2", C=1.0, fit_intercept=True,
                                           class_weight=None, solver="lbfgs", max_iter=1000,
                                           tol=1e-8, random_state=630, warm_start=False, verbose=0)

    def test_missing_or_malformed_clinical_rows_stop_safely(self):
        for malformed in ({}, {p.patient_id: row(smoking=-1) for p in self.plan.inner_train},
                          {p.patient_id: row(onset_to_ct_time="private-value") for p in self.plan.inner_train}):
            with self.subTest(rows=malformed), self.assertRaises(baseline.ClinicalBaselineError) as caught:
                self.run_baseline(rows=malformed)
            self.assertNotIn("private-value", str(caught.exception))
        self.estimator = FixedEstimator()
        with self.assertRaisesRegex(baseline.ClinicalBaselineError, "rows could not be read"):
            self.run_baseline(rows={p.patient_id: self.rows[p.patient_id] for p in self.plan.inner_train})
        self.assertEqual([call[0] for call in self.estimator.calls], ["fit"])

    def test_source_model_and_warning_errors_suppress_private_details(self):
        sensitive = "fictional-secret /private/source patient-value"
        for failure in ("read", "fit", "predict", "warning"):
            self.estimator = FixedEstimator()
            if failure == "fit":
                self.estimator.fit = Mock(side_effect=ValueError(sensitive))
            elif failure == "predict":
                self.estimator.predict_proba = Mock(side_effect=ValueError(sensitive))
            elif failure == "warning":
                self.estimator.fit = lambda *args: warnings.warn(sensitive)

            class FailingRows(dict):
                def __getitem__(inner, key):
                    raise OSError(sensitive)

            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(baseline.ClinicalBaselineError) as caught:
                    self.run_baseline(rows=FailingRows() if failure == "read" else self.rows)
            self.assertNotIn(sensitive, str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)
            self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")

    def test_invalid_estimator_classes_coefficients_and_probabilities_stop(self):
        for classes, coefficient, probabilities in (
            ([1, 0], 0, None), ([0, 1], math.nan, None),
            ([0, 1], 0, np.ones((4, 1))), ([0, 1], 0, np.full((4, 2), math.nan)),
            ([0, 1], 0, np.full((4, 2), 0.7)), ([0, 1], 0, np.tile([-0.1, 1.1], (4, 1))),
        ):
            self.estimator = FixedEstimator()
            self.estimator.classes_ = np.array(classes)
            self.estimator.coef_[0, 0] = coefficient
            if probabilities is not None:
                self.estimator.predict_proba = Mock(return_value=probabilities)
            with self.subTest(classes=classes), self.assertRaises(baseline.ClinicalBaselineError):
                self.run_baseline()


SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed; real estimator gate pending")
class RealLogisticRegressionTests(unittest.TestCase):
    def test_real_estimator_repeatability_label_direction_and_prior(self):
        plan = fictional_plan()
        rows = clinical_rows(plan)
        # NIHSS separates fictional good/poor outcomes; never a real-data claim.
        for p in plan.inner_train + plan.inner_validation:
            rows[p.patient_id] = row(nihss_admission=40 if p.outcome == "poor" else 0)
        a = baseline.run_clinical_development(plan, rows)
        b = baseline.run_clinical_development(plan, rows)
        self.assertEqual(a, b)
        self.assertEqual(a.logistic_regression.auroc, 1)
        self.assertEqual(a.logistic_regression.balanced_accuracy_0_5, 1)
        self.assertLess(a.logistic_regression.bce, a.training_prior.bce)
        self.assertEqual(a.training_poor_prevalence, 0.25)

    def test_real_convergence_warning_is_rejected_without_retry(self):
        from sklearn.exceptions import ConvergenceWarning
        estimator = FixedEstimator()
        estimator.fit = lambda *args: warnings.warn("synthetic convergence failure", ConvergenceWarning)
        with patch.object(baseline, "_make_logistic_regression", return_value=estimator):
            with self.assertRaisesRegex(baseline.ClinicalBaselineError, "fitting failed"):
                baseline.run_clinical_development(fictional_plan(), clinical_rows(fictional_plan()))


if __name__ == "__main__":
    unittest.main()
