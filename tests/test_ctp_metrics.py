"""Hand-verifiable fictional scores; no image or PyTorch dependency."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.ctp_metrics import CTPMetricError, aggregate_ctp_metrics


class CTPMetricsTests(unittest.TestCase):
    def test_hand_computed_aggregates_and_counts(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.4, 0.35, 0.8]
        losses = [0.2, 0.3, 0.4, 0.5]
        result = aggregate_ctp_metrics(y, p, losses)
        self.assertAlmostEqual(result.bce, 0.35)
        self.assertAlmostEqual(result.auroc, 0.75)
        self.assertAlmostEqual(result.average_precision, 5 / 6)
        self.assertAlmostEqual(result.brier_score, (0.01 + 0.16 + 0.65**2 + 0.2**2) / 4)
        self.assertAlmostEqual(result.balanced_accuracy_0_5, 0.75)
        self.assertAlmostEqual(result.mean_predicted_probability, 0.4125)
        self.assertEqual((result.validation_count, result.validation_good_count,
                          result.validation_poor_count), (4, 2, 2))
        self.assertEqual(len(result.as_dict()), 9)

    def test_ranking_uses_scores_even_when_every_fixed_decision_is_good(self):
        result = aggregate_ctp_metrics([0, 0, 1, 1], [0.1, 0.2, 0.15, 0.4],
                                       [0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(result.auroc, 0.75)
        self.assertAlmostEqual(result.average_precision, 5 / 6)
        self.assertAlmostEqual(result.balanced_accuracy_0_5, 0.5)

    def test_fixed_threshold_includes_exactly_point_five_and_ties_are_grouped(self):
        tied = aggregate_ctp_metrics([0, 1], [0.5, 0.5], [math.log(2)] * 2)
        self.assertAlmostEqual(tied.auroc, 0.5)
        self.assertAlmostEqual(tied.average_precision, 0.5)
        self.assertAlmostEqual(tied.balanced_accuracy_0_5, 0.5)
        just_below = aggregate_ctp_metrics([0, 1], [0.499999, 0.5], [0.6, 0.7])
        self.assertAlmostEqual(just_below.balanced_accuracy_0_5, 1.0)

    def test_bce_and_brier_match_direct_reference(self):
        logits = np.array([-2.0, -0.5, 0.2, 1.5], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probabilities = 1 / (1 + np.exp(-logits))
        case_bce = np.logaddexp(0, logits) - labels * logits
        result = aggregate_ctp_metrics(labels.tolist(), probabilities.tolist(),
                                       case_bce.tolist())
        self.assertAlmostEqual(result.bce, float(np.mean(case_bce)), places=14)
        self.assertAlmostEqual(result.brier_score,
                               float(np.mean((probabilities - labels) ** 2)), places=14)

    def test_single_class_or_invalid_values_fail_without_echoing_values(self):
        for labels, probabilities, losses in (
            ([0, 0], [0.1, 0.2], [0.1, 0.2]),
            ([0, 1], [0.1, 1.01], [0.1, 0.2]),
            ([0, 1], [0.1, float("nan")], [0.1, 0.2]),
            ([0, 1], [0.1, 0.2], [0.1, -1]),
            ([0, 1], [0.1], [0.1, 0.2]),
            ([0, 2], [0.1, 0.2], [0.1, 0.2]),
        ):
            with self.subTest(labels=labels), self.assertRaises(CTPMetricError) as caught:
                aggregate_ctp_metrics(labels, probabilities, losses)
            self.assertNotIn("1.01", str(caught.exception))
        with self.assertRaisesRegex(CTPMetricError, "AUROC"):
            aggregate_ctp_metrics([0, 0], [0.1, 0.2], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
