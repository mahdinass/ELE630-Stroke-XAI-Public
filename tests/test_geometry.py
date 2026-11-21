"""Synthetic physical-coordinate tests for the proposed grid planner."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ele630.geometry import GeometryError, plan_output_grid


LPS_AFFINE = (
    (-2.0, 0.0, 0.0, 100.0),
    (0.0, -2.0, 0.0, 200.0),
    (0.0, 0.0, 3.0, -30.0),
    (0.0, 0.0, 0.0, 1.0),
)


class GeometryTests(unittest.TestCase):
    def test_even_output_centers_on_input_world_center(self):
        plan = plan_output_grid(
            (5, 7, 3), LPS_AFFINE,
            target_spacing_mm=(1.0, 1.0, 3.0), target_shape=(10, 12, 4),
        )
        self.assertEqual(plan.center_world_mm, (96.0, 194.0, -27.0))
        self.assertEqual(
            plan.output_index_to_world((4.5, 5.5, 1.5)),
            plan.center_world_mm,
        )
        self.assertEqual(plan.output_affine[0][:3], (-1.0, 0.0, 0.0))
        self.assertEqual(plan.output_affine[1][:3], (0.0, -1.0, 0.0))
        self.assertEqual(plan.output_affine[2][:3], (0.0, 0.0, 3.0))

    def test_physical_coordinate_mapping_is_exact(self):
        plan = plan_output_grid(
            (5, 5, 3), LPS_AFFINE,
            target_spacing_mm=(2.0, 2.0, 3.0), target_shape=(5, 5, 3),
        )
        for index in ((0, 0, 0), (2, 1, 1), (4, 4, 2)):
            self.assertEqual(plan.input_index_to_world(index), plan.output_index_to_world(index))
        self.assertTrue(plan.contains_input_voxel_centers())
        self.assertEqual(plan.input_corner_output_bounds(), ((0.0, 4.0), (0.0, 4.0), (0.0, 2.0)))

    def test_candidate_covers_axis_aligned_example_at_reported_spans(self):
        xy_spacing = 242.525 / 255
        source = (
            (-xy_spacing, 0.0, 0.0, 0.0),
            (0.0, -xy_spacing, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        plan = plan_output_grid(
            (256, 256, 59), source,
            target_spacing_mm=(1.0, 1.0, 3.0), target_shape=(256, 256, 72),
        )
        self.assertTrue(plan.contains_input_voxel_centers())
        bounds = plan.input_corner_output_bounds()
        self.assertAlmostEqual(bounds[0][0], (255 - 242.525) / 2)
        self.assertAlmostEqual(bounds[0][1], (255 + 242.525) / 2)
        self.assertEqual(bounds[2], (19.5 / 3, 19.5 / 3 + 58))

    def test_candidate_detects_axis_aligned_crop(self):
        plan = plan_output_grid(
            (260, 200, 59),
            ((-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0),
             (0.0, 0.0, 3.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            target_spacing_mm=(1.0, 1.0, 3.0), target_shape=(256, 256, 72),
        )
        self.assertFalse(plan.contains_input_voxel_centers())
        self.assertLess(plan.input_corner_output_bounds()[0][0], 0)

    def test_oblique_source_can_crop_despite_short_native_spans(self):
        c = math.sqrt(0.5)
        oblique = (
            (-c, c, 0.0, 0.0),
            (-c, -c, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        plan = plan_output_grid(
            (200, 200, 59), oblique,
            target_spacing_mm=(1.0, 1.0, 3.0), target_shape=(256, 256, 72),
        )
        self.assertFalse(plan.contains_input_voxel_centers())
        self.assertGreater(plan.input_corner_output_bounds()[0][1], 255)

    def test_same_inputs_produce_same_affine(self):
        arguments = ((5, 7, 3), LPS_AFFINE)
        first = plan_output_grid(*arguments, target_spacing_mm=(1, 1, 3), target_shape=(8, 10, 4))
        second = plan_output_grid(*arguments, target_spacing_mm=(1, 1, 3), target_shape=(8, 10, 4))
        self.assertEqual(first, second)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaisesRegex(GeometryError, "three positive integers"):
            plan_output_grid((5, 0, 3), LPS_AFFINE, target_spacing_mm=(1, 1, 3), target_shape=(8, 8, 8))
        with self.assertRaisesRegex(GeometryError, "finite positive"):
            plan_output_grid((5, 5, 3), LPS_AFFINE, target_spacing_mm=(1, 0, 3), target_shape=(8, 8, 8))
        singular = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        with self.assertRaisesRegex(GeometryError, "singular"):
            plan_output_grid((5, 5, 3), singular, target_spacing_mm=(1, 1, 3), target_shape=(8, 8, 8))


if __name__ == "__main__":
    unittest.main()
