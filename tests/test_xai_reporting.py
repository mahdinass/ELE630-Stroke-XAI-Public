"""Report tests use entirely fictional scalar fixtures, never measured patients."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test_xai_infrastructure import fictional_public

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class ReportingTests(unittest.TestCase):
    def test_tables_metadata_and_negative_results_no_patient_fields(self):
        from ele630.xai_reporting import write_report
        record = fictional_public()
        row = record["summary"]["gradcam.fidelity.excess_drop"]
        row.update(mean=-.3, median=-.3, ci_low=-.5, ci_high=-.1)
        with tempfile.TemporaryDirectory() as temp:
            path = write_report(record, Path(temp)/"fictional_report", fictional=True)
            files = {p.name for p in path.iterdir()}
            self.assertIn("missingness.csv", files)
            self.assertIn("paired_method_differences.csv", files)
            self.assertIn("parameter_randomization.csv", files)
            self.assertIn("temporal_windows.csv", files)
            self.assertIn("-0.3", (path/"gradcam_vs_hirescam.csv").read_text())
            metadata = json.loads((path/"reproducibility.json").read_text())
            self.assertTrue(metadata["fictional_synthetic_example"])
            self.assertFalse(metadata["between_seed_stability_assessed"])
            self.assertIn("FICTIONAL SYNTHETIC EXAMPLE", (path/"README.md").read_text())
            all_text = "".join(p.read_text() for p in path.iterdir())
            for field in ("patient_id", "coefficients", "embedding", "probabilities"):
                self.assertNotIn(field, all_text)
            with self.assertRaises(ValueError):
                write_report(record, path)

    def test_invalid_input_rejected_before_any_output(self):
        from ele630.xai_reporting import write_report
        record = fictional_public()
        record["patient_id"] = "fictional-sensitive"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"fictional_report"
            with self.assertRaises(ValueError):
                write_report(record, path)
            self.assertFalse(path.exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "existing environment has no Matplotlib; tables remain available")
    def test_standard_figures_all40_axes_and_synthetic_label(self):
        from ele630.xai_reporting import render_figures
        record = fictional_public()
        figures, version = render_figures(record, fictional=True)
        self.assertEqual(len(figures), 10)
        self.assertTrue(version)
        temporal = figures["ctp_temporal_sensitivity.svg"].decode()
        self.assertIn("FICTIONAL SYNTHETIC EXAMPLE", temporal)
        self.assertIn("frame-content sensitivity", temporal)
        self.assertIn("no clinical phase assignment", temporal)
        self.assertTrue(figures["roi_enrichment.png"].startswith(b"\x89PNG"))
        ncct = copy.deepcopy(record)
        ncct["mode"] = "ncct"
        ncct_figures, _ = render_figures(ncct, fictional=True)
        self.assertNotIn("ctp_temporal_sensitivity.svg", ncct_figures)


if __name__ == "__main__":
    unittest.main()
