"""Synthetic-only tests. No imports from private UiS patient/model code."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "export_stage7_private_overlay.py"
if not SCRIPT.is_file():  # synthetic-only standalone local development
    SCRIPT = Path(__file__).with_name("export_stage7_private_overlay.py")
SPEC = importlib.util.spec_from_file_location("stage7_private_overlay", SCRIPT)
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


class SelectionTests(unittest.TestCase):
    def test_deterministic_middle_absolute_logit(self):
        values = [float(i) for i in reversed(range(23))]
        self.assertEqual(overlay.select_development_ordinal(values), 11)
        self.assertEqual(overlay.select_development_ordinal([0.] * 23), 11)

    def test_requires_exact_finite_development_count(self):
        for values in ([0.] * 22, [0.] * 24, [float("nan")] + [0.] * 22):
            with self.assertRaises(ValueError):
                overlay.select_development_ordinal(values)


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "optional matplotlib not installed")
class ImageTests(unittest.TestCase):
    def make_inputs(self):
        z, y, x = np.indices((8, 31, 35))
        image = (x + y + z).astype(float)
        grads = np.maximum(0., 1. - ((x - 12)**2 + (y - 15)**2)/180.)
        hires = np.maximum(0., 1. - ((x - 18)**2 + (y - 14)**2)/70.)
        affine = np.diag([1.2, 0.8, 2.5, 1.])
        return image, {"gradcam": grads, "hirescam": hires}, affine

    def test_patient_free_synthetic_png(self):
        image, maps, affine = self.make_inputs()
        for mode in ("ncct", "ctp"):
            data = overlay.render_triptych(image, maps, affine, modality=mode)
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(data), 5000)
            self.assertNotIn(b"patient_id", data)

    def test_rejects_misaligned_and_empty_cam(self):
        image, maps, affine = self.make_inputs()
        bad = dict(maps)
        bad["gradcam"] = maps["gradcam"][..., :-1]
        with self.assertRaises(ValueError):
            overlay.render_triptych(image, bad, affine, modality="ncct")
        bad = {"gradcam": np.zeros_like(image), "hirescam": np.zeros_like(image)}
        with self.assertRaises(ValueError):
            overlay.render_triptych(image, bad, affine, modality="ncct")
        with self.assertRaises(ValueError):
            overlay.render_triptych(image, maps, np.zeros_like(affine), modality="ncct")


class PrivacyTests(unittest.TestCase):
    def test_exclusive_owner_only_file_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.chmod(temporary, 0o700)
            directory = overlay.private_output_dir(temporary, "stage7_overlay_synthetic_01")
            self.assertEqual(directory.stat().st_mode & 0o077, 0)
            digest = overlay.exclusive_write(directory, "ncct_development_example.png", b"fictional")
            self.assertEqual(len(digest), 64)
            file = directory / "ncct_development_example.png"
            self.assertEqual(file.stat().st_mode & 0o077, 0)
            with self.assertRaises(FileExistsError):
                overlay.exclusive_write(directory, "ncct_development_example.png", b"fictional")
            with self.assertRaises(FileExistsError):
                overlay.private_output_dir(temporary, "stage7_overlay_synthetic_01")

    def test_refuses_unsafe_output_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            for slug in ("../out", "stage7_overlay_", "stage7_overlay_a/b"):
                with self.assertRaises(ValueError):
                    overlay.private_output_dir(temporary, slug)

    def test_refuses_symlinked_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "target"
            real.mkdir()
            (root / "stage7_private_overlay_examples").symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                overlay.private_output_dir(root, "stage7_overlay_synthetic_02")


if __name__ == "__main__":
    unittest.main()
