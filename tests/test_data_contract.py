"""Fabricated rows only; no UiS patient data or assignments."""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "preflight"))

from ele630.config import ServerPaths
from ele630.data_contract import ContractError, ValidationProfile, load_and_validate
from validate_active_dataset import main as preflight_main


MANIFEST_FIELDS = [
    "patient_id", "cohort", "has_ncct_skull_stripped", "has_ctp_registered",
    "has_ctp_registered_mask", "mrs_3months", "mrs_binary", "eligible_fusion_mrs",
]
FOLD_FIELDS = ["patient_id", "fold", "cohort", "mrs_binary", "stratum", "has_gt"]


def fixture_rows():
    manifest = []
    folds = []
    for index in range(10):
        pid = f"fictional-case-{index:02d}"
        cohort = f"{index % 2 + 1:02d}"
        outcome = "poor" if index % 2 else "good"
        gt = int(index != 9)
        manifest.append({
            "patient_id": pid, "cohort": cohort,
            "has_ncct_skull_stripped": 1, "has_ctp_registered": 1,
            "has_ctp_registered_mask": gt,
            "mrs_3months": 4 if outcome == "poor" else 1,
            "mrs_binary": outcome, "eligible_fusion_mrs": 1,
        })
        folds.append({
            "patient_id": pid, "fold": index // 2, "cohort": cohort,
            "mrs_binary": outcome, "stratum": f"c{cohort}_{outcome}",
            "has_gt": gt,
        })
    return manifest, folds


def synthetic_fingerprint(folds):
    payload = "\n".join(sorted(f"{r['patient_id']}:{r['fold']}" for r in folds))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.manifest_file = root / "manifest.csv"
        self.fold_file = root / "folds.csv"
        self.manifest, self.folds = fixture_rows()
        self.profile = ValidationProfile(
            name="synthetic", patients=10, good=5, poor=5, with_gt=9,
            fold_sizes=(2, 2, 2, 2, 2), poor_per_fold=(1, 1, 1, 1, 1),
            fingerprint=synthetic_fingerprint(self.folds),
        )

    def write(self, manifest_fields=MANIFEST_FIELDS, fold_fields=FOLD_FIELDS):
        for path, fields, rows in (
            (self.manifest_file, manifest_fields, self.manifest),
            (self.fold_file, fold_fields, self.folds),
        ):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    def validate(self, profile=None):
        self.write()
        return load_and_validate(self.manifest_file, self.fold_file, profile or self.profile)

    def expect_failure(self, message, profile=None):
        with self.assertRaisesRegex(ContractError, message):
            self.validate(profile)

    def test_valid_five_fold_dataset(self):
        contract = self.validate()
        self.assertEqual((len(contract.patients), contract.good, contract.poor), (10, 5, 5))
        self.assertEqual(tuple(f.patients for f in contract.fold_summaries), (2,) * 5)

    def test_duplicate_patient_in_manifest(self):
        self.manifest.append(dict(self.manifest[0]))
        self.expect_failure("duplicate patient_id")

    def test_invalid_outcome_label(self):
        self.manifest[0]["mrs_binary"] = "improved"
        self.expect_failure("mrs_binary")

    def test_eligible_patient_without_fold(self):
        self.folds.pop()
        self.expect_failure("eligible patient.*no fold")

    def test_patient_assigned_to_multiple_folds(self):
        row = dict(self.folds[0])
        row["fold"] = 1
        self.folds.append(row)
        self.expect_failure("multiple folds")

    def test_duplicate_patient_fold_assignment(self):
        self.folds.append(dict(self.folds[0]))
        self.expect_failure("duplicate patient/fold")

    def test_unknown_patient_in_fold_file(self):
        self.folds[0]["patient_id"] = "fictional-outsider"
        self.expect_failure("unknown or ineligible")

    def test_wrong_number_of_folds(self):
        self.folds[-1]["fold"] = 3
        self.folds[-2]["fold"] = 3
        self.expect_failure("expected exactly five folds")

    def test_wrong_active_fingerprint(self):
        self.expect_failure("does not match active", replace(self.profile, fingerprint="0" * 16))

    def test_archived_fingerprint_detected(self):
        profile = replace(
            self.profile, fingerprint="0" * 16,
            archived_fingerprints=(synthetic_fingerprint(self.folds),),
        )
        self.expect_failure("archived outcome-CV fingerprint", profile)

    def test_no_acute_gt_is_still_prediction_eligible(self):
        contract = self.validate()
        self.assertEqual(contract.with_gt, 9)
        self.assertEqual(len(contract.patients), 10)
        self.assertFalse(next(p.has_gt for p in contract.patients if p.patient_id == "fictional-case-09"))

    def test_exported_blank_mrs_is_not_prediction_eligible(self):
        # 01_build_manifest.py converts raw N/A (and other unparseable mRS)
        # to an empty mrs_3months/mrs_binary pair in the exported CSV.
        self.manifest.append({
            "patient_id": "fictional-unlabelled", "cohort": "01",
            "has_ncct_skull_stripped": 1, "has_ctp_registered": 1,
            "has_ctp_registered_mask": 1, "mrs_3months": "",
            "mrs_binary": "", "eligible_fusion_mrs": 0,
        })
        contract = self.validate()
        self.assertEqual(len(contract.patients), 10)
        self.assertFalse(any(p.patient_id == "fictional-unlabelled" for p in contract.patients))

    def test_exported_blank_mrs_cannot_be_marked_eligible(self):
        self.manifest.append({
            "patient_id": "fictional-unlabelled", "cohort": "01",
            "has_ncct_skull_stripped": 1, "has_ctp_registered": 1,
            "has_ctp_registered_mask": 0, "mrs_3months": "",
            "mrs_binary": "", "eligible_fusion_mrs": 1,
        })
        self.expect_failure("eligible_fusion_mrs disagrees")

    def test_raw_na_in_exported_manifest_requests_regeneration(self):
        self.manifest.append({
            "patient_id": "fictional-unlabelled", "cohort": "01",
            "has_ncct_skull_stripped": 1, "has_ctp_registered": 1,
            "has_ctp_registered_mask": 0, "mrs_3months": "N/A",
            "mrs_binary": "", "eligible_fusion_mrs": 0,
        })
        self.expect_failure("regenerate the patient manifest")

    def test_aggregate_count_mismatch(self):
        self.expect_failure("eligible outcome patients = 10; expected 11", replace(self.profile, patients=11))

    def test_missing_required_manifest_column(self):
        self.write(manifest_fields=MANIFEST_FIELDS[:-1])
        with self.assertRaisesRegex(ContractError, "missing required columns: eligible_fusion_mrs"):
            load_and_validate(self.manifest_file, self.fold_file, self.profile)

    def test_missing_required_fold_column_rejects_other_split(self):
        self.write(fold_fields=["patient_id", "fold"])
        with self.assertRaisesRegex(ContractError, "missing required columns"):
            load_and_validate(self.manifest_file, self.fold_file, self.profile)

    def test_missing_file(self):
        with self.assertRaisesRegex(ContractError, "cannot read"):
            load_and_validate(self.manifest_file, self.fold_file, self.profile)

    def test_mrs_label_mismatch(self):
        self.manifest[0]["mrs_3months"] = 5
        self.expect_failure("mrs_binary")

    def test_eligibility_flag_mismatch(self):
        self.manifest[0]["eligible_fusion_mrs"] = 0
        self.expect_failure("eligible_fusion_mrs disagrees")

    def test_fold_metadata_mismatch(self):
        self.folds[0]["has_gt"] = 0
        self.expect_failure("has_gt disagrees")

    def test_unrepresented_cohort_outcome_stratum(self):
        # Keep aggregate and per-fold label counts unchanged, but place one
        # cohort/outcome stratum in only four of the five folds.
        self.manifest[0]["cohort"] = "03"
        self.folds[0]["cohort"] = "03"
        self.folds[0]["stratum"] = "c03_good"
        self.profile = replace(self.profile, fingerprint=synthetic_fingerprint(self.folds))
        self.expect_failure("does not represent every cohort")

    def test_path_overrides(self):
        root = Path(self.temp.name)
        paths = ServerPaths.from_overrides(project_root=root, environ={})
        self.assertEqual(paths.manifest, root / "outputs/tables/patient_manifest.csv")
        self.assertEqual(paths.folds, root / "outputs/tables/outcome_cv_folds_private.csv")

    def test_cli_pass_summary_contains_no_patient_ids(self):
        contract = self.validate()
        output = io.StringIO()
        with patch("validate_active_dataset.ServerPaths.check_server_locations"), \
             patch("validate_active_dataset.load_and_validate", return_value=contract), \
             redirect_stdout(output):
            code = preflight_main([])
        self.assertEqual(code, 0)
        self.assertIn("Status: PASS", output.getvalue())
        self.assertNotIn("fictional-case", output.getvalue())

    def test_cli_failure_exits_nonzero(self):
        output = io.StringIO()
        with redirect_stderr(output):
            code = preflight_main(["--project-root", str(Path(self.temp.name) / "absent")])
        self.assertEqual(code, 1)
        self.assertIn("Status: FAIL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
