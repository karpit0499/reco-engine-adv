from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPOSITORY_ROOT / "analysis"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.ab_validation import (  # noqa: E402
    SimulationConfig,
    UserAggregate,
    analyze_user_aggregates,
    build_validation_report,
    production_user_assignment,
    sample_ratio_mismatch,
    stable_user_assignment,
)


class StableAssignmentTests(unittest.TestCase):
    def test_production_assignment_matches_api_golden_results(self) -> None:
        expected = {
            "alice": "control",
            "bob": "treatment",
            "carol": "treatment",
            "synthetic-user-000000": "treatment",
            "synthetic-user-000001": "control",
        }
        self.assertEqual(
            expected,
            {user_id: production_user_assignment(user_id) for user_id in expected},
        )

    def test_sha256_assignment_has_stable_golden_results(self) -> None:
        salt = "reco-engine-adv/synthetic-validation/v1"
        expected = {
            "alice": "treatment",
            "bob": "control",
            "carol": "control",
            "synthetic-user-000000": "treatment",
            "synthetic-user-000001": "control",
        }
        actual = {
            user_id: stable_user_assignment(user_id, salt)
            for user_id in reversed(list(expected))
        }
        self.assertEqual(expected, actual)
        self.assertEqual(actual, {
            user_id: stable_user_assignment(user_id, salt)
            for user_id in reversed(list(expected))
        })

    def test_assignment_rejects_missing_identity_or_salt(self) -> None:
        with self.assertRaises(ValueError):
            production_user_assignment("")
        with self.assertRaises(ValueError):
            stable_user_assignment("", "salt")
        with self.assertRaises(ValueError):
            stable_user_assignment("user", "")


class UserLevelInferenceTests(unittest.TestCase):
    def test_inference_equal_weights_users_not_impressions(self) -> None:
        users = [
            UserAggregate("c-1", "control", clicks=1, impressions=1),
            UserAggregate("c-2", "control", clicks=0, impressions=100),
            UserAggregate("c-3", "control", clicks=0, impressions=100),
            UserAggregate("t-1", "treatment", clicks=1, impressions=2),
            UserAggregate("t-2", "treatment", clicks=1, impressions=2),
            UserAggregate("t-3", "treatment", clicks=0, impressions=2),
        ]
        result = analyze_user_aggregates(users)

        # Both arms have mean per-user CTR 1/3. Impression weighting would instead
        # produce a large apparent treatment lift (2/6 versus 1/201).
        self.assertAlmostEqual(result["comparison"]["absolute_difference"], 0.0)
        self.assertEqual(result["randomization_unit"], "user_id")
        self.assertEqual(result["analysis_unit"], "user_id")
        self.assertEqual(result["control"]["users"], 3)
        self.assertEqual(result["treatment"]["users"], 3)
        self.assertGreater((2 / 6) - (1 / 201), 0.3)

    def test_duplicate_randomized_user_is_rejected(self) -> None:
        users = [
            UserAggregate("duplicate", "control", 0, 1),
            UserAggregate("duplicate", "treatment", 1, 1),
            UserAggregate("c-2", "control", 0, 1),
            UserAggregate("t-2", "treatment", 1, 1),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate user_id"):
            analyze_user_aggregates(users)


class SampleRatioMismatchTests(unittest.TestCase):
    def test_balanced_allocation_passes_and_planted_mismatch_fails(self) -> None:
        expected = {"control": 0.5, "treatment": 0.5}
        balanced = sample_ratio_mismatch(
            {"control": 5_000, "treatment": 5_000}, expected, alpha=0.001
        )
        planted = sample_ratio_mismatch(
            {"control": 4_500, "treatment": 5_500}, expected, alpha=0.001
        )
        self.assertFalse(balanced["mismatch_detected"])
        self.assertTrue(planted["mismatch_detected"])
        self.assertLess(planted["p_value"], 0.001)


class StoppingRuleSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads((ANALYSIS_DIR / "stopping_rule.schema.json").read_text())
        cls.declaration = json.loads(
            (ANALYSIS_DIR / "stopping_rule.synthetic-example.json").read_text()
        )
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_predeclared_fixed_horizon_example_is_valid(self) -> None:
        self.assertTrue(self.validator.is_valid(self.declaration))
        self.assertFalse(
            self.declaration["predeclaration"]["outcomes_viewed_at_declaration"]
        )
        self.assertEqual(self.declaration["stopping_rule"]["max_primary_analyses"], 1)

    def test_schema_rejects_outcome_peeking(self) -> None:
        invalid = copy.deepcopy(self.declaration)
        invalid["predeclaration"]["outcomes_viewed_at_declaration"] = True
        self.assertFalse(self.validator.is_valid(invalid))

    def test_schema_rejects_unplanned_early_stopping(self) -> None:
        invalid = copy.deepcopy(self.declaration)
        invalid["stopping_rule"]["early_efficacy_stopping"] = "peek_daily"
        invalid["stopping_rule"]["max_primary_analyses"] = 99
        self.assertFalse(self.validator.is_valid(invalid))


class SimulationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = replace(
            SimulationConfig(),
            repetitions=600,
            assignment_validation_users=2_000,
            power_effects=(0.0, 0.03, 0.04),
        )
        cls.report = build_validation_report(cls.config)

    def test_calibration_and_coverage_use_predeclared_mc_bands(self) -> None:
        aa = self.report["aa_and_coverage"]["aa_false_positive_rate"]
        coverage = self.report["aa_and_coverage"]["planted_effect_ci_coverage"]
        self.assertTrue(aa["passes_acceptance_band"])
        self.assertTrue(coverage["passes_acceptance_band"])
        self.assertEqual(aa["acceptance_band"]["target_rate"], self.config.alpha)
        self.assertEqual(
            coverage["acceptance_band"]["target_rate"], self.config.confidence_level
        )
        self.assertIn("monte_carlo_95_ci", aa)
        self.assertIn("monte_carlo_95_ci", coverage)

    def test_power_curve_and_srm_validation_pass(self) -> None:
        self.assertTrue(self.report["power_curve"]["passes"])
        self.assertTrue(
            self.report["power_curve"]["monotonic_with_monte_carlo_uncertainty"]
        )
        self.assertTrue(self.report["sample_ratio_mismatch"]["passes"])
        self.assertTrue(self.report["stable_assignment"]["passes"])

    def test_every_simulated_claim_is_explicitly_labeled(self) -> None:
        scope = self.report["claim_scope"]
        self.assertTrue(scope["synthetic"])
        self.assertFalse(scope["business_impact_estimate"])
        self.assertIn("not evidence", scope["warning"])

    def test_same_seed_produces_identical_report(self) -> None:
        self.assertEqual(self.report, build_validation_report(self.config))

    def test_all_required_validation_checks_pass(self) -> None:
        self.assertTrue(self.report["overall_pass"])
        self.assertTrue(all(self.report["validation_checks"].values()))


class CommandLineReportTests(unittest.TestCase):
    def test_cli_stdout_and_written_json_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.json"
            command = [
                sys.executable,
                str(ANALYSIS_DIR / "ab_validation.py"),
                "--repetitions",
                "600",
                "--power-effects",
                "0,0.03,0.04",
                "--json-out",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(completed.stdout), json.loads(output.read_text()))
            self.assertTrue(json.loads(completed.stdout)["overall_pass"])


class CheckedReportTests(unittest.TestCase):
    def test_checked_default_report_is_machine_readable_and_passes(self) -> None:
        report_path = ANALYSIS_DIR / "ab_validation_report.json"
        report = json.loads(report_path.read_text())
        self.assertTrue(report["overall_pass"])
        self.assertTrue(report["claim_scope"]["synthetic"])
        self.assertEqual(report["reproducibility"]["configuration"]["repetitions"], 2_500)
        self.assertTrue(all(report["validation_checks"].values()))


if __name__ == "__main__":
    unittest.main()
