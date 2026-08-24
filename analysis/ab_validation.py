"""Deterministic synthetic validation for the repository's A/B design.

This module tests statistical machinery; it does not estimate product or business
impact.  All generated observations are synthetic.  The simulator deliberately
creates repeated impressions within users, then performs primary inference on one
equal-weight CTR per randomized user.

The default command writes JSON to stdout.  Use ``--json-out`` to also persist the
same payload::

    python analysis/ab_validation.py \
        --json-out analysis/ab_validation_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_STOPPING_SCHEMA = ANALYSIS_DIR / "stopping_rule.schema.json"
DEFAULT_STOPPING_DECLARATION = ANALYSIS_DIR / "stopping_rule.synthetic-example.json"


@dataclass(frozen=True)
class UserAggregate:
    """Sufficient statistics for one randomized user."""

    user_id: str
    variant: str
    clicks: int
    impressions: int


@dataclass(frozen=True)
class SimulationConfig:
    """Predeclared inputs for the synthetic validation run."""

    seed: int = 20260820
    repetitions: int = 2_500
    users_per_arm: int = 400
    baseline_click_probability: float = 0.10
    user_heterogeneity_half_width: float = 0.04
    minimum_impressions_per_user: int = 3
    mean_impressions_per_user: float = 8.0
    alpha: float = 0.05
    confidence_level: float = 0.95
    planted_effect: float = 0.02
    power_effects: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04)
    declared_minimum_detectable_effect: float = 0.03
    target_power: float = 0.80
    monte_carlo_acceptance_confidence: float = 0.999
    assignment_validation_users: int = 20_000
    assignment_salt: str = "reco-engine-adv/synthetic-validation/v1"
    srm_alpha: float = 0.001
    srm_total_users: int = 10_000
    srm_planted_treatment_fraction: float = 0.55
    srm_detection_target: float = 0.99

    def validate(self) -> None:
        if self.repetitions < 100:
            raise ValueError("repetitions must be at least 100")
        if self.users_per_arm < 2:
            raise ValueError("users_per_arm must be at least 2")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if self.minimum_impressions_per_user < 1:
            raise ValueError("minimum_impressions_per_user must be positive")
        if self.mean_impressions_per_user < self.minimum_impressions_per_user:
            raise ValueError("mean impressions cannot be below the minimum")
        low = self.baseline_click_probability - self.user_heterogeneity_half_width
        high = (
            self.baseline_click_probability
            + self.user_heterogeneity_half_width
            + max(self.power_effects + (self.planted_effect,))
        )
        if low <= 0 or high >= 1:
            raise ValueError("configured synthetic click probabilities leave (0, 1)")
        if self.declared_minimum_detectable_effect not in self.power_effects:
            raise ValueError("power_effects must include the declared MDE")
        if tuple(sorted(set(self.power_effects))) != self.power_effects:
            raise ValueError("power_effects must be unique and increasing")
        if not 0 < self.monte_carlo_acceptance_confidence < 1:
            raise ValueError("Monte Carlo acceptance confidence must be in (0, 1)")
        if self.assignment_validation_users < 2:
            raise ValueError("assignment_validation_users must be at least 2")
        if self.srm_total_users < 2:
            raise ValueError("srm_total_users must be at least 2")
        if not 0 < self.srm_alpha < 1:
            raise ValueError("srm_alpha must be between zero and one")
        if not 0 < self.srm_planted_treatment_fraction < 1:
            raise ValueError("planted treatment fraction must be in (0, 1)")


def _sha256_seed(base_seed: int, namespace: str) -> int:
    material = f"{base_seed}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment_digest(user_id: str, experiment_salt: str) -> bytes:
    return hashlib.sha256(f"{experiment_salt}\0{user_id}".encode("utf-8")).digest()


def production_user_assignment(user_id: str) -> str:
    """Mirror ``api/main.py::variant_for`` exactly for regression testing.

    The deployed API assigns on the parity of the complete SHA-256 digest.  Keep
    this separate from the salted threshold rule used by the synthetic design so
    the report cannot imply that a different randomization algorithm was tested.
    """

    if not user_id:
        raise ValueError("user_id must be non-empty")
    digest_as_int = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16)
    return "treatment" if digest_as_int % 2 else "control"


def stable_user_assignment(
    user_id: str,
    experiment_salt: str,
    treatment_fraction: float = 0.5,
) -> str:
    """Assign a user with SHA-256, independent of process hash randomization."""

    if not user_id:
        raise ValueError("user_id must be non-empty")
    if not experiment_salt:
        raise ValueError("experiment_salt must be non-empty")
    if not 0 < treatment_fraction < 1:
        raise ValueError("treatment_fraction must be between zero and one")

    fraction = Fraction(str(treatment_fraction))
    threshold = (fraction.numerator * (1 << 64)) // fraction.denominator
    bucket = int.from_bytes(_assignment_digest(user_id, experiment_salt)[:8], "big")
    return "treatment" if bucket < threshold else "control"


def sample_ratio_mismatch(
    observed: Mapping[str, int],
    expected_fractions: Mapping[str, float],
    *,
    alpha: float = 0.001,
) -> dict[str, Any]:
    """Pearson chi-square SRM check with explicitly declared expected allocation."""

    if set(observed) != set(expected_fractions):
        raise ValueError("observed and expected arms must match")
    arms = sorted(observed)
    counts = np.asarray([observed[arm] for arm in arms], dtype=np.float64)
    fractions = np.asarray([expected_fractions[arm] for arm in arms], dtype=np.float64)
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("observed counts must be non-negative with a positive total")
    if np.any(fractions <= 0) or not math.isclose(float(fractions.sum()), 1.0):
        raise ValueError("expected fractions must be positive and sum to one")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    expected = counts.sum() * fractions
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    degrees_of_freedom = len(arms) - 1
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return {
        "method": "Pearson chi-square goodness-of-fit",
        "arms": arms,
        "observed": {arm: int(observed[arm]) for arm in arms},
        "expected": {arm: float(expected[index]) for index, arm in enumerate(arms)},
        "statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "alpha": alpha,
        "mismatch_detected": p_value < alpha,
    }


def analyze_user_aggregates(
    users: Sequence[UserAggregate],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare equal-weight per-user CTRs with a two-sided Welch test."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    seen: set[str] = set()
    by_arm: dict[str, list[float]] = {"control": [], "treatment": []}
    for user in users:
        if user.user_id in seen:
            raise ValueError(f"duplicate user_id: {user.user_id}")
        seen.add(user.user_id)
        if user.variant not in by_arm:
            raise ValueError(f"unexpected variant: {user.variant}")
        if user.impressions <= 0:
            raise ValueError("impressions must be positive")
        if not 0 <= user.clicks <= user.impressions:
            raise ValueError("clicks must be between zero and impressions")
        by_arm[user.variant].append(user.clicks / user.impressions)

    control = np.asarray(by_arm["control"], dtype=np.float64)
    treatment = np.asarray(by_arm["treatment"], dtype=np.float64)
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("each arm needs at least two randomized users")

    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    difference = treatment_mean - control_mean
    control_component = float(control.var(ddof=1) / len(control))
    treatment_component = float(treatment.var(ddof=1) / len(treatment))
    standard_error = math.sqrt(control_component + treatment_component)
    if standard_error == 0:
        raise ValueError("user-level outcome variance is zero")
    df = (control_component + treatment_component) ** 2 / (
        control_component**2 / (len(control) - 1)
        + treatment_component**2 / (len(treatment) - 1)
    )
    critical = float(stats.t.ppf(1 - alpha / 2, df))
    test_statistic = difference / standard_error
    p_value = float(2 * stats.t.sf(abs(test_statistic), df))
    confidence_interval = [
        difference - critical * standard_error,
        difference + critical * standard_error,
    ]
    return {
        "method": "two-sided Welch t-test on equal-weight per-user CTR",
        "randomization_unit": "user_id",
        "analysis_unit": "user_id",
        "weighting": "one equal-weight CTR per randomized user",
        "warning": "impressions within a user are not treated as independent",
        "control": {
            "users": len(control),
            "mean_user_ctr": control_mean,
            "sample_standard_deviation": float(control.std(ddof=1)),
        },
        "treatment": {
            "users": len(treatment),
            "mean_user_ctr": treatment_mean,
            "sample_standard_deviation": float(treatment.std(ddof=1)),
        },
        "comparison": {
            "absolute_difference": difference,
            "standard_error": standard_error,
            "degrees_of_freedom": df,
            "test_statistic": test_statistic,
            "p_value_two_sided": p_value,
            "confidence_level": 1 - alpha,
            "confidence_interval": confidence_interval,
            "reject_null": p_value < alpha,
        },
    }


def _validate_user_level_inference() -> dict[str, Any]:
    """Demonstrate that repeated impressions cannot silently become the unit."""

    users = [
        UserAggregate("synthetic-c-1", "control", clicks=1, impressions=1),
        UserAggregate("synthetic-c-2", "control", clicks=0, impressions=100),
        UserAggregate("synthetic-c-3", "control", clicks=0, impressions=100),
        UserAggregate("synthetic-t-1", "treatment", clicks=1, impressions=2),
        UserAggregate("synthetic-t-2", "treatment", clicks=1, impressions=2),
        UserAggregate("synthetic-t-3", "treatment", clicks=0, impressions=2),
    ]
    primary = analyze_user_aggregates(users)
    totals = {
        arm: {
            "clicks": sum(user.clicks for user in users if user.variant == arm),
            "impressions": sum(
                user.impressions for user in users if user.variant == arm
            ),
        }
        for arm in ("control", "treatment")
    }
    impression_weighted_difference = (
        totals["treatment"]["clicks"] / totals["treatment"]["impressions"]
        - totals["control"]["clicks"] / totals["control"]["impressions"]
    )
    user_level_difference = primary["comparison"]["absolute_difference"]
    passes = (
        primary["randomization_unit"] == "user_id"
        and primary["analysis_unit"] == "user_id"
        and math.isclose(user_level_difference, 0.0, abs_tol=1e-15)
        and impression_weighted_difference > 0.3
    )
    return {
        "synthetic": True,
        "purpose": (
            "adversarial unequal-impression fixture proving that the primary method "
            "weights randomized users equally"
        ),
        "primary_user_level_result": primary,
        "descriptive_impression_weighted_difference": impression_weighted_difference,
        "expected_user_level_difference": 0.0,
        "passes": passes,
    }


def _simulate_user_level_trials(
    config: SimulationConfig,
    effect: float,
    *,
    namespace: str,
) -> dict[str, np.ndarray]:
    """Vectorized repeated experiments with clustered impressions per user."""

    rng = np.random.default_rng(_sha256_seed(config.seed, namespace))
    shape = (config.repetitions, config.users_per_arm)
    poisson_mean = config.mean_impressions_per_user - config.minimum_impressions_per_user

    control_impressions = config.minimum_impressions_per_user + rng.poisson(
        poisson_mean, size=shape
    )
    treatment_impressions = config.minimum_impressions_per_user + rng.poisson(
        poisson_mean, size=shape
    )
    control_probability = config.baseline_click_probability + rng.uniform(
        -config.user_heterogeneity_half_width,
        config.user_heterogeneity_half_width,
        size=shape,
    )
    treatment_probability = (
        config.baseline_click_probability
        + effect
        + rng.uniform(
            -config.user_heterogeneity_half_width,
            config.user_heterogeneity_half_width,
            size=shape,
        )
    )
    control_ctr = rng.binomial(control_impressions, control_probability) / control_impressions
    treatment_ctr = (
        rng.binomial(treatment_impressions, treatment_probability) / treatment_impressions
    )

    control_mean = control_ctr.mean(axis=1)
    treatment_mean = treatment_ctr.mean(axis=1)
    difference = treatment_mean - control_mean
    control_component = control_ctr.var(axis=1, ddof=1) / config.users_per_arm
    treatment_component = treatment_ctr.var(axis=1, ddof=1) / config.users_per_arm
    standard_error = np.sqrt(control_component + treatment_component)
    degrees_of_freedom = (control_component + treatment_component) ** 2 / (
        control_component**2 / (config.users_per_arm - 1)
        + treatment_component**2 / (config.users_per_arm - 1)
    )
    statistic = difference / standard_error
    p_value = 2 * stats.t.sf(np.abs(statistic), degrees_of_freedom)
    critical = stats.t.ppf(1 - config.alpha / 2, degrees_of_freedom)
    lower = difference - critical * standard_error
    upper = difference + critical * standard_error
    return {
        "difference": difference,
        "standard_error": standard_error,
        "p_value": p_value,
        "reject": p_value < config.alpha,
        "contains_effect": (lower <= effect) & (effect <= upper),
        "lower": lower,
        "upper": upper,
    }


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return [max(0.0, centre - half_width), min(1.0, centre + half_width)]


def _exact_monte_carlo_acceptance_band(
    target_rate: float,
    repetitions: int,
    confidence: float,
) -> dict[str, Any]:
    """Central exact binomial band for a correctly calibrated simulation count."""

    tail = (1 - confidence) / 2
    minimum_successes = int(stats.binom.ppf(tail, repetitions, target_rate))
    maximum_successes = int(stats.binom.ppf(1 - tail, repetitions, target_rate))
    return {
        "method": "central exact binomial acceptance band",
        "confidence": confidence,
        "target_rate": target_rate,
        "minimum_successes": minimum_successes,
        "maximum_successes": maximum_successes,
        "rate_interval": [
            minimum_successes / repetitions,
            maximum_successes / repetitions,
        ],
    }


def _rate_validation(
    successes: int,
    repetitions: int,
    *,
    target_rate: float,
    acceptance_confidence: float,
) -> dict[str, Any]:
    band = _exact_monte_carlo_acceptance_band(
        target_rate, repetitions, acceptance_confidence
    )
    estimate = successes / repetitions
    return {
        "successes": successes,
        "repetitions": repetitions,
        "estimate": estimate,
        "monte_carlo_standard_error": math.sqrt(estimate * (1 - estimate) / repetitions),
        "monte_carlo_95_ci": _wilson_interval(successes, repetitions),
        "design_target": target_rate,
        "acceptance_band": band,
        "passes_acceptance_band": (
            band["minimum_successes"] <= successes <= band["maximum_successes"]
        ),
    }


def _validate_assignment(config: SimulationConfig) -> dict[str, Any]:
    user_ids = [
        f"synthetic-user-{index:06d}" for index in range(config.assignment_validation_users)
    ]
    first = [stable_user_assignment(user_id, config.assignment_salt) for user_id in user_ids]
    repeated = [stable_user_assignment(user_id, config.assignment_salt) for user_id in user_ids]
    reverse_lookup = {
        user_id: stable_user_assignment(user_id, config.assignment_salt)
        for user_id in reversed(user_ids)
    }
    observed = {
        "control": first.count("control"),
        "treatment": first.count("treatment"),
    }
    srm = sample_ratio_mismatch(
        observed,
        {"control": 0.5, "treatment": 0.5},
        alpha=config.srm_alpha,
    )
    golden_vectors = []
    for user_id, variant in zip(user_ids[:8], first[:8]):
        digest = _assignment_digest(user_id, config.assignment_salt)
        golden_vectors.append(
            {
                "user_id": user_id,
                "sha256_prefix_64_bit_hex": digest[:8].hex(),
                "variant": variant,
            }
        )
    repeated_consistent = first == repeated
    order_invariant = all(
        variant == reverse_lookup[user_id] for user_id, variant in zip(user_ids, first)
    )
    production_first = [production_user_assignment(user_id) for user_id in user_ids]
    production_repeated = [production_user_assignment(user_id) for user_id in user_ids]
    production_observed = {
        "control": production_first.count("control"),
        "treatment": production_first.count("treatment"),
    }
    production_srm = sample_ratio_mismatch(
        production_observed,
        {"control": 0.5, "treatment": 0.5},
        alpha=config.srm_alpha,
    )
    production_golden_vectors = [
        {
            "user_id": user_id,
            "sha256_hex": hashlib.sha256(user_id.encode("utf-8")).hexdigest(),
            "variant": variant,
        }
        for user_id, variant in zip(user_ids[:8], production_first[:8])
    ]
    production_passes = (
        production_first == production_repeated and not production_srm["mismatch_detected"]
    )
    synthetic_passes = (
        repeated_consistent and order_invariant and not srm["mismatch_detected"]
    )
    return {
        "synthetic": True,
        "randomization_unit": "user_id",
        "users_checked": len(user_ids),
        "one_variant_per_user": True,
        "production_api_rule": {
            "source": "api/main.py::variant_for",
            "algorithm": "parity of the complete SHA-256(user_id UTF-8) digest",
            "treatment_fraction": 0.5,
            "repeat_consistent": production_first == production_repeated,
            "observed_counts": production_observed,
            "allocation_srm_check": production_srm,
            "golden_vectors": production_golden_vectors,
            "passes": production_passes,
        },
        "synthetic_design_rule": {
            "algorithm": (
                "SHA-256 of UTF-8 experiment_salt, NUL delimiter, and user_id; first "
                "64 bits compared with an integer allocation threshold"
            ),
            "experiment_salt": config.assignment_salt,
            "treatment_fraction": 0.5,
            "repeat_consistent": repeated_consistent,
            "order_invariant": order_invariant,
            "observed_counts": observed,
            "allocation_srm_check": srm,
            "golden_vectors": golden_vectors,
            "passes": synthetic_passes,
        },
        "passes": production_passes and synthetic_passes,
    }


def _validate_aa_and_coverage(config: SimulationConfig) -> dict[str, Any]:
    aa = _simulate_user_level_trials(config, 0.0, namespace="aa")
    false_positives = int(np.count_nonzero(aa["reject"]))
    false_positive_validation = _rate_validation(
        false_positives,
        config.repetitions,
        target_rate=config.alpha,
        acceptance_confidence=config.monte_carlo_acceptance_confidence,
    )

    coverage_trials = _simulate_user_level_trials(
        config,
        config.planted_effect,
        namespace="planted-effect-coverage",
    )
    covered = int(np.count_nonzero(coverage_trials["contains_effect"]))
    coverage_validation = _rate_validation(
        covered,
        config.repetitions,
        target_rate=config.confidence_level,
        acceptance_confidence=config.monte_carlo_acceptance_confidence,
    )
    coverage_validation.update(
        {
            "planted_absolute_effect": config.planted_effect,
            "mean_estimated_effect": float(coverage_trials["difference"].mean()),
            "simulation_bias": float(
                coverage_trials["difference"].mean() - config.planted_effect
            ),
        }
    )
    return {
        "aa_false_positive_rate": false_positive_validation,
        "planted_effect_ci_coverage": coverage_validation,
        "passes": (
            false_positive_validation["passes_acceptance_band"]
            and coverage_validation["passes_acceptance_band"]
        ),
    }


def _validate_power_curve(config: SimulationConfig) -> dict[str, Any]:
    points = []
    for effect in config.power_effects:
        trials = _simulate_user_level_trials(
            config,
            effect,
            namespace=f"power:{effect:.12g}",
        )
        rejections = int(np.count_nonzero(trials["reject"]))
        point: dict[str, Any] = {
            "planted_absolute_effect": effect,
            "rejections": rejections,
            "repetitions": config.repetitions,
            "estimated_power": rejections / config.repetitions,
            "monte_carlo_95_ci": _wilson_interval(rejections, config.repetitions),
            "mean_estimated_effect": float(trials["difference"].mean()),
        }
        if effect == 0:
            point["null_calibration"] = _rate_validation(
                rejections,
                config.repetitions,
                target_rate=config.alpha,
                acceptance_confidence=config.monte_carlo_acceptance_confidence,
            )
        points.append(point)

    comparisons = []
    monotonic_with_uncertainty = True
    for left, right in zip(points, points[1:]):
        left_power = left["estimated_power"]
        right_power = right["estimated_power"]
        difference = right_power - left_power
        delta_mc_se = math.sqrt(
            left_power * (1 - left_power) / config.repetitions
            + right_power * (1 - right_power) / config.repetitions
        )
        # A reversal is tolerated only when it is no larger than a 99.9% normal
        # Monte Carlo error bound. Effects are simulated independently.
        allowed_reversal = float(
            stats.norm.ppf(0.9995) * delta_mc_se + 1 / config.repetitions
        )
        passes = difference >= -allowed_reversal
        monotonic_with_uncertainty &= passes
        comparisons.append(
            {
                "from_effect": left["planted_absolute_effect"],
                "to_effect": right["planted_absolute_effect"],
                "observed_power_difference": difference,
                "maximum_accepted_monte_carlo_reversal": allowed_reversal,
                "passes": passes,
            }
        )

    mde_point = next(
        point
        for point in points
        if point["planted_absolute_effect"]
        == config.declared_minimum_detectable_effect
    )
    mde_passes = mde_point["monte_carlo_95_ci"][0] >= config.target_power
    null_passes = points[0]["null_calibration"]["passes_acceptance_band"]
    return {
        "points": points,
        "monotonicity_checks": comparisons,
        "monotonic_with_monte_carlo_uncertainty": monotonic_with_uncertainty,
        "declared_mde_check": {
            "minimum_detectable_effect": config.declared_minimum_detectable_effect,
            "target_power": config.target_power,
            "estimated_power": mde_point["estimated_power"],
            "monte_carlo_95_ci": mde_point["monte_carlo_95_ci"],
            "criterion": "lower bound of Monte Carlo 95% CI is at least target power",
            "passes": mde_passes,
        },
        "passes": monotonic_with_uncertainty and mde_passes and null_passes,
    }


def _srm_statistics_from_treatment_counts(
    treatment_counts: np.ndarray,
    total_users: int,
    expected_treatment_fraction: float,
) -> np.ndarray:
    expected_treatment = total_users * expected_treatment_fraction
    expected_control = total_users - expected_treatment
    control_counts = total_users - treatment_counts
    return (
        (treatment_counts - expected_treatment) ** 2 / expected_treatment
        + (control_counts - expected_control) ** 2 / expected_control
    )


def _validate_srm(config: SimulationConfig) -> dict[str, Any]:
    expected = {"control": 0.5, "treatment": 0.5}
    balanced = sample_ratio_mismatch(
        {"control": config.srm_total_users // 2, "treatment": config.srm_total_users // 2},
        expected,
        alpha=config.srm_alpha,
    )
    planted_treatment = round(
        config.srm_total_users * config.srm_planted_treatment_fraction
    )
    planted = sample_ratio_mismatch(
        {
            "control": config.srm_total_users - planted_treatment,
            "treatment": planted_treatment,
        },
        expected,
        alpha=config.srm_alpha,
    )

    null_rng = np.random.default_rng(_sha256_seed(config.seed, "srm-null"))
    null_treatment_counts = null_rng.binomial(
        config.srm_total_users, 0.5, size=config.repetitions
    )
    null_statistics = _srm_statistics_from_treatment_counts(
        null_treatment_counts, config.srm_total_users, 0.5
    )
    null_p_values = stats.chi2.sf(null_statistics, 1)
    false_alarms = int(np.count_nonzero(null_p_values < config.srm_alpha))
    false_alarm_validation = _rate_validation(
        false_alarms,
        config.repetitions,
        target_rate=config.srm_alpha,
        acceptance_confidence=config.monte_carlo_acceptance_confidence,
    )

    planted_rng = np.random.default_rng(_sha256_seed(config.seed, "srm-planted"))
    planted_counts = planted_rng.binomial(
        config.srm_total_users,
        config.srm_planted_treatment_fraction,
        size=config.repetitions,
    )
    planted_statistics = _srm_statistics_from_treatment_counts(
        planted_counts, config.srm_total_users, 0.5
    )
    planted_p_values = stats.chi2.sf(planted_statistics, 1)
    detections = int(np.count_nonzero(planted_p_values < config.srm_alpha))
    detection_interval = _wilson_interval(detections, config.repetitions)
    detection_validation = {
        "planted_treatment_fraction": config.srm_planted_treatment_fraction,
        "detections": detections,
        "repetitions": config.repetitions,
        "estimated_detection_probability": detections / config.repetitions,
        "monte_carlo_95_ci": detection_interval,
        "target": config.srm_detection_target,
        "criterion": "lower bound of Monte Carlo 95% CI is at least target",
        "passes": detection_interval[0] >= config.srm_detection_target,
    }
    return {
        "alpha": config.srm_alpha,
        "expected_allocation": expected,
        "balanced_example": balanced,
        "planted_mismatch_example": planted,
        "null_false_alarm_rate": false_alarm_validation,
        "planted_mismatch_detection": detection_validation,
        "passes": (
            not balanced["mismatch_detected"]
            and planted["mismatch_detected"]
            and false_alarm_validation["passes_acceptance_band"]
            and detection_validation["passes"]
        ),
    }


def validate_stopping_declaration(
    declaration_path: Path = DEFAULT_STOPPING_DECLARATION,
    schema_path: Path = DEFAULT_STOPPING_SCHEMA,
) -> dict[str, Any]:
    """Validate the fixed-horizon declaration against its JSON Schema."""

    from jsonschema import Draft202012Validator, FormatChecker

    schema = json.loads(schema_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(declaration), key=lambda error: list(error.path))
    schema_errors = [
        {
            "path": "/" + "/".join(str(part) for part in error.path),
            "message": error.message,
        }
        for error in errors
    ]
    semantic_errors = []
    inference = declaration.get("inference", {})
    sample_size = declaration.get("sample_size", {})
    assignment = declaration.get("assignment", {})
    primary_metric = declaration.get("primary_metric", {})
    if not math.isclose(
        inference.get("confidence_level", math.nan),
        1 - inference.get("alpha", math.nan),
        abs_tol=1e-12,
    ):
        semantic_errors.append("confidence_level must equal 1 - alpha")
    if sample_size.get("planned_total_users", 0) < 2 * sample_size.get(
        "minimum_users_per_arm", math.inf
    ):
        semantic_errors.append(
            "planned_total_users must cover minimum_users_per_arm in both arms"
        )
    if assignment.get("treatment_fraction") == 0.5 and (
        sample_size.get("planned_total_users", 0) % 2
    ):
        semantic_errors.append("equal allocation requires an even planned_total_users")
    if assignment.get("arms") == ["control", "treatment"] and (
        declaration.get("population", {}).get("randomization_unit")
        != primary_metric.get("analysis_unit")
    ):
        semantic_errors.append("randomization_unit and primary analysis_unit must match")

    def display_path(path: Path) -> str:
        try:
            return path.relative_to(ANALYSIS_DIR.parent).as_posix()
        except ValueError:
            return str(path)

    is_valid = not schema_errors and not semantic_errors
    return {
        "valid": is_valid,
        "schema_errors": schema_errors,
        "semantic_errors": semantic_errors,
        "schema": {
            "path": display_path(schema_path),
            "sha256": _file_sha256(schema_path),
            "draft": schema["$schema"],
        },
        "declaration": {
            "path": display_path(declaration_path),
            "sha256": _file_sha256(declaration_path),
            "experiment_id": declaration.get("experiment_id"),
            "synthetic_example": declaration.get("synthetic_example"),
            "outcomes_viewed_at_declaration": declaration.get("predeclaration", {}).get(
                "outcomes_viewed_at_declaration"
            ),
            "stopping_type": declaration.get("stopping_rule", {}).get("type"),
            "randomization_unit": declaration.get("population", {}).get(
                "randomization_unit"
            ),
            "analysis_unit": primary_metric.get("analysis_unit"),
            "alpha": inference.get("alpha"),
            "confidence_level": inference.get("confidence_level"),
            "planned_total_users": sample_size.get("planned_total_users"),
            "minimum_detectable_absolute_effect": sample_size.get(
                "minimum_detectable_absolute_effect"
            ),
            "target_power": sample_size.get("target_power"),
        },
        "passes": is_valid,
    }


def build_validation_report(
    config: SimulationConfig | None = None,
    *,
    stopping_declaration: Path = DEFAULT_STOPPING_DECLARATION,
    stopping_schema: Path = DEFAULT_STOPPING_SCHEMA,
) -> dict[str, Any]:
    """Run every validation and return a deterministic JSON-compatible payload."""

    config = config or SimulationConfig()
    config.validate()
    assignment = _validate_assignment(config)
    user_level_inference = _validate_user_level_inference()
    calibration = _validate_aa_and_coverage(config)
    power = _validate_power_curve(config)
    srm = _validate_srm(config)
    stopping = validate_stopping_declaration(stopping_declaration, stopping_schema)
    declared = stopping["declaration"]
    alignment_checks = {
        "alpha": math.isclose(declared.get("alpha", math.nan), config.alpha),
        "confidence_level": math.isclose(
            declared.get("confidence_level", math.nan), config.confidence_level
        ),
        "planned_total_users": declared.get("planned_total_users")
        == 2 * config.users_per_arm,
        "minimum_detectable_absolute_effect": math.isclose(
            declared.get("minimum_detectable_absolute_effect", math.nan),
            config.declared_minimum_detectable_effect,
        ),
        "target_power": math.isclose(
            declared.get("target_power", math.nan), config.target_power
        ),
        "randomization_and_analysis_unit": (
            declared.get("randomization_unit")
            == declared.get("analysis_unit")
            == "user_id"
        ),
    }
    stopping["simulation_configuration_alignment"] = {
        "checks": alignment_checks,
        "passes": all(alignment_checks.values()),
    }
    stopping["passes"] = stopping["passes"] and all(alignment_checks.values())
    checks = {
        "stable_user_assignment": assignment["passes"],
        "aa_false_positive_rate": calibration["aa_false_positive_rate"][
            "passes_acceptance_band"
        ],
        "planted_effect_ci_coverage": calibration["planted_effect_ci_coverage"][
            "passes_acceptance_band"
        ],
        "power_curve": power["passes"],
        "sample_ratio_mismatch": srm["passes"],
        "predeclared_stopping_rule": stopping["passes"],
        "user_level_assignment_matches_inference": user_level_inference["passes"],
    }
    return {
        "report_version": "1.0.0",
        "claim_scope": {
            "synthetic": True,
            "purpose": "validation of A/B statistical machinery",
            "business_impact_estimate": False,
            "warning": (
                "All observations and planted lifts in this report are synthetic. "
                "They are not evidence of product, revenue, or business impact."
            ),
        },
        "design": {
            "randomization_unit": "user_id",
            "primary_analysis_unit": "user_id",
            "primary_outcome": "equal-weight per-user click-through rate",
            "inference": "two-sided Welch t-test",
            "repeated_impressions_clustered_within_user": True,
        },
        "reproducibility": {
            "deterministic": True,
            "seed": config.seed,
            "seed_derivation": "SHA-256(base seed, NUL delimiter, validation namespace)",
            "configuration": asdict(config),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "jsonschema")
            },
        },
        "monte_carlo_policy": {
            "repetitions": config.repetitions,
            "reported_uncertainty": "Wilson 95% interval for simulated rates",
            "calibration_acceptance": (
                "central exact binomial band under the predeclared design target"
            ),
            "acceptance_confidence": config.monte_carlo_acceptance_confidence,
        },
        "stable_assignment": assignment,
        "user_level_inference": user_level_inference,
        "aa_and_coverage": calibration,
        "power_curve": power,
        "sample_ratio_mismatch": srm,
        "stopping_rule": stopping,
        "validation_checks": checks,
        "overall_pass": all(checks.values()),
    }


def _parse_effects(raw: str) -> tuple[float, ...]:
    effects = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not effects:
        raise argparse.ArgumentTypeError("at least one power effect is required")
    return effects


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    parser.add_argument("--repetitions", type=int, default=SimulationConfig.repetitions)
    parser.add_argument("--users-per-arm", type=int, default=SimulationConfig.users_per_arm)
    parser.add_argument(
        "--power-effects",
        type=_parse_effects,
        default=SimulationConfig.power_effects,
        help="comma-separated planted absolute CTR effects",
    )
    parser.add_argument(
        "--stopping-declaration", type=Path, default=DEFAULT_STOPPING_DECLARATION
    )
    parser.add_argument("--stopping-schema", type=Path, default=DEFAULT_STOPPING_SCHEMA)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = replace(
        SimulationConfig(),
        seed=args.seed,
        repetitions=args.repetitions,
        users_per_arm=args.users_per_arm,
        power_effects=args.power_effects,
    )
    report = build_validation_report(
        config,
        stopping_declaration=args.stopping_declaration,
        stopping_schema=args.stopping_schema,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
        print(f"wrote {args.json_out}", file=sys.stderr)
    sys.stdout.write(rendered)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
