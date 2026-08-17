"""Reproduce the synthetic A/B result at the user randomization unit.

The API assigns variants by hashing ``user_id``, so repeated impressions from the
same user are correlated. The primary analysis gives every user one equal-weight
CTR and compares arms with Welch's t-test. Impression totals are descriptive; a
cluster-robust linear probability model is a secondary sensitivity analysis.

By default this reads the sanitized checked-in fixture and needs no GCP access.
Pass ``--refresh-from-gcp`` to rebuild that fixture with read-only cloud queries.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

from metrics_fixture import (
    CUTOFF,
    CUTOFF_IMPRESSION_ID,
    DEFAULT_FIXTURE,
    fixture_fingerprint,
    load_fixture,
    refresh_fixture_from_gcp,
)


ap = argparse.ArgumentParser()
ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
ap.add_argument(
    "--cutoff",
    default=CUTOFF,
    help="snapshot cutoff used only with --refresh-from-gcp",
)
ap.add_argument(
    "--cutoff-impression-id",
    default=CUTOFF_IMPRESSION_ID,
    help="stable tie-break at the exact cutoff, used only when refreshing",
)
ap.add_argument(
    "--refresh-from-gcp",
    action="store_true",
    help="replace the sanitized fixture using read-only queries, then evaluate it",
)
ap.add_argument("--json-out", type=Path)
args = ap.parse_args()

if args.refresh_from_gcp:
    refresh_fixture_from_gcp(
        args.fixture,
        cutoff=args.cutoff,
        cutoff_impression_id=args.cutoff_impression_id,
    )
fixture = load_fixture(args.fixture)
records = fixture["experiment"]["per_user"]
if not records:
    raise SystemExit("fixture has no experiment users")

expected_arms = {"control", "treatment"}
present_arms = {row["variant"] for row in records}
if present_arms != expected_arms:
    raise SystemExit(f"expected arms {sorted(expected_arms)}, found {sorted(present_arms)}")
if len({row["user_id"] for row in records}) != len(records):
    raise RuntimeError("fixture has duplicate user rows")

user_ctrs = {
    arm: np.asarray([
        row["clicks"] / row["impressions"]
        for row in records
        if row["variant"] == arm
    ], dtype=np.float64)
    for arm in sorted(expected_arms)
}
t_users, c_users = user_ctrs["treatment"], user_ctrs["control"]

# Primary inference: equal weight per randomized user.
t_mean, c_mean = float(t_users.mean()), float(c_users.mean())
user_diff = t_mean - c_mean
t_var = float(t_users.var(ddof=1)) / len(t_users)
c_var = float(c_users.var(ddof=1)) / len(c_users)
user_se = (t_var + c_var) ** 0.5
welch_df = (t_var + c_var) ** 2 / (
    t_var ** 2 / (len(t_users) - 1) + c_var ** 2 / (len(c_users) - 1)
)
welch_t = user_diff / user_se
welch_p = float(2 * stats.t.sf(abs(welch_t), welch_df))
welch_critical = float(stats.t.ppf(0.975, welch_df))
user_ci = (
    user_diff - welch_critical * user_se,
    user_diff + welch_critical * user_se,
)

# Descriptive impression totals. These are not independent observations.
by_arm = {
    arm: {
        "impressions": sum(row["impressions"] for row in records if row["variant"] == arm),
        "clicks": sum(row["clicks"] for row in records if row["variant"] == arm),
    }
    for arm in sorted(expected_arms)
}
for arm in by_arm:
    by_arm[arm]["ctr"] = by_arm[arm]["clicks"] / by_arm[arm]["impressions"]
t_ctr, c_ctr = by_arm["treatment"]["ctr"], by_arm["control"]["ctr"]

# Per-user counts are sufficient to recreate the row-level binary outcome exactly;
# order and impression IDs do not enter this model. Expanding them here reproduces
# statsmodels' original-N cluster small-sample correction exactly.
y_values, treatment_values, group_values = [], [], []
for row in records:
    clicks = int(row["clicks"])
    misses = int(row["impressions"]) - clicks
    y_values.extend([1.0] * clicks + [0.0] * misses)
    treatment_values.extend([row["variant"] == "treatment"] * row["impressions"])
    group_values.extend([row["user_id"]] * row["impressions"])
y = np.asarray(y_values, dtype=np.float64)
treatment = np.asarray(treatment_values, dtype=np.float64)
x = sm.add_constant(treatment)
groups = np.asarray(group_values)
cluster_fit = sm.OLS(y, x).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups, "use_correction": True},
    use_t=False,
)
cluster_ci = tuple(float(value) for value in cluster_fit.conf_int()[1])

snapshot = fixture["source"]["historical_experiment_snapshot"]
eligible_n = sum(row["impressions"] for row in records)
impressions_per_user = [row["impressions"] for row in records]
treatment_fallbacks = sum(
    row["fallbacks"] for row in records if row["variant"] == "treatment"
)

print(
    f"scope: historical README snapshot through {snapshot['cutoff']} inclusive "
    f"({eligible_n:,} impressions; {snapshot['later_impressions_excluded']:,} "
    "later rows excluded when fixture was refreshed)"
)
print(
    f"randomization unit: user_id ({len(records)} users; "
    f"{min(impressions_per_user)}-{max(impressions_per_user)} impressions/user; "
    "assignment stable)"
)
print()
print("PRIMARY — equal-weight mean per-user CTR (Welch)")
print(
    f"control    mean user CTR: {c_mean:.4f}  "
    f"(n={len(c_users)} users, SD={c_users.std(ddof=1):.4f})"
)
print(
    f"treatment  mean user CTR: {t_mean:.4f}  "
    f"(n={len(t_users)} users, SD={t_users.std(ddof=1):.4f})"
)
print(
    f"difference: {user_diff:+.4f}   95% CI [{user_ci[0]:+.4f}, {user_ci[1]:+.4f}]"
)
print(f"Welch t={welch_t:.3f}, df={welch_df:.2f}, p={welch_p:.3e}")
print()
print("DESCRIPTIVE — aggregate impression CTR")
print(
    f"control    CTR: {c_ctr:.4f}  "
    f"({by_arm['control']['clicks']}/{by_arm['control']['impressions']})"
)
print(
    f"treatment  CTR: {t_ctr:.4f}  "
    f"({by_arm['treatment']['clicks']}/{by_arm['treatment']['impressions']})"
)
print(f"raw difference: {t_ctr - c_ctr:+.4f}; relative lift: {t_ctr / c_ctr - 1:+.1%}")
print()
print("SECONDARY — impression-weighted OLS, SE clustered by user")
print(
    f"difference: {cluster_fit.params[1]:+.4f}   "
    f"95% CI [{cluster_ci[0]:+.4f}, {cluster_ci[1]:+.4f}]   "
    f"z={cluster_fit.tvalues[1]:.3f}, p={cluster_fit.pvalues[1]:.3e}"
)
print()
print(
    f"cold-start fallbacks inside treatment: {treatment_fallbacks}/"
    f"{by_arm['treatment']['impressions']} "
    f"({treatment_fallbacks / by_arm['treatment']['impressions']:.1%}) — intent-to-treat"
)

if args.json_out:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "statsmodels")
            },
        },
        "invocation": ["python", *sys.argv],
        "fixture": fixture_fingerprint(args.fixture),
        "fixture_refreshed_from_gcp_this_run": args.refresh_from_gcp,
        "repository": {
            "url": "https://github.com/karpit0499/reco-engine-adv",
            "commit_role": "repository/model state inspected before audit edits",
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "scope": snapshot,
        "design_checks": {
            "randomization_unit": "user_id",
            "users": len(records),
            "treatment_users": len(t_users),
            "control_users": len(c_users),
            "min_impressions_per_user": min(impressions_per_user),
            "max_impressions_per_user": max(impressions_per_user),
            "max_variants_per_user": 1,
        },
        "primary_user_level": {
            "method": "equal-weight per-user CTR; two-sided Welch t-test",
            "control": {
                "users": len(c_users),
                "mean_user_ctr": c_mean,
                "sample_sd": float(c_users.std(ddof=1)),
            },
            "treatment": {
                "users": len(t_users),
                "mean_user_ctr": t_mean,
                "sample_sd": float(t_users.std(ddof=1)),
            },
            "comparison": {
                "absolute_difference": user_diff,
                "standard_error": user_se,
                "difference_95_ci": list(user_ci),
                "welch_t": welch_t,
                "welch_df": welch_df,
                "p_two_sided": welch_p,
            },
        },
        "descriptive_impression_totals": {
            "control": by_arm["control"],
            "treatment": {
                **by_arm["treatment"],
                "cold_start_fallbacks": treatment_fallbacks,
            },
            "raw_absolute_difference": t_ctr - c_ctr,
            "raw_relative_lift": t_ctr / c_ctr - 1,
            "warning": "impressions repeat within user and are not independent",
        },
        "secondary_cluster_robust": {
            "method": (
                "OLS linear probability model on impressions reconstructed from per-user "
                "counts; intercept + treatment; cluster covariance by user_id with "
                "small-sample correction; asymptotic normal CI/p"
            ),
            "observations": eligible_n,
            "clusters": len(records),
            "absolute_difference": float(cluster_fit.params[1]),
            "cluster_robust_standard_error": float(cluster_fit.bse[1]),
            "difference_95_ci": list(cluster_ci),
            "z": float(cluster_fit.tvalues[1]),
            "p_two_sided": float(cluster_fit.pvalues[1]),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.json_out}")
