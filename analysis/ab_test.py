import sys

from google.cloud import bigquery
from statsmodels.stats.proportion import (
    proportions_ztest, proportion_confint, confint_proportions_2indep)

PROJECT = "reco-engine-adv"
bq = bigquery.Client(project=PROJECT)

q = f"""
WITH clicks AS (
  SELECT DISTINCT impression_id
  FROM `{PROJECT}.reco.feedback` WHERE event = 'click'
)
SELECT
  i.variant,
  COUNT(*) AS n,
  COUNTIF(c.impression_id IS NOT NULL) AS clicks,
  COUNTIF(i.served_by = 'popular') AS fallbacks
FROM `{PROJECT}.reco.impressions` i
LEFT JOIN clicks c USING (impression_id)
GROUP BY variant
"""
d = {r["variant"]: dict(r) for r in bq.query(q).result()}

missing = [v for v in ("control", "treatment") if v not in d]
if missing:
    sys.exit(f"no impressions for {missing} — run the Phase 12 simulator longer")

t, c = d["treatment"], d["control"]
tc, tn = t["clicks"], t["n"]
cc, cn = c["clicks"], c["n"]
if min(tn, cn) < 100:
    print(f"WARNING: only {min(tn, cn)} impressions in the smaller arm — "
          f"any result here is noise. Run more rounds.\n")

t_ctr, c_ctr = tc / tn, cc / cn
stat, pval = proportions_ztest([tc, cc], [tn, cn])
t_lo, t_hi = proportion_confint(tc, tn, method="wilson")
c_lo, c_hi = proportion_confint(cc, cn, method="wilson")
d_lo, d_hi = confint_proportions_2indep(tc, tn, cc, cn, compare="diff")

print(f"control    CTR: {c_ctr:.4f}  ({cc}/{cn})   95% CI [{c_lo:.4f}, {c_hi:.4f}]")
print(f"treatment  CTR: {t_ctr:.4f}  ({tc}/{tn})   95% CI [{t_lo:.4f}, {t_hi:.4f}]")
print()
print(f"absolute difference: {t_ctr - c_ctr:+.4f}   95% CI [{d_lo:+.4f}, {d_hi:+.4f}]")
print(f"relative lift:       {t_ctr / c_ctr - 1:+.1%}")
print(f"z = {stat:.3f}   p = {pval:.5f}")
print("SIGNIFICANT at alpha=0.05" if pval < 0.05 else
      "not yet significant — run the simulator for more samples")
print()
print(f"cold-start fallbacks inside treatment: {t['fallbacks']}/{tn} "
      f"({t['fallbacks'] / tn:.1%}) — kept IN the numbers above (intent-to-treat)")