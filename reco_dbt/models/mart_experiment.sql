WITH clicks AS (
  SELECT DISTINCT impression_id FROM {{ source('reco', 'feedback') }} WHERE event = 'click'
),
convs AS (
  SELECT DISTINCT impression_id FROM {{ source('reco', 'feedback') }} WHERE event = 'conversion'
)
SELECT
  DATE(i.served_at) AS day,
  i.variant,
  COUNT(*) AS impressions,
  COUNTIF(i.served_by = 'popular') AS fallbacks,
  COUNTIF(c.impression_id IS NOT NULL) AS clicks,
  SAFE_DIVIDE(COUNTIF(c.impression_id IS NOT NULL), COUNT(*)) AS ctr,
  COUNTIF(v.impression_id IS NOT NULL) AS conversions,
  SAFE_DIVIDE(COUNTIF(v.impression_id IS NOT NULL), COUNT(*)) AS cvr
FROM {{ ref('stg_impressions') }} i
LEFT JOIN clicks c USING (impression_id)
LEFT JOIN convs  v USING (impression_id)
GROUP BY day, variant