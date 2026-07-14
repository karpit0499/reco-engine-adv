SELECT impression_id, user_id, variant, served_by, served_at
FROM {{ source('reco', 'impressions') }}