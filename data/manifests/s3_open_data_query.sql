-- Reproducible iNaturalist Open Data metadata scan (CC0 / CC-BY only)
SELECT
    o.observation_uuid AS observation_id,
    o.observer_id,
    p.photo_id,
    p.photo_uuid,
    LOWER(p.license) AS photo_license,
    t.name AS scientific_name,
    t.taxon_id,
    o.observed_on AS date_observed,
    o.latitude,
    o.longitude
FROM read_csv_auto('inaturalist-open-data/metadata/observations.csv.gz') AS o
JOIN read_csv_auto('inaturalist-open-data/metadata/photos.csv.gz') AS p
  ON o.observation_uuid = p.observation_uuid
JOIN read_csv_auto('inaturalist-open-data/metadata/taxa.csv.gz') AS t
  ON o.taxon_id = t.taxon_id
WHERE o.quality_grade = 'research'
  AND LOWER(p.license) IN ('cc0', 'cc-by')
  AND t.taxon_id IN (42158, 42051, 41976, 42076, 41663, 41880, 179919, 46017, 46020, 43114, 906, 1409, 8229, 509665, 8010, 18195, 6317, 12727, 9235, 9373)
  AND o.latitude BETWEEN 32.5 AND 42.0
  AND o.longitude BETWEEN -124.5 AND -114.1;
