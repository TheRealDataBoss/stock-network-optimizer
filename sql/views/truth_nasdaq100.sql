CREATE OR REPLACE VIEW `originami-sno-prod.sno.truth_nasdaq100` AS
SELECT t.*
FROM `originami-sno-prod.sno.truth` t
JOIN `originami-sno-prod.sno.universe_membership` m
USING (date, universe, symbol)
WHERE m.universe = 'NASDAQ100'
