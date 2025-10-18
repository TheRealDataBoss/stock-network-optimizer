CREATE TABLE IF NOT EXISTS `originami-sno-prod.sno.universe_membership`
(
  date DATE,
  universe STRING,
  symbol STRING
)
PARTITION BY date
CLUSTER BY universe, symbol
