CREATE TABLE IF NOT EXISTS `originami-sno-prod.sno.truth`
(
  date DATE,
  universe STRING,
  symbol STRING,
  close FLOAT64,
  arith_ret FLOAT64,
  log_ret FLOAT64
)
PARTITION BY date
CLUSTER BY universe, symbol
