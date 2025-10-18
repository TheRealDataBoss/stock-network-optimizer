CREATE TABLE IF NOT EXISTS `originami-sno-prod.sno.predictions`
(
  date DATE,
  universe STRING,
  symbol STRING,
  pred_log_ret FLOAT64,
  model_name STRING,
  version STRING,
  run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY universe, symbol
