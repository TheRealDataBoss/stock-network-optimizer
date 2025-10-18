CREATE TABLE IF NOT EXISTS `originami-sno-prod.sno.metrics_history`
(
  run_date DATE,
  universe STRING,
  metric STRING,
  value FLOAT64,
  window STRING,
  run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY run_date
CLUSTER BY universe, metric
