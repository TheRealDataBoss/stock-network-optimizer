import datetime as dt, pandas as pd, numpy as np
from google.cloud import bigquery
import yfinance as yf

PROJECT   = "originami-sno-prod"
DATASET   = "sno"
TABLE     = f"{PROJECT}.{DATASET}.truth"
UNIVERSE  = "SP500"
SINCE     = dt.date(2025,10,1)
TODAY     = dt.date.today()

print(f"Downloading {UNIVERSE} from {SINCE} → {TODAY}")

tickers = yf.tickers_sp500()
tickers = sorted(set([t for t in tickers if t and t.isalpha()]))

df = yf.download(
    tickers=tickers,
    start=SINCE.isoformat(),
    end=(TODAY + dt.timedelta(days=1)).isoformat(),
    group_by="ticker",
    auto_adjust=False,
    progress=False,
    threads=True,
    interval="1d",
)

rows = []
if isinstance(df.columns, pd.MultiIndex):
    for sym in tickers:
        if sym in df.columns.get_level_values(0):
            try:
                s = df[(sym, "Close")].dropna()
            except KeyError:
                continue
            for d, c in s.items():
                rows.append((d.date(), UNIVERSE, sym, float(c)))
else:
    s = df["Close"].dropna()
    for d, c in s.items():
        rows.append((d.date(), UNIVERSE, tickers[0], float(c)))

prices = pd.DataFrame(rows, columns=["date","universe","symbol","close"]).sort_values(["symbol","date"])
prices["prev_close"] = prices.groupby("symbol")["close"].shift(1)
prices = prices.dropna(subset=["prev_close"])
prices["arith_ret"] = prices["close"] / prices["prev_close"] - 1.0
prices["log_ret"]   = np.log(prices["close"] / prices["prev_close"])

print(f"Uploading {len(prices)} rows to {TABLE}")

client = bigquery.Client(project=PROJECT)
job = client.load_table_from_dataframe(
    prices[["date","universe","symbol","log_ret","arith_ret","close"]],
    TABLE,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
)
job.result()
print(f"✅ Appended {job.output_rows} rows into {TABLE}")
