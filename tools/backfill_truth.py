import os, glob
from pathlib import Path
import pandas as pd
import numpy as np

USE_BQ = all(k in os.environ for k in ["GOOGLE_APPLICATION_CREDENTIALS","GCP_PROJECT_ID","BQ_DATASET"])
if USE_BQ:
    from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR    = REPO_ROOT / "artifacts"

def _col(df, names):
    for n in names:
        if n in df.columns: return n
    raise KeyError(f"Missing any of {names}. Have: {list(df.columns)}")

def load_pred_index():
    files = glob.glob(str(ART_DIR / "predictions" / "**" / "*.csv"), recursive=True)
    if not files:
        return pd.DataFrame(columns=["date","symbol","universe"])
    rows = []
    for f in files:
        df = pd.read_csv(f)
        dcol = _col(df, ["date","Date"])
        scol = _col(df, ["symbol","Symbol","ticker","Ticker"])
        df["date"]   = pd.to_datetime(df[dcol]).dt.date
        df["symbol"] = df[scol].str.upper()
        # universe from path …/predictions/<UNIVERSE>/file.csv
        parts = Path(f).parts
        uni = "UNKNOWN"
        if "predictions" in parts:
            i = parts.index("predictions")
            if i+1 < len(parts):
                uni = parts[i+1]
        rows.append(df[["date"]].assign(symbol=df["symbol"], universe=uni))
    idx = pd.concat(rows, ignore_index=True).drop_duplicates()
    return idx

def fetch_truth(idx: pd.DataFrame) -> pd.DataFrame:
    if idx.empty:
        return pd.DataFrame(columns=["date","universe","symbol","log_ret","arith_ret","close"])
    import yfinance as yf
    # date range with one extra day before to compute returns
    start = (pd.to_datetime(idx["date"]).min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(idx["date"]).max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    syms  = sorted(idx["symbol"].unique().tolist())
    px = yf.download(syms, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.sort_index()
    # long form
    lf = px.reset_index().melt(id_vars=["Date"], var_name="symbol", value_name="close")
    lf["date"] = pd.to_datetime(lf["Date"]).dt.date
    lf = lf.sort_values(["symbol","date"])
    # returns per symbol
    lf["prev_close"] = lf.groupby("symbol")["close"].shift(1)
    lf["arith_ret"]  = (lf["close"] / lf["prev_close"]) - 1.0
    lf["log_ret"]    = np.log(lf["close"] / lf["prev_close"])
    lf = lf.dropna(subset=["arith_ret","log_ret"])
    # keep only dates & symbols we actually predicted, and carry universe from predictions
    uni_map = idx.drop_duplicates(["symbol","universe"]).set_index("symbol")["universe"]
    out = lf.merge(idx[["date","symbol"]].drop_duplicates(), on=["date","symbol"], how="inner")
    out["universe"] = out["symbol"].map(uni_map).fillna("UNKNOWN")
    return out[["date","universe","symbol","log_ret","arith_ret","close"]]

def upsert_bq(df):
    if df.empty:
        print("No truth rows to write.")
        return
    client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
    table_id = f'{os.environ["GCP_PROJECT_ID"]}.{os.environ["BQ_DATASET"]}.truth'
    job = client.load_table_from_dataframe(
        df.astype({
            "date":"datetime64[ns]",
            "universe":"string",
            "symbol":"string",
            "log_ret":"float64",
            "arith_ret":"float64",
            "close":"float64",
        }),
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    )
    job.result()
    print(f"Wrote {len(df)} rows to {table_id}")

def main():
    idx = load_pred_index()
    truth = fetch_truth(idx)
    if USE_BQ:
        upsert_bq(truth)
    # always keep a local artifact for inspection
    outdir = ART_DIR / "truth"
    outdir.mkdir(parents=True, exist_ok=True)
    if not truth.empty:
        truth.to_csv(outdir / "latest_truth.csv", index=False)
        print(f"Saved artifacts/truth/latest_truth.csv with {len(truth)} rows.")
    else:
        print("No truth to save.")

if __name__ == "__main__":
    main()