import os, glob, io, json, datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path

# Optional BigQuery
USE_BQ = all(k in os.environ for k in ["GOOGLE_APPLICATION_CREDENTIALS","GCP_PROJECT_ID","BQ_DATASET"])
if USE_BQ:
    from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR   = REPO_ROOT / "artifacts"
DOCS_DIR  = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date.today()
RUN_TS = pd.Timestamp.utcnow()

def _col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    # try index
    if df.index.name in candidates: return df.index.name
    # try after reset_index
    tmp = df.reset_index()
    for c in candidates:
        if c in tmp.columns: return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)} (index: {df.index.name})")

def load_predictions():
    files = glob.glob(str(ART_DIR / "predictions" / "**" / "*.csv"), recursive=True)
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # normalize columns
        dcol = _col(df, ["date","Date"])
        scol = _col(df, ["symbol","Symbol","ticker","Ticker"])
        df["date"]   = pd.to_datetime(df[dcol]).dt.date
        df["symbol"] = df[scol].str.upper()
        # universe from path (e.g., artifacts/predictions/SP500/xxxx.csv)
        parts = Path(f).parts
        uni = "UNKNOWN"
        if "predictions" in parts:
            i = parts.index("predictions")
            if i+1 < len(parts): uni = parts[i+1]
        df["universe"] = uni
        # prediction value
        if "pred_log_ret" not in df.columns:
            # fallback: any of these?
            for c in ["pred","pred_ret","predicted","y_hat","y_pred"]:
                if c in df.columns:
                    df["pred_log_ret"] = df[c].astype(float)
                    break
        if "pred_log_ret" not in df.columns:
            raise ValueError(f"{f} missing pred_log_ret")
        # meta
        if "model_name" not in df.columns: df["model_name"] = "baseline"
        if "version"    not in df.columns: df["version"]    = "v1"
        if "run_timestamp" not in df.columns: df["run_timestamp"] = RUN_TS
        dfs.append(df[["date","universe","symbol","pred_log_ret","model_name","version","run_timestamp"]])
    out = pd.concat(dfs, ignore_index=True).dropna(subset=["date","symbol","pred_log_ret"])
    return out

def compute_truth(pred):
    # build minimal daily truth from Yahoo closes and next-day returns for the prediction dates window
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not available in runner") from e

    syms = sorted(pred["symbol"].unique().tolist())
    if not syms: return pd.DataFrame()

    lo = pred["date"].min() - pd.Timedelta(days=5)
    hi = pred["date"].max() + pd.Timedelta(days=5)
    px = yf.download(syms, start=str(lo), end=str(hi), auto_adjust=True, progress=False)

    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"].copy()
    else:
        # single symbol case
        pass
    px = px.sort_index()
    px.index = pd.to_datetime(px.index).date

    # wide to long
    df = px.reset_index().melt(id_vars=["index"], var_name="symbol", value_name="close")
    df = df.rename(columns={"index":"date"})
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.sort_values(["symbol","date"])
    # log and arithmetic returns
    df["log_ret"] = df.groupby("symbol")["close"].apply(lambda s: np.log(s).diff())
    df["arith_ret"] = df.groupby("symbol")["close"].pct_change()
    # keep only dates we have predictions on (same-day realized; downstream can align to T+1 if desired)
    uni_map = pred[["symbol","universe"]].drop_duplicates()
    out = df.merge(uni_map, on="symbol", how="left")
    return out[["date","universe","symbol","log_ret","arith_ret","close"]].dropna(subset=["close"])

def metric_hit_rate(pred, truth, horizon="same_day"):
    # align on date & symbol; sign agreement as simple skill proxy
    m = pred.merge(truth, on=["date","symbol","universe"], how="inner")
    if m.empty:
        return pd.DataFrame([{
            "run_date": TODAY, "universe":"NA", "metric":"hit_rate", "value": np.nan,
            "window": horizon, "run_timestamp": RUN_TS
        }])
    yhat = np.sign(m["pred_log_ret"].astype(float))
    y    = np.sign(m["log_ret"].astype(float).fillna(0))
    hit  = (yhat == y).mean()
    return pd.DataFrame([{
        "run_date": TODAY, "universe":"ALL", "metric":"hit_rate", "value": float(hit),
        "window": horizon, "run_timestamp": RUN_TS
    }])

def write_bq(df, table):
    client = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
    table_id = f'{os.environ["GCP_PROJECT_ID"]}:{os.environ["BQ_DATASET"]}.{table}'
    job = client.load_table_from_dataframe(df, table_id)
    job.result()

def write_html(metrics_df):
    # Minimal HTML report
    html = io.StringIO()
    html.write("<html><head><meta charset='utf-8'><title>Model Skill Over Time</title></head><body>")
    html.write(f"<h2>Run: {TODAY}</h2>")
    html.write(metrics_df.to_html(index=False))
    html.write("</body></html>")
    (DOCS_DIR / "model_skill_over_time.html").write_text(html.getvalue(), encoding="utf-8")

def main():
    pred = load_predictions()
    if pred.empty:
        raise SystemExit("No predictions found under artifacts/predictions/**.csv")

    truth = compute_truth(pred)
    # metrics (extend later)
    metrics = metric_hit_rate(pred, truth, horizon="same_day")

    # Write artifacts
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "predictions_parsed.parquet").write_bytes(pred.to_parquet(index=False))
    (ART_DIR / "truth_parsed.parquet").write_bytes(truth.to_parquet(index=False))
    write_html(metrics)

    # Write BigQuery if configured
    if USE_BQ:
        write_bq(pred,    "predictions")
        write_bq(truth,   "truth")
        write_bq(metrics, "metrics_history")

if __name__ == "__main__":
    main()