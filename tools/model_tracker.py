import os, glob, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR   = REPO_ROOT / "artifacts"
DOCS_DIR  = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TODAY  = dt.date.today()
RUN_TS = pd.Timestamp.utcnow()

# Optional BigQuery
USE_BQ = all(k in os.environ for k in ["GOOGLE_APPLICATION_CREDENTIALS","GCP_PROJECT_ID","BQ_DATASET"])
if USE_BQ:
    from google.cloud import bigquery
    BQ_CLIENT = bigquery.Client(project=os.environ["GCP_PROJECT_ID"])
    BQ_DATASET = os.environ["BQ_DATASET"]
else:
    BQ_CLIENT = None

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    if df.index.name in candidates:      # allow index to match
        df.reset_index(inplace=True)
        for c in candidates:
            if c in df.columns: return c
    raise KeyError(f"Need one of {candidates} in {list(df.columns)}")

def load_predictions():
    files = glob.glob(str(ART_DIR / "predictions" / "**" / "*.csv"), recursive=True)
    if not files:
        return pd.DataFrame(columns=["date","universe","symbol","pred_log_ret","model_name","version","run_timestamp"])
    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        dcol = pick_col(df, ["date","Date","as_of","timestamp"])
        scol = pick_col(df, ["symbol","ticker","SYM","Symbol","Ticker"])
        df["date"]   = pd.to_datetime(df[dcol]).dt.date
        df["symbol"] = df[scol].astype(str).str.upper()

        # infer universe from path: artifacts/predictions/<UNIVERSE>/file.csv
        parts = Path(fp).parts
        universe = "UNKNOWN"
        if "predictions" in parts:
            i = parts.index("predictions")
            if i+1 < len(parts):
                universe = parts[i+1]
        df["universe"] = universe

        # prediction column
        if "pred_log_ret" not in df.columns:
            for c in ["pred","pred_ret","predicted","y_hat","y_pred","prediction"]:
                if c in df.columns:
                    df["pred_log_ret"] = pd.to_numeric(df[c], errors="coerce")
                    break
        if "pred_log_ret" not in df.columns:
            # skip this file if still missing
            continue

        if "model_name" not in df.columns: df["model_name"] = "baseline"
        if "version"    not in df.columns: df["version"]    = "v1"
        if "run_timestamp" not in df.columns: df["run_timestamp"] = RUN_TS

        frames.append(df[["date","universe","symbol","pred_log_ret","model_name","version","run_timestamp"]])

    if not frames:
        return pd.DataFrame(columns=["date","universe","symbol","pred_log_ret","model_name","version","run_timestamp"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date","symbol","pred_log_ret"])
    return out

def compute_min_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    # Simple, robust metric: count of predictions per universe
    if preds.empty:
        return pd.DataFrame(columns=["run_date","universe","metric","value","window","run_timestamp"])
    agg = (preds.groupby("universe", as_index=False)["pred_log_ret"]
                .size()
                .rename(columns={"size":"value"}))
    agg["metric"] = "pred_count"
    agg["window"] = "all"
    agg["run_date"] = TODAY
    agg["run_timestamp"] = RUN_TS
    return agg[["run_date","universe","metric","value","window","run_timestamp"]].reset_index(drop=True)

def write_metrics_bq(df: pd.DataFrame):
    if not USE_BQ or df.empty:
        return
    table_id = f'{os.environ["GCP_PROJECT_ID"]}.{os.environ["BQ_DATASET"]}.metrics_history'
    job = BQ_CLIENT.load_table_from_dataframe(df, table_id)
    job.result()

def write_html(preds: pd.DataFrame, metrics: pd.DataFrame):
    # Very small HTML so CI has a visible artifact
    p = DOCS_DIR / "model_skill_over_time.html"
    rows = int(metrics["value"].sum()) if not metrics.empty else 0
    p.write_text(f"""
<!doctype html>
<meta charset='utf-8'/>
<title>Model Tracker</title>
<h2>Model Tracker — {TODAY}</h2>
<p>Total predictions ingested: <b>{rows}</b></p>
<pre>Universes:
{metrics.to_string(index=False) if not metrics.empty else "(no data)"}
</pre>
""", encoding="utf-8")

def main():
    preds = load_predictions()
    metrics = compute_min_metrics(preds)
    write_metrics_bq(metrics)
    write_html(preds, metrics)

if __name__ == "__main__":
    main()