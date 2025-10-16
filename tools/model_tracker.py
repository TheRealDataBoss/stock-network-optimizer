import os, glob, datetime as dt
import numpy as np, pandas as pd, plotly.express as px
from google.cloud import bigquery
import yfinance as yf

PROJECT  = os.environ["GCP_PROJECT_ID"]
DATASET  = os.environ["BQ_DATASET"]
ART_BASE = "artifacts"
DOCS_DIR = "docs"

PRED_DIRS = [f"{ART_BASE}/predictions/SP500",
             f"{ART_BASE}/predictions/DOW30",
             f"{ART_BASE}/predictions/NASDAQ100"]

METRICS_PQ = f"{ART_BASE}/metrics/model_performance_history.parquet"

os.makedirs(os.path.dirname(METRICS_PQ), exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

bq = bigquery.Client(project=PROJECT)

def load_predictions():
    frames=[]
    for d in PRED_DIRS:
        uni = d.split("/")[-1]
        for f in glob.glob(f"{d}/*.csv"):
            df = pd.read_csv(f)
            df["universe"]=uni
            frames.append(df)
    if not frames: return pd.DataFrame()
    df=pd.concat(frames,ignore_index=True)
    df["date"]=pd.to_datetime(df["date"]).dt.date
    df["symbol"]=df["symbol"].str.replace(".","-").str.upper()
    keep=["date","universe","symbol","pred_log_ret","model_name","version","run_timestamp"]
    return df[keep]

def truth_for(symbols, start):
    tick=" ".join(symbols)
    px=yf.download(tickers=tick, start=start, auto_adjust=True, threads=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex): px=px["Close"]
    px=px.sort_index()
    lr=np.log(px).diff().dropna()
    lr.index=lr.index.date
    return lr

def load_to_bq(df, table):
    if df.empty: return
    bq.load_table_from_dataframe(df, f"{PROJECT}.{DATASET}.{table}").result()

def append_parquet(df, path):
    if df.empty: return
    if os.path.exists(path):
        old=pd.read_parquet(path)
        df=pd.concat([old,df],ignore_index=True).drop_duplicates()
    df.to_parquet(path,index=False)

def compute_metrics(preds, truth):
    m = preds.merge(truth.stack().rename("log_ret").reset_index().rename(columns={"level_1":"symbol"}),
                    on=["date","symbol"], how="inner")
    if m.empty: return pd.DataFrame()
    m["hit"] = np.sign(m["pred_log_ret"]) == np.sign(m["log_ret"])
    out=[]
    today=dt.date.today()
    for (u,mn,v), g in m.groupby(["universe","model_name","version"]):
        g=g.sort_values("date")
        rmse=np.sqrt(((g["pred_log_ret"]-g["log_ret"])**2).mean())
        mape=(np.abs((g["log_ret"]-g["pred_log_ret"])/(g["log_ret"].replace(0,np.nan))).dropna()).mean()
        corr=g[["pred_log_ret","log_ret"]].corr().iloc[0,1]
        da=g["hit"].mean()
        s_lr=g.groupby("date")["log_ret"].mean()
        mu, sig = s_lr.mean()*252, s_lr.std(ddof=1)*np.sqrt(252)
        rsh = mu/sig if sig>0 else np.nan
        p_lr=g.groupby("date")["pred_log_ret"].mean()
        pm, ps = p_lr.mean()*252, p_lr.std(ddof=1)*np.sqrt(252)
        psh = pm/ps if ps>0 else np.nan
        out.append(dict(run_date=today,universe=u,model_name=mn or "model",version=v or "v1",
                        window_days=252,rmse=rmse,mape=mape,corr_pred_actual=corr,
                        directional_accuracy=da,realized_sharpe=rsh,predicted_sharpe=psh))
    return pd.DataFrame(out)

def main():
    preds=load_predictions()
    if preds.empty:
        print("No predictions found."); return
    syms=preds.groupby("universe")["symbol"].unique().to_dict()
    start=preds["date"].min()
    truth_list=[]
    for u, arr in syms.items():
        tr=truth_for(arr.tolist(),start=start)
        tr=tr.stack().rename("log_ret").reset_index()
        tr["date"]=pd.to_datetime(tr["Date"]).dt.date
        tr=tr.rename(columns={"level_1":"symbol"})[["date","symbol","log_ret"]]
        tr["universe"]=u
        truth_list.append(tr[["date","universe","symbol","log_ret"]])
    truth=pd.concat(truth_list,ignore_index=True)

    load_to_bq(preds, "predictions")
    load_to_bq(truth, "truth")

    t_piv=truth.pivot_table(index="date",columns="symbol",values="log_ret")
    metrics=compute_metrics(preds,t_piv)
    load_to_bq(metrics,"metrics_history")
    append_parquet(metrics, METRICS_PQ)

    fig=px.line(metrics.sort_values("run_date"), x="run_date",
                y=["rmse","corr_pred_actual","directional_accuracy"],
                color="universe", title="Model Skill — RMSE / Corr / DA")
    fig.write_html(f"{DOCS_DIR}/model_skill_over_time.html", include_plotlyjs="cdn")

if __name__=="__main__":
    main()
