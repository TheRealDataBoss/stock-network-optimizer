import os
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_gbq import to_gbq

PROJECT = "originami-sno-prod"
DATASET = "sno"
START  = "2010-01-01"

def gbq_table(t): return f"{PROJECT}.{DATASET}.{t}"

def write_gbq(df, table):
    if df is None or df.empty: return
    to_gbq(df, gbq_table(table), project_id=PROJECT, if_exists="append")

def get_sp500():
    try:
        return yf.tickers_sp500()
    except Exception:
        return []

def get_dow30():
    try:
        return yf.tickers_dow()
    except Exception:
        return []

def get_nasdaq100():
    # yfinance doesn’t ship a NASDAQ100 helper; use Wikipedia
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            col = [c for c in t.columns if "icker" in str(c).lower() or "symbol" in str(c).lower()][0]
            return sorted(t[col].astype(str).str.replace(".", "-", regex=False).unique().tolist())
    return []

def download_ohlcv(symbols):
    if not symbols: return pd.DataFrame()
    data = yf.download(tickers=" ".join(symbols), start=START, auto_adjust=True, group_by="ticker", threads=True)
    frames = []
    for s in symbols:
        try:
            df = data[s].copy()
            df.reset_index(inplace=True)   # Date, Open, High, Low, Close, Volume
            df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
            df["symbol"] = s
            frames.append(df[["date","symbol","open","high","low","close","volume"]])
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def add_returns(df):
    df = df.sort_values(["symbol","date"]).copy()
    df["arith_ret"] = df.groupby("symbol")["close"].pct_change()
    df["log_ret"]   = np.log(df["close"]/df.groupby("symbol")["close"].shift(1))
    return df

def write_truth(df, universe):
    if df.empty: return
    df = df.copy()
    df["universe"] = universe
    # only the ORIGINAL agreed metrics; extra columns in table will be NULL (fine)
    df = df[["date","universe","symbol","close","arith_ret","log_ret"]]
    write_gbq(df, "truth")

def write_membership(df, universe):
    if df.empty: return
    mem = df[["date","symbol"]].dropna().drop_duplicates().copy()
    mem["universe"] = universe
    mem = mem[["date","universe","symbol"]]
    write_gbq(mem, "universe_membership")

def run():
    print("Fetching universe symbols…")
    spx = get_sp500()
    dji = get_dow30()
    ndx = get_nasdaq100()

    universes = [("SP500", spx), ("DOW30", dji), ("NASDAQ100", ndx)]

    for uni, syms in universes:
        if not syms: 
            print(f"[{uni}] no symbols found, skipping")
            continue
        print(f"[{uni}] symbols: {len(syms)} — downloading OHLCV since {START}")
        px = download_ohlcv(syms)
        if px.empty:
            print(f"[{uni}] no price data, skipping")
            continue
        px = add_returns(px)
        print(f"[{uni}] writing truth rows: {len(px):,}")
        write_truth(px, uni)
        print(f"[{uni}] writing membership rows…")
        write_membership(px, uni)

if __name__ == "__main__":
    run()
