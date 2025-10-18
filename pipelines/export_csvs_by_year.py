import os
import io
import math
import zipfile
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

START = "2010-01-01"
TODAY = date.today().isoformat()

def get_sp500():
    try: return yf.tickers_sp500()
    except Exception: return []

def get_dow30():
    try: return yf.tickers_dow()
    except Exception: return []

def get_nasdaq100():
    # yfinance has no direct helper; read from Wikipedia
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            col = [c for c in t.columns if "icker" in str(c).lower() or "symbol" in str(c).lower()][0]
            syms = t[col].astype(str).str.replace(".", "-", regex=False).str.strip().unique().tolist()
            return sorted(syms)
    return []

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def download_ohlcv(symbols):
    """Return DataFrame: date, symbol, open, high, low, close, volume"""
    if not symbols: return pd.DataFrame()
    frames = []
    # Pull in small batches to avoid timeouts/huge payloads
    for chunk in batched(symbols, 50):
        data = yf.download(
            tickers=" ".join(chunk), start=START, end=TODAY,
            auto_adjust=True, group_by="ticker", threads=True, progress=False
        )
        # yfinance returns wide panel; normalize per symbol
        for s in chunk:
            try:
                df = data[s].copy()
                if df is None or df.empty: continue
                df.reset_index(inplace=True)
                df.rename(columns={
                    "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
                }, inplace=True)
                df["symbol"] = s
                df = df[["date","symbol","open","high","low","close","volume"]]
                frames.append(df)
            except Exception:
                continue
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def add_returns(df):
    if df.empty: return df
    df = df.sort_values(["symbol","date"]).copy()
    df["arith_ret"] = df.groupby("symbol")["close"].pct_change()
    df["log_ret"]   = np.log(df["close"]/df.groupby("symbol")["close"].shift(1))
    return df

def write_yearly_csvs(df, universe, z: zipfile.ZipFile):
    if df.empty: return 0
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    cols = ["date","symbol","open","high","low","close","volume","arith_ret","log_ret"]
    years = sorted(df["year"].dropna().unique())
    written = 0
    for yr in years:
        part = df[df["year"]==yr][cols].sort_values(["symbol","date"])
        if part.empty: continue
        # path inside zip: csv/<universe>/<year>.csv
        arcname = f"csv/{universe}/{yr}.csv"
        with io.StringIO() as buf:
            part.to_csv(buf, index=False)
            z.writestr(arcname, buf.getvalue())
            written += len(part)
    return written

def main():
    universes = [
        ("SP500", get_sp500()),
        ("DOW30", get_dow30()),
        ("NASDAQ100", get_nasdaq100()),
    ]
    os.makedirs("out", exist_ok=True)
    zip_path = os.path.join("out", "ohlcv_by_year_2010_to_today.zip")
    total_rows = 0
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for uni, syms in universes:
            print(f"[{uni}] symbols={len(syms)}")
            if not syms:
                print(f"[{uni}] no symbols found, skipping.")
                continue
            px = download_ohlcv(syms)
            if px.empty:
                print(f"[{uni}] no price data, skipping.")
                continue
            px = add_returns(px)
            written = write_yearly_csvs(px, uni, z)
            total_rows += written
            print(f"[{uni}] wrote {written:,} rows into yearly CSVs.")
    print(f"ALL DONE. Total rows written: {total_rows:,}")
    print(f"ZIP artifact: {zip_path}")

if __name__ == "__main__":
    main()
