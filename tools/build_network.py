import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=1)
    ap.add_argument("--since", default="2010-01-01")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--min_symbols", type=int, default=200)
    ap.add_argument("--coverage", type=float, default=0.90)
    args = ap.parse_args()

    print("=== RUN PLAN (dry) ===")
    print("Project/Dataset    : originami-sno-prod.sno")
    print(f"Data since         : {args.since}")
    print(f"Lookback/step      : {args.lookback} / {args.step} days")
    print("Gaps allowed       : ffill<=5 days; drop_long_gaps=True")
    print(f"Coverage threshold : {int(args.coverage*100)}% per window; min symbols={args.min_symbols}")
    print("Portfolios         : central=25, peripheral=25, weighting=equal")
    print("Tables (assumed)   : sno.truth, sno.universe_membership, sno.metrics_history")
    print("Views (optional)   : sno.truth_sp500, sno.truth_dow30, sno.truth_nasdaq100")

if __name__ == "__main__":
    main()
