# stock-network-optimizer
Dynamic, index-agnostic stock network optimizer for SP500/DOW30/NASDAQ100/custom universes.
- End-to-end pipeline (data → returns → risk → network → optimization).
- Colab-friendly; daily T-1 refresh; cached artifacts.

## Architecture

- **Universe**: current or timeline constituents for SP500, DOW30, NASDAQ100, or custom.
- **Market data**: adjusted closes with caching and min-history filters.
- **Returns & risk**: log returns; Ledoit–Wolf shrinkage covariance; optional PC1 de-toning.
- **Visualization**: normalized equal-weight benchmarks; overlap-aligned comparisons.
- **Prediction overlay**: optional per-constituent predicted log returns.
- **Optimization (next)**: HRP and risk-budgeted variants with caps and diagnostics.

