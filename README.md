# OFI-based Short-Term Futures Strategy (TX / MTX)

## Overview
This project studies whether Order Flow Imbalance (OFI) features
can predict short-term price movements in Taiwan index futures.

## Research Pipeline
1. Settlement calendar construction
2. Near-month contract selection
3. Feature engineering (OFI, lag, rolling)
4. XGBoost model training (in-sample)
5. Strategy backtesting (out-of-sample)

## Data
- Tick-level transaction data (2017–2023)
- Day & night session included
- Near-month contract rolled by settlement calendar

## Model
- XGBoost multiclass classifier
- Labels: Up / Down / Flat (1-minute horizon)

## Strategy
- Signal: P(up) − P(down)
- Top-K selection
- Fixed holding period
- Symmetric long/short
- Transaction cost considered

## Results
- In-sample performance: positive Sharpe
- Out-of-sample performance: degraded under realistic costs
- Strong sensitivity to transaction cost

## Key Takeaways
- OFI contains short-term predictive signal
- ML improves ranking but not robustness
- Transaction cost is the dominant constraint

## Repository Structure
- research_raw/: original research scripts
- src/: cleaned pipeline for presentation
- results/: backtest outputs
- docs/: presentation slides
