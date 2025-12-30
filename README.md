# Quantitative Mean Reversion Backtester

## Overview
This project implements a vectorised backtesting engine in Python to analyse **Mean Reversion** strategies on S&P 500 data (using `SPY` as a proxy). The system utilises **Bollinger Bands** to identify statistical anomalies (overbought/oversold conditions) and executes trades to capture reversion to the mean.

## Key Features
* **Vectorised Execution:** Uses `pandas` and `numpy` for high-speed calculation of strategy logic across large time-series datasets, avoiding inefficient loops.
* **Transaction Costs:** Incorporates realistic transaction costs (slippage/commissions) to produce net P&L estimates.
* **Risk Analytics:** Calculates industry-standard metrics:
    * **Sharpe Ratio:** Measures risk-adjusted return.
    * **Max Drawdown:** Measures downside risk.

## Strategy Logic
The strategy assumes that prices generally revert to a moving average over time.
1.  **Indicators:** Calculates a Simple Moving Average (SMA) and Standard Deviation ($\sigma$).
2.  **Bollinger Bands:**
    * Upper Band = $SMA + 2\sigma$
    * Lower Band = $SMA - 2\sigma$
3.  **Signals:**
    * **Long:** Price < Lower Band (Oversold).
    * **Short:** Price > Upper Band (Overbought).

## Tech Stack
* **Python 3.10+**
* **Pandas & NumPy:** For data manipulation and vectorisation.
* **yfinance:** For fetching historical market data.
* **Matplotlib:** For performance visualisation.

## References
* Hilpisch, Yves. *Python for Finance*. O'Reilly.
* Hull, John C. *Options, Futures, and Other Derivatives*. (Concept of Mean Reversion).

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt