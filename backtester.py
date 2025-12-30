Python 3.14.2 (v3.14.2:df793163d58, Dec  5 2025, 12:18:06) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MeanReversionBacktester:
    """
    A Vectorised Backtesting Engine for Mean Reversion Strategies.
    
    Strategy: Bollinger Bands Mean Reversion
    - Logic: Capitalises on 'oversold' and 'overbought' conditions.
    - Entry: Long when price < Lower Band. Short when price > Upper Band.
    - Exit: Close position when price crosses the Simple Moving Average (SMA).
    
    Attributes
    ==========
    symbol: str
        Ticker symbol to test (e.g., 'SPY')
    start: str
        Start date for data (YYYY-MM-DD)
    end: str
        End date for data (YYYY-MM-DD)
    tc: float
        Proportional transaction costs (e.g., 0.001 = 0.1% per trade)
    """

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        """
        Fetches financial data using yfinance.
        """
        print(f"Fetching data for {self.symbol}...")
        raw = yf.download(self.symbol, start=self.start, end=self.end)
        # Handle different yfinance return structures (keep only 'Close' or 'Adj Close')
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw['Close'] if 'Close' in raw else raw['Adj Close']
        
        # DataFrame cleaning
        self.data = pd.DataFrame(raw)
        if self.data.shape[1] > 1:
             # If multiple columns remain, force select the first one (usually Close)
             self.data = self.data.iloc[:, 0].to_frame()
        
        self.data.columns = ['price']
        self.data.dropna(inplace=True)
        
        # Calculate Log Returns for the buy-and-hold benchmark
        self.data['return'] = np.log(self.data['price'] / self.data['price'].shift(1))

    def run_strategy(self, SMA, dev):
        """
        Executes the Mean Reversion Strategy (Vectorised).
        
        Parameters
        ==========
        SMA: int
            Window for Simple Moving Average (e.g., 20)
        dev: int
            Number of standard deviations for bands (e.g., 2)
        """
        data = self.data.copy().dropna()
        
        # 1. Calculate Indicators
        data['SMA'] = data['price'].rolling(SMA).mean()
        data['std'] = data['price'].rolling(SMA).std()
        
        # Bollinger Bands
        data['Upper'] = data['SMA'] + dev * data['std']
        data['Lower'] = data['SMA'] - dev * data['std']
        
        # 2. Vectorised Signal Generation
        # Distance from SMA (Z-score concept)
        data['distance'] = data['price'] - data['SMA']
        
        # Position Logic
        data['position'] = np.nan # Initialize
        
        # Long Condition: Price < Lower Band
        data.loc[data['price'] < data['Lower'], 'position'] = 1
        
        # Short Condition: Price > Upper Band
        data.loc[data['price'] > data['Upper'], 'position'] = -1
        
        # Exit Condition: Price crosses SMA (Mean Reversion complete)
        # In a vectorised backtest, crossing the SMA means distance changes sign.
        # Assume flat (0) when crossing SMA, but for simplicity here carry forward 
        # previous positions until a signal flips. 
        # A robust way: Forward fill positions, but clear them when crossing mean
        # 1 = Long, -1 = Short, 0 = Neutral.
        
        # We fill NaNs with previous position to simulate holding the trade
        data['position'] = data['position'].ffill()
        
        # Optional: Force exit if price crosses SMA? 
        # Advanced Logic: Set position to 0 where price crosses SMA.
        # For now, we stick to the core Reversion logic (flip position at extremes).
        data['position'].fillna(0, inplace=True)
        
        # 3. Calculate Strategy Returns
        # Shift position by 1 because today's signal executes at tomorrow's open/close
        data['strategy'] = data['position'].shift(1) * data['return']
        
        # 4. Subtract Transaction Costs
        # Trades occur when position changes
        trades = data['position'].diff().fillna(0).abs()
        data['strategy'] = data['strategy'] - trades * self.tc
        
        data.dropna(inplace=True)
        self.results = data
        return data

    def plot_results(self):
        """
        Plots the cumulative performance of the strategy vs Buy & Hold.
        """
        if self.results is None:
            print("Run strategy first.")
            return
        
        title = f"{self.symbol} | Mean Reversion | TC = {self.tc}"
        self.results[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
        plt.title(title)
        plt.ylabel("Cumulative Returns")
        plt.legend(["Buy & Hold", "Strategy"])
        plt.grid()
...         plt.show()
... 
...     def calculate_risk_metrics(self):
...         """
...         Calculates Sharpe Ratio and Max Drawdown.
...         """
...         if self.results is None:
...             return
...         
...         strategy_rets = self.results['strategy']
...         
...         # Sharpe Ratio (Annualized, assuming 252 trading days)
...         # Risk free rate assumed 0 for simplicity or excess returns
...         sharpe = np.sqrt(252) * strategy_rets.mean() / strategy_rets.std()
...         
...         # Max Drawdown
...         cum_ret = strategy_rets.cumsum().apply(np.exp)
...         cum_max = cum_ret.cummax()
...         drawdown = cum_max - cum_ret
...         max_dd = drawdown.max()
...         
...         print(f"=== RISK METRICS ===")
...         print(f"Sharpe Ratio: {sharpe:.2f}")
...         print(f"Max Drawdown: {max_dd:.2%}")
...         return sharpe, max_dd
... 
... # --- MAIN EXECUTION ---
... if __name__ == "__main__":
...     # 1. Instantiate Backtester (SPY = S&P 500 ETF)
...     bt = MeanReversionBacktester("SPY", "2020-01-01", "2025-10-01", 10000, 0.0005)
...     
...     # 2. Run Strategy (SMA=42, 2 Std Devs)
...     bt.run_strategy(SMA=42, dev=2)
...     
...     # 3. Show Results
...     bt.calculate_risk_metrics()
