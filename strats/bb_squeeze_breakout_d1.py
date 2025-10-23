# strats/bb_squeeze_breakout_d1.py
import pandas as pd
import numpy as np
from strats.base import Strategy, BacktestResult
from strats.utils import get_df, bbands

# --- PATCH: safe _clamp fallback ---
try: _clamp
except NameError:
    def _clamp(x, lo, hi):
        try:
            if not np.isfinite(x): return lo
            x = float(x)
            if x < lo: return lo
            if x > hi: return hi
            return x
        except (ValueError, TypeError): return lo
# --- END PATCH ---

class Strat_BB_Squeeze_BO_D1(Strategy):
    name = "BB_Squeeze_BO_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]
        lo, ma, up = bbands(c, 20, 2.0)
        bbw = (up - lo) / ma
        thresh = bbw.rolling(120).quantile(0.2)
        long_sig = ((bbw < thresh) & (c > up)).astype(int)
        short_sig = ((bbw < thresh) & (c < lo)).astype(int) * -1
        sig = long_sig + short_sig
        
        # --- FIX 3: Rimosso FutureWarning ---
        sig.ffill(inplace=True)
        
        return self._toy_backtest(df, sig)