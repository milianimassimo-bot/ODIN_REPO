import pandas as pd
from strats.base import Strategy, BacktestResult
from strats.utils import get_df

# --- PATCH: safe _clamp fallback ---
try:
    _clamp  # noqa
except NameError:
    def _clamp(x, lo, hi):
        try:
            x = float(x)
        except Exception:
            return lo
        if x < lo: return lo
        if x > hi: return hi
        return x
# --- END PATCH ---

class Strat_Donchian_BO_D1(Strategy):
    name = "Donchian_BO_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        h = df["high"].rolling(20).max()
        l = df["low"].rolling(20).min()
        c = df["close"]
        long_sig = (c > h.shift(1)).astype(int)
        short_sig = (c < l.shift(1)).astype(int) * -1
        sig = long_sig + short_sig
        sig.ffill(inplace=True)
        return self._toy_backtest(df, sig)