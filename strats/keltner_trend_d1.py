import pandas as pd
from strats.base import Strategy, BacktestResult
from strats.utils import get_df, ema

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

class Strat_Keltner_Trend_D1(Strategy):
    name = "Keltner_Trend_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]; h = df["high"]; l = df["low"]
        e20 = ema(c, 20)
        tr = (h-l).ewm(alpha=1/20, adjust=False).mean()
        up = e20 + 2*tr; lo = e20 - 2*tr
        long_sig = (c > up).astype(int)
        short_sig = (c < lo).astype(int) * -1
        sig = long_sig + short_sig
        sig.ffill(inplace=True)
        return self._toy_backtest(df, sig)