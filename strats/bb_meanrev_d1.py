import pandas as pd
from strats.base import Strategy, BacktestResult
from strats.utils import get_df, bbands, rsi

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

class Strat_BB_MR_D1(Strategy):
    name = "BB_MR_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]
        lo, ma, up = bbands(c, 20, 2.0)
        r = rsi(c, 14)
        long_sig = ((c.shift(1) < lo.shift(1)) & (c > lo) & (r <= 35)).astype(int)
        short_sig = ((c.shift(1) > up.shift(1)) & (c < up) & (r >= 65)).astype(int) * -1
        sig = long_sig + short_sig
        sig.ffill(inplace=True)
        return self._toy_backtest(df, sig)