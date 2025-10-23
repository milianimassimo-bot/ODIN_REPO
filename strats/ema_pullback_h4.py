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

class Strat_EMA_Pullback_H4(Strategy):
    name = "EMA_Pullback_H4"
    timeframe = "H4"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]
        e20 = ema(c, 20); e50 = ema(c, 50)
        trend = (e20 > e50).astype(int)*2 - 1
        sig = ( (trend==1) & (c<e20) ).astype(int) - ( (trend==-1) & (c>e20) ).astype(int)
        sig.ffill(inplace=True)
        return self._toy_backtest(df, sig)