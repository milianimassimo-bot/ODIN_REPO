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

class Strat_VolTarget_Trend_D1(Strategy):
    name = "VolTarget_Trend_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]
        e50 = ema(c, 50); e200 = ema(c, 200)
        dir = (e50 > e200).astype(int)*2 -1
        ret = c.pct_change().fillna(0.0).rolling(20).std().replace(0, 1e-9)
        w = (ret.median() / ret).clip(0.2, 3.0)
        sig = (dir * (w / w.median())).clip(-1,1)
        sig = sig.apply(lambda x: 1 if x>0.15 else (-1 if x<-0.15 else 0))
        sig.ffill(inplace=True)
        return self._toy_backtest(df, sig)