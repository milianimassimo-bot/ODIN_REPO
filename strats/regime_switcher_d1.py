import os, json
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

REGIME_JSON = os.getenv("ODIN_REGIME_JSON", os.path.join(os.path.dirname(__file__), "..", "odin_regime_report_d1.json"))

def _get_regime_for(asset:str)->str:
    try:
        with open(REGIME_JSON,"r",encoding="utf-8") as f:
            payload = json.load(f)
        for it in payload.get("items", []):
            if it.get("asset")==asset:
                return it.get("regime","INDEFINITO")
    except Exception:
        pass
    return "INDEFINITO"

class Strat_Regime_Switcher_D1(Strategy):
    name = "Regime_Switcher_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]
        regime = _get_regime_for(self._asset_label) if hasattr(self, "_asset_label") else "INDEFINITO"
        e50 = ema(c,50); e200=ema(c,200)
        if "TREND" in regime:
            sig = (e50>e200).astype(int)*2-1
        elif "LATERALE" in regime:
            sig = ((c < c.rolling(20).mean()-2*c.rolling(20).std()).astype(int) -
                   (c > c.rolling(20).mean()+2*c.rolling(20).std()).astype(int))
            sig.ffill(inplace=True)
        else:
            sig = pd.Series(0, index=df.index)
        return self._toy_backtest(df, sig)