import pandas as pd
from strats.base import Strategy, BacktestResult
from strats.utils import get_df, ema

class Strat_EMA_Trend_D1(Strategy):
    name = "EMA_Trend_D1"
    timeframe = "D1"

    def fetch_data(self, symbol:str)->pd.DataFrame:
        return get_df(symbol, self.timeframe)

    def run(self, df:pd.DataFrame)->BacktestResult:
        c = df["close"]
        ema50 = ema(c, 50); ema200 = ema(c, 200)
        sig = (ema50 > ema200).astype(int)*2 - 1
        return self._toy_backtest(df, sig)