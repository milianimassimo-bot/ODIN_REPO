# strats/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

@dataclass
class BacktestResult:
    name: str
    pf: float
    dd: float
    sharpe: float
    trades: int
    meta: Dict[str, Any]

class Strategy(ABC):
    name: str
    timeframe: str  # "D1" o "H4"

    @abstractmethod
    def fetch_data(self, symbol:str)->pd.DataFrame:
        ...

    @abstractmethod
    def run(self, df: pd.DataFrame)->BacktestResult:
        ...

    def _toy_backtest(self, df: pd.DataFrame, sig: pd.Series)->BacktestResult:
        import numpy as np
        ret = df["close"].pct_change().fillna(0.0)
        strat_ret = ret * sig.shift(1).fillna(0.0)

        eq = (1.0 + strat_ret).cumprod()
        dd = float(1.0 - (eq / eq.cummax()).min()) if not eq.empty else 0.0
        
        annual_factor = 252 if self.timeframe == 'D1' else 252 * 6
        mu = strat_ret.mean() * annual_factor
        sigma = strat_ret.std() * (annual_factor**0.5)
        sharpe = float(mu / sigma) if sigma > 1e-12 else 0.0

        gains = strat_ret[strat_ret > 0].sum()
        losses = -strat_ret[strat_ret < 0].sum()
        pf = float(gains / losses) if losses > 1e-9 else 0.0

        trades = int((sig.diff().abs() > 0).sum() // 2)

        return BacktestResult(
            name=self.name, pf=round(pf,3), dd=round(dd,3),
            sharpe=round(sharpe,3), trades=trades,
            meta={}
        )