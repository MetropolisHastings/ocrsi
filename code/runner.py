"""把单股票因子函数应用到整个面板，并做基本的安全检查。"""
import numpy as np
import pandas as pd

BASE_COLS = ["open", "high", "low", "close", "volume"]


def apply_factor(func, panel: pd.DataFrame) -> pd.Series:
    """按股票逐只计算（避免跨股票串数据），只保留当天是成分股的样本。"""
    out = []
    for inst, g in panel.groupby(level="instrument", sort=False):
        df = g.droplevel("instrument")[BASE_COLS].astype("float64")
        s = pd.Series(np.asarray(func(df.copy()), dtype="float64"), index=g.index)
        out.append(s)
    f = pd.concat(out).replace([np.inf, -np.inf], np.nan)
    return f[panel["in_index"]]


def lookahead_test(func, panel: pd.DataFrame, n_inst: int = 5, cut: int = 300) -> bool:
    """未来函数检测：把数据截断到第 cut 天，因子在截断点之前的值应当不变。"""
    insts = panel.index.get_level_values("instrument").unique()[:n_inst]
    for inst in insts:
        df = panel.loc[inst, BASE_COLS].astype("float64")
        if len(df) <= cut + 10:
            continue
        full = pd.Series(np.asarray(func(df.copy()), dtype="float64"), index=df.index)
        part = pd.Series(np.asarray(func(df.iloc[:cut].copy()), dtype="float64"),
                         index=df.index[:cut])
        a, b = full.iloc[:cut], part
        same = np.isclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-10)
        if not same.all():
            return False
    return True
