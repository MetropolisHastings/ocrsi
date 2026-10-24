"""对照组：8 个教科书式的经典量价因子（人工设计，不经过任何挖掘）。
AI 挖出来的因子要明显好过这一组，才能说明“挖掘”本身有价值。"""
import numpy as np

EPS = 1e-12


def _ret(df):
    return df["close"].pct_change()


BASELINE = {
    "rev_5": lambda df: -df["close"].pct_change(5),                                   # 短期反转
    "rev_20": lambda df: -df["close"].pct_change(20),                                 # 月度反转
    "mom_120_20": lambda df: df["close"].shift(20) / df["close"].shift(120) - 1,      # 中期动量（跳过最近一月）
    "vol_20": lambda df: _ret(df).rolling(20, min_periods=15).std(),                  # 波动率
    "dollar_vol_20": lambda df: np.log(
        (df["close"] * df["volume"]).rolling(20, min_periods=15).mean() + 1),         # 成交额（规模/流动性代理）
    "amihud_20": lambda df: (_ret(df).abs() / (df["close"] * df["volume"] + EPS)
                             ).rolling(20, min_periods=15).mean(),                    # 非流动性
    "max_ret_20": lambda df: _ret(df).rolling(20, min_periods=15).max(),              # 彩票效应
    "pv_corr_20": lambda df: _ret(df).rolling(20, min_periods=15).corr(
        np.log(df["volume"] + EPS).diff()),                                           # 量价相关
}
