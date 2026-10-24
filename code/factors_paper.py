"""论文附录里给出的因子代码（Listing 1~5），用来验证评估流程。

约定（与论文一致）：函数输入是单只股票按日期排序的 DataFrame，
列为 open/high/low/close/volume，返回同索引的一列因子值。
"""
import numpy as np
import pandas as pd


def factor_liquidity_impact(df):
    """Listing 1：(high - close) / volume，单位成交量带来的价格上行阻力。"""
    return (df["high"] - df["close"]) / (df["volume"] + 1e-9)


def factor_dayhigh_impact_per_vol(df):
    """Listing 2（变异版）：(high - low) / volume。"""
    return (df["high"] - df["low"]) / (df["volume"] + 1e-9)


def factor_price_impact_per_vol_tanh_1d(df):
    """Listing 3（进化版）：tanh(|close - open| / (volume * close))。"""
    eps = 1e-9
    abs_move = (df["close"] - df["open"]).abs()
    dollar_vol = df["volume"] * df["close"]
    return np.tanh(abs_move / (dollar_vol + eps))


def factor_lownorm_slopecos_30d_low(df):
    """Listing 4：低价/30日EMA × 量价斜率 × EMA12/48 余弦。"""
    eps = 1e-9
    low_ema30 = df["low"].ewm(span=30, adjust=False).mean()
    norm_low = df["low"] / (low_ema30 + eps)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    log_vol = np.log(df["volume"] / df["volume"].shift(1))
    log_vol = log_vol.replace([np.inf, -np.inf], np.nan)
    cov = log_ret.ewm(halflife=20, adjust=False).cov(log_vol)
    var = log_vol.ewm(halflife=20, adjust=False).var() + eps
    slope_raw = cov / var
    slope = slope_raw / (1.0 + slope_raw.abs())
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema48 = df["close"].ewm(span=48, adjust=False).mean()
    cos_phase = ema12 / np.sqrt(ema12 ** 2 + ema48 ** 2 + eps)
    return norm_low * slope * cos_phase


def factor_pressure_drawdown_fisher_10d(df):
    """Listing 5：压力中位数 × Fisher 变换的量价相关 × 回撤。"""
    eps = 1e-12
    pressure_raw = (df["close"] - df["open"]).abs() * np.log1p(df["volume"])
    pressure_med10 = pressure_raw.rolling(10, min_periods=10).median()
    roll_max = df["close"].rolling(252, min_periods=1).max()
    drawdown = -((df["close"] - roll_max) / (roll_max + eps))
    decayed_dd = drawdown.ewm(span=8, adjust=False).mean()
    log_ret = np.log(df["close"] + eps).diff()
    log_vol = np.log(df["volume"] + eps).diff()
    corr20 = log_ret.rolling(20, min_periods=20).corr(log_vol)
    fisher = np.arctanh(corr20.clip(-0.999, 0.999))
    return pressure_med10 * fisher * np.tanh(decayed_dd)


PAPER_FACTORS = {
    "L1_流动性冲击": factor_liquidity_impact,
    "L2_变异版": factor_dayhigh_impact_per_vol,
    "L3_进化版": factor_price_impact_per_vol_tanh_1d,
    "L4_低价斜率余弦": factor_lownorm_slopecos_30d_low,
    "L5_压力回撤Fisher": factor_pressure_drawdown_fisher_10d,
}
