"""因子评估：IC、RankIC、ICIR、RankICIR、MI（对应论文附录 B.3）。"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def daily_ic(factor: pd.Series, label: pd.Series, method: str = "pearson") -> pd.Series:
    """每天在截面上算因子与未来收益的相关系数。"""
    df = pd.concat([factor.rename("f"), label.rename("r")], axis=1).dropna()
    df = df[np.isfinite(df["f"])]
    g = df.groupby(level="datetime")
    if method == "spearman":
        df = df.assign(f=g["f"].rank(), r=g["r"].rank())
        g = df.groupby(level="datetime")
    ic = g.apply(lambda x: x["f"].corr(x["r"]) if len(x) >= 20 else np.nan)
    return ic.dropna()


def mutual_info(factor: pd.Series, label: pd.Series, n_sample: int = 100_000,
                seed: int = 0) -> float:
    """互信息：先在每天截面上做排名标准化（去掉量纲和时间漂移），
    再抽样用 k 近邻估计器计算。"""
    df = pd.concat([factor.rename("f"), label.rename("r")], axis=1).dropna()
    df = df[np.isfinite(df["f"])]
    if len(df) < 1000:
        return np.nan
    g = df.groupby(level="datetime")
    df = df.assign(f=g["f"].rank(pct=True), r=g["r"].rank(pct=True))
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed)
    return float(mutual_info_regression(df[["f"]].values, df["r"].values,
                                        n_neighbors=3, random_state=seed)[0])


def evaluate(factor: pd.Series, label: pd.Series, with_mi: bool = True) -> dict:
    ic = daily_ic(factor, label, "pearson")
    ric = daily_ic(factor, label, "spearman")
    out = {
        "IC": ic.mean(),
        "RankIC": ric.mean(),
        "ICIR": ic.mean() / ic.std(),
        "RankICIR": ric.mean() / ric.std(),
        "NaN率": float(factor.isna().mean()),
    }
    if with_mi:
        out["MI"] = mutual_info(factor, label)
    return out
