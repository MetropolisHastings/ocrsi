"""读取 Qlib 二进制数据，构造沪深300面板数据和 10 日收益标签。

数据来源：github.com/chenditc/investment_data 的 qlib_bin.tar.gz
不依赖 qlib 包，直接读 .bin 文件。
"""
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]


def load_calendar(root: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(root / "calendars/day.txt", header=None)[0])


def load_membership(root: Path, index: str = "csi300") -> pd.DataFrame:
    df = pd.read_csv(root / f"instruments/{index}.txt", sep="\t", header=None,
                     names=["inst", "start", "end"], parse_dates=["start", "end"])
    return df


def read_bin(path: Path, cal: pd.DatetimeIndex) -> pd.Series:
    arr = np.fromfile(path, dtype="<f4")
    if arr.size < 2:
        return pd.Series(dtype="float32")
    start = int(arr[0])
    vals = arr[1:]
    return pd.Series(vals, index=cal[start:start + len(vals)])


def load_panel(root: str, index: str = "csi300", start: str = "2010-01-01",
               end: str = "2024-12-31", cache: str | None = None) -> pd.DataFrame:
    """返回 MultiIndex(instrument, datetime) 的 DataFrame，
    列为 open/high/low/close/volume 以及 in_index（当天是否为指数成分股）。"""
    if cache and Path(cache).exists():
        return pd.read_pickle(cache)
    root = Path(root)
    cal = load_calendar(root)
    mem = load_membership(root, index)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    mem = mem[(mem.end >= s) & (mem.start <= e)]
    frames = []
    for inst, grp in mem.groupby("inst"):
        d = root / "features" / inst.lower()
        if not d.exists():
            continue
        cols = {f: read_bin(d / f"{f}.day.bin", cal) for f in FIELDS}
        df = pd.DataFrame(cols).loc[s:e]
        if df.empty:
            continue
        mask = pd.Series(False, index=df.index)
        for _, r in grp.iterrows():
            mask |= (df.index >= r.start) & (df.index <= r.end)
        df["in_index"] = mask.values
        df["instrument"] = inst
        frames.append(df)
    panel = pd.concat(frames)
    panel.index.name = "datetime"
    panel = panel.reset_index().set_index(["instrument", "datetime"]).sort_index()
    if cache:
        panel.to_pickle(cache)
    return panel


def make_label(panel: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """开盘价交易的未来 horizon 日收益：T+1 开盘买入，T+1+horizon 开盘卖出。
    即 Qlib 写法 Ref($open, -(h+1)) / Ref($open, -1) - 1。"""
    o = panel["open"].groupby(level="instrument")
    return (o.shift(-(horizon + 1)) / o.shift(-1) - 1).rename("label")
