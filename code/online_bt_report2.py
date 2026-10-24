"""回测汇总（完整版）：年化收益、年化超额、夏普比率、信息比率、最大回撤、超额最大回撤、Calmar。
每日组合收益 = 每日超额 + 沪深 300 当日收益（与 backtest.py 同口径：开盘到开盘），不需要重跑。
把 2021–2026 六个部署年首尾相接，得到每个方法、每条规则、每个种子的“实际部署路径”，
并与“从不改动的起点状态”比较。输出终端表格，同时写 bt_summary.csv。
用法：python online_bt_report2.py runs/frame/online_bt ../qlib_bin"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

d, data = sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "../qlib_bin")
cal = pd.DatetimeIndex(pd.read_csv(data / "calendars/day.txt", header=None)[0])
arr = np.fromfile(data / "features/sh000300/open.day.bin", dtype="<f4")
b = pd.Series(arr[1:], index=cal[int(arr[0]):int(arr[0]) + len(arr) - 1]).astype(float)
bench = (b.shift(-2) / b.shift(-1) - 1)


def path(j, key):
    dates, ex = [], []
    for r in j["rounds"]:
        if key in r:
            dates += r[key]["dates"]; ex += r[key]["daily_excess"]
    s = pd.Series(ex, index=pd.DatetimeIndex(dates)).sort_index()
    s = s[~s.index.duplicated()]
    return s, s + bench.reindex(s.index).fillna(0.0)


def metrics(ex, ret):
    nav = (1 + ret).cumprod()
    cum_ex = (1 + ex).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    ann = float(ret.mean() * 252)
    return {"年化收益": ann, "年化超额": float(ex.mean() * 252),
            "夏普": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0,
            "IR": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
            "最大回撤": mdd, "超额最大回撤": float((cum_ex / cum_ex.cummax() - 1).min()),
            "Calmar": ann / abs(mdd) if mdd < 0 else float("nan")}


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


runs = {}
for f in sorted(glob.glob(f"{d}/online_*_s*.json")):
    j = json.load(open(f))
    if len(j["rounds"]) < 6 or "bt_deployed" not in j["rounds"][0]:
        continue
    runs.setdefault((j["method"], j["rule"]), []).append(j)

rows = []
for (m, r), js in sorted(runs.items()):
    dep = [metrics(*path(j, "bt_deployed")) for j in js]
    rows.append({"method": m, "rule": r, "seeds": len(js), **{k: ci([x[k] for x in dep]) for k in dep[0]}})
    if not any(x["method"] == m and x["rule"] == "start" for x in rows):
        sta = [metrics(*path(j, "bt_static")) for j in js]
        rows.append({"method": m, "rule": "start", "seeds": len(js), **{k: ci([x[k] for x in sta]) for k in sta[0]}})

order = {"start": 0, "naive": 1, "certified": 2}
rows.sort(key=lambda x: (x["method"], order.get(x["rule"], 9)))
keys = ["年化收益", "年化超额", "夏普", "IR", "最大回撤", "超额最大回撤", "Calmar"]
pct = {"年化收益", "年化超额", "最大回撤", "超额最大回撤"}
print(f"{'方法':10s}{'规则':10s}{'种子':>3s}  " + "  ".join(f"{k:>14s}" for k in keys))
out = []
for x in rows:
    cells = []
    for k in keys:
        mu, h = x[k]
        cells.append(f"{mu:+7.2%}±{h:5.2%}" if k in pct else f"{mu:+6.2f}±{h:4.2f}  ")
    print(f"{x['method']:10s}{x['rule']:10s}{x['seeds']:>3d}  " + "  ".join(f"{c:>14s}" for c in cells))
    out.append({"method": x["method"], "rule": x["rule"], "seeds": x["seeds"],
                **{f"{k}": x[k][0] for k in keys}, **{f"{k}_ci": x[k][1] for k in keys}})
pd.DataFrame(out).to_csv(Path(d) / "bt_summary.csv", index=False, encoding="utf-8-sig")
print("已写出", Path(d) / "bt_summary.csv")
