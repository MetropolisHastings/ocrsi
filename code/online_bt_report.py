"""汇总 --bt 回测：把 6 个部署年（2021–2026）的日超额收益首尾相接，得到每个方法、每条规则、每个种子的
“实际部署路径”年化超额与信息比率，并与“从不改动的起点状态”比较。
用法：python online_bt_report.py runs/frame/online_bt"""
import glob
import json
import sys

import numpy as np
from scipy import stats


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def path(j, key):
    ex = []
    for r in j["rounds"]:
        if key in r:
            ex += r[key]["daily_excess"]
    ex = np.asarray(ex, float)
    return ex.mean() * 252, (ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0.0)


d = sys.argv[1]
runs = {}
for f in sorted(glob.glob(f"{d}/online_*_s*.json")):
    j = json.load(open(f))
    if len(j["rounds"]) < 6 or "bt_deployed" not in j["rounds"][0]:
        continue
    runs.setdefault((j["method"], j["rule"]), []).append(j)

print(f"{'方法':14s}{'规则':11s}{'种子':>4s}  {'部署路径年化超额':>16s}  {'IR':>12s}  {'相对起点的超额差':>16s}")
static_done = set()
for (m, r), js in sorted(runs.items()):
    dep = [path(j, "bt_deployed") for j in js]
    sta = [path(j, "bt_static") for j in js]
    ex_m, ex_h = ci([x[0] for x in dep])
    ir_m, ir_h = ci([x[1] for x in dep])
    df_m, df_h = ci([a[0] - b[0] for a, b in zip(dep, sta)])
    print(f"{m:14s}{r:11s}{len(js):>4d}  {ex_m:+7.2%} ± {ex_h:5.2%}  {ir_m:+.2f} ± {ir_h:.2f}  {df_m:+7.2%} ± {df_h:5.2%}")
    if m not in static_done:
        static_done.add(m)
        s_m, s_h = ci([x[0] for x in sta])
        print(f"{m:14s}{'(起点)':11s}{len(js):>4d}  {s_m:+7.2%} ± {s_h:5.2%}")
