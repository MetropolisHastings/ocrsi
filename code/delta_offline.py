"""δ 修正认证的离线重算（用已有的 certified 结果，不需要重跑）。
规则：第 t 轮接受当且仅当  LCB_t − max(δ̂_t, 0) > 0，
δ̂_t = 此前各轮（部署年 ≤ t，已经实测）"确认年增益 − 部署年真实增益" 的平均，只用过去的数据。
近似：接受决策改变后，后续轮次的起点状态也会变，离线无法重现，这里用各轮候选的真实增益近似。
用法：python delta_offline.py runs/frame/online"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

NAMES = {"orch": "LLM-Orch", "gat": "GATs", "rf": "RandomForest", "dens": "DoubleEnsemble"}


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main(d):
    runs = defaultdict(list)
    for f in sorted(Path(d).glob("online_*_certified_s*.json")):
        j = json.loads(f.read_text())
        if len(j["rounds"]) >= 6:
            runs[j["method"]].append(j)
    # 全部方法、全部种子的 (决策年, 确认增益 − 部署真实增益)，用于跨方法的因果 δ̂
    gaps = defaultdict(list)
    for js in runs.values():
        for j in js:
            for r in j["rounds"]:
                if r.get("cert"):
                    gaps[r["decision_year"]].append(r["cert"]["gain_rsi"] - r["realized_gain_if_accepted"])

    def delta_hat(t, own):
        past = [g for y, v in gaps.items() if y < t for g in v]       # 部署年 y+1 ≤ t，决策时已知
        return max(float(np.mean(past)), 0.0) if past else 0.0

    print(f"{'方法':14s} {'原版累计':>16s} {'δ修正累计':>16s} {'原版 好/坏':>10s} {'δ修正 好/坏':>11s}")
    for m, js in sorted(runs.items()):
        base, corr, gb, gc = [], [], [0, 0], [0, 0]
        for j in js:
            b = c = 0.0
            for r in j["rounds"]:
                g = r["realized_gain_if_accepted"]
                if r["accepted"]:
                    b += g
                    gb[0 if g > 0 else 1] += 1
                cert = r.get("cert")
                if cert and cert["lcb_rsi"] - delta_hat(r["decision_year"], j) > 0 and r["kept"]:
                    c += g
                    gc[0 if g > 0 else 1] += 1
            base.append(b)
            corr.append(c)
        mb, hb = ci(base)
        mc, hc = ci(corr)
        print(f"{NAMES.get(m, m):14s} {mb:+.3f} ± {hb:.3f}   {mc:+.3f} ± {hc:.3f}   {gb[0]:>4d}/{gb[1]:<4d}   {gc[0]:>4d}/{gc[1]:<4d}")
    print("\n各决策年使用的 δ̂：", {t: round(delta_hat(t, None), 4) for t in sorted(gaps)})


if __name__ == "__main__":
    main(sys.argv[1])
