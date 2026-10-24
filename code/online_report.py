"""汇总在线 RSI 实验：各方法 × 规则，多种子的累计真实增益、接受轮数、接受决策的正确率。
用法：python online_report.py <online 结果目录>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


NAMES = {"orch": "LLM-Orch", "reflect": "Reflexion", "rdagent": "RD-Agent(Q)", "gat": "GATs", "rf": "RandomForest", "dens": "DoubleEnsemble"}


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main(d):
    runs = {}
    for f in sorted(Path(d).glob("online_*_s*.json")):
        j = json.loads(f.read_text())
        if len(j["rounds"]) < 6:
            continue
        runs.setdefault((j["method"], j["rule"]), []).append(j)
    rows = []
    for (m, rule), js in sorted(runs.items()):
        tot = [sum(r["realized_gain"] for r in j["rounds"]) for j in js]
        acc = [sum(r["accepted"] for r in j["rounds"]) for j in js]
        good = [sum(r["accepted"] and r["realized_gain"] > 0 for r in j["rounds"]) for j in js]
        bad = [sum(r["accepted"] and r["realized_gain"] < 0 for r in j["rounds"]) for j in js]
        cand = [np.mean([r["realized_gain_if_accepted"] for r in j["rounds"] if r["kept"]]) for j in js]
        mu, h = ci(tot)
        rows.append({"方法": NAMES.get(m, m), "规则": rule, "种子数": len(js), "6轮累计真实增益": f"{mu:+.3f} ± {h:.3f}",
                     "平均接受轮数": f"{np.mean(acc):.1f}/6", "接受后变好": int(np.sum(good)),
                     "接受后变差": int(np.sum(bad)),
                     "接受决策正确率": f"{np.sum(good) / max(1, np.sum(good) + np.sum(bad)):.0%}",
                     "初筛候选的平均真实增益": f"{np.mean(cand):+.3f}"})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))
    tab.to_csv(Path(d) / "online_summary.csv", index=False, encoding="utf-8-sig")
    # 逐年：确认增益（认证规则）与部署年真实增益的关系 → 估计 δ
    pts = []
    for (m, rule), js in runs.items():
        if rule != "certified":
            continue
        for j in js:
            for r in j["rounds"]:
                if r["cert"]:
                    pts.append({"method": m, "seed": j["seed"], "year": r["decision_year"],
                                "conf_gain": r["cert"]["gain_rsi"], "deploy_gain": r["realized_gain_if_accepted"]})
    if pts:
        p = pd.DataFrame(pts)
        p["delta"] = p["conf_gain"] - p["deploy_gain"]
        print("\n确认年增益 vs 部署年真实增益（δ = 两者之差）：")
        print(p.groupby("year")[["conf_gain", "deploy_gain", "delta"]].mean().round(3).to_string())
        print("相关系数：", round(p["conf_gain"].corr(p["deploy_gain"]), 3))
        p.to_csv(Path(d) / "online_delta.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
