"""汇总统一框架实验：各方法 × 入池规则，多随机种子的 RSI 均值和 95% 置信区间。
用法：python frame_report.py runs/frame"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NAMES = {"classic": "传统因子库", "ml": "机器学习（GBDT）", "gp": "遗传规划"}
RULES = {"naive": "只看验证分", "certified": "可认证（确认 1 年）", "certified2y": "可认证（确认 2 年）"}


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main(d):
    rows = []
    for f in sorted(Path(d).glob("*_s*.json")):
        j = json.loads(f.read_text())
        r0 = j["rounds"][0]
        for r in j["rounds"]:
            rows.append({"method": j["method"], "rule": j["admit"], "seed": j["seed"], "round": r["round"],
                         "val_gain": r["sel"] - r0["sel"],                       # 智能体 / 研究员看得到的提升
                         "test_ric_gain": r["off"]["test_ric"] - r0["off"]["test_ric"],
                         "rsi": r["rsi"], "rsi3": r["rsi3"], "lcb": r["lcb_rsi3"],
                         "accepted": r.get("accepted", False), "minutes": r["minutes"]})
    df = pd.DataFrame(rows)
    df.to_csv(Path(d) / "frame_all.csv", index=False, encoding="utf-8-sig")
    last = df[df["round"] == df.groupby(["method", "rule", "seed"])["round"].transform("max")]
    out = []
    for (m, rule), g in last.groupby(["method", "rule"]):
        rsi, rsi_h = ci(g["rsi"])
        r3, r3_h = ci(g["rsi3"])
        vg, vg_h = ci(g["val_gain"])
        tg, tg_h = ci(g["test_ric_gain"])
        allr = df[(df.method == m) & (df.rule == rule) & (df["round"] > 0)]
        # 逐轮不降的比例：每个种子、每一轮，RSI3 是否 ≥ 上一轮
        mono = []
        for _, s in df[(df.method == m) & (df.rule == rule)].groupby("seed"):
            v = s.sort_values("round")["rsi3"].values
            mono.append(bool(np.all(np.diff(v) >= -1e-12)))
        out.append({"方法": NAMES.get(m, m), "入池规则": RULES.get(rule, rule), "种子数": len(g),
                    "验证分提升": f"{vg:+.4f} ± {vg_h:.4f}", "测试期RankIC变化": f"{tg:+.4f} ± {tg_h:.4f}",
                    "RSI(4项)": f"{rsi:+.4f} ± {rsi_h:.4f}", "RSI3": f"{r3:+.4f} ± {r3_h:.4f}",
                    "RSI3下界(均值)": f"{g['lcb'].mean():+.4f}",
                    "下界≤实际 的比例": f"{np.mean(g['lcb'] <= g['rsi3'] + 1e-12):.0%}",
                    "逐轮不降的比例": f"{np.mean(mono):.0%}",
                    "通过的轮数/总轮数": f"{int(allr['accepted'].sum())}/{len(allr)}",
                    "每轮分钟": f"{allr['minutes'].mean():.1f}"})
    tab = pd.DataFrame(out)
    tab.to_csv(Path(d) / "frame_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ms = [m for m in NAMES if m in set(df.method)]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"naive": "#c0392b", "certified": "#2e6fba", "certified2y": "#1b9e77"}
    for ax, m in zip(axes, ms):
        for rule in RULES:
            g = df[(df.method == m) & (df.rule == rule)]
            if g.empty:
                continue
            st = g.groupby("round")["rsi3"].agg(list)
            mu = np.array([ci(v)[0] for v in st])
            h = np.nan_to_num(np.array([ci(v)[1] for v in st]))
            ax.plot(st.index, mu, "-o", ms=3, color=colors[rule], label=RULES[rule])
            ax.fill_between(st.index, mu - h, mu + h, color=colors[rule], alpha=0.15)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(NAMES[m])
        ax.set_xlabel("轮次")
        ax.set_xticks(sorted(set(df["round"])))
    axes[0].set_ylabel("测试期 RSI（3 项）")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(Path(d) / "frame_rsi.png", dpi=200)
    print("图：", Path(d) / "frame_rsi.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/frame")
