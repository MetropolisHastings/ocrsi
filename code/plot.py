"""因子可视化：为每个通过质检的因子画一张图，左边累计 IC 曲线，右边五分组超额收益。

用法：python plot.py --agent AgentLiquidity --data ../qlib_bin
图片保存在 runs/<agent>/plots/ 下。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_panel, make_label
from metrics import daily_ic
from runner import apply_factor
from checker import compile_factor

TRAIN_END, TEST_START = pd.Timestamp("2019-12-31"), pd.Timestamp("2021-01-01")
BLUE, ORANGE = "#2a78d6", "#eb6834"          # 分类色第 1、2 位
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def quintile_excess(f: pd.Series, lab: pd.Series) -> pd.Series:
    """每天按因子值分 5 组，算各组相对当天平均的 10 日超额收益（%），再对所有天取平均。"""
    df = pd.concat([f.rename("f"), lab.rename("r")], axis=1).dropna()
    g = df.groupby(level="datetime")
    df["q"] = (g["f"].rank(pct=True) * 5).clip(upper=4.999).astype(int) + 1
    df["ex"] = df["r"] - g["r"].transform("mean")
    return df.groupby(["datetime", "q"])["ex"].mean().groupby(level="q").mean() * 100


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_one(name, f, lab, out: Path):
    ic = daily_ic(f, lab, "spearman")
    dt = f.index.get_level_values("datetime")
    tr, te = (dt <= TRAIN_END), (dt >= TEST_START)
    q_tr, q_te = quintile_excess(f[tr], lab[tr]), quintile_excess(f[te], lab[te])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(SURFACE)
    style(a1); style(a2)

    # 左：累计 RankIC
    cum = ic.cumsum()
    a1.axvspan(TEST_START, cum.index.max(), color="#f0efec", zorder=0)
    a1.plot(cum.index, cum.values, color=BLUE, linewidth=2)
    a1.axhline(0, color=INK2, linewidth=0.8)
    a1.text(TEST_START, a1.get_ylim()[1], "  test period", va="top", color=INK2, fontsize=9)
    a1.set_title(f"Cumulative daily RankIC  (mean {ic.mean():+.4f})", loc="left", color=INK, fontsize=11)
    a1.set_ylabel("cumulative RankIC", color=INK2, fontsize=9)

    # 右：五分组超额收益（训练 / 测试）
    x = np.arange(1, 6)
    w = 0.38
    a2.bar(x - w / 2 - 0.01, q_tr.reindex(x).values, w, color=BLUE, label="train 2011-2019")
    a2.bar(x + w / 2 + 0.01, q_te.reindex(x).values, w, color=ORANGE, label="test 2021-2024")
    a2.axhline(0, color=INK2, linewidth=0.8)
    a2.set_xticks(x, ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    a2.set_title("Excess 10-day return by factor quintile (%)", loc="left", color=INK, fontsize=11)
    a2.legend(frameon=False, fontsize=9, labelcolor=INK2)

    fig.suptitle(name, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="AgentLiquidity")
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--cache", default="./csi300_panel.pkl")
    a = ap.parse_args()
    run = Path("runs") / a.agent
    state = json.loads((run / "state.json").read_text())
    panel = load_panel(a.data, cache=a.cache)
    label = make_label(panel)
    (run / "plots").mkdir(exist_ok=True)
    for fac in state["factors"]:
        if fac.get("error"):
            continue
        name, func, _ = compile_factor(fac["code"])
        f = apply_factor(func, panel)
        out = run / "plots" / f"gen{fac['gen']}_{name}.png"
        plot_one(name, f, label.reindex(f.index), out)
        print("已保存", out)
