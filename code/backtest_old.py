"""因子合成 + 按 A 股真实交易规则回测，检验挖出的因子对实际投资有没有用。

对比三组（模型、回测规则完全相同）：
  baseline    8 个经典因子（人工设计）
  cogalpha    CogAlpha 挖出的全部精英因子
  synergy     从精英因子中按“组合贡献”贪心挑选的子集（在 2020 验证集上挑，避免偷看测试期）
两段样本外：2021-2024（与论文一致）、2025 至今（论文没见过的新数据）。

交易规则：T 日收盘后出信号，T+1 开盘成交；每天持有 50 只、最多换 5 只（论文 top50/drop5）；
买入费 0.05%、卖出费 0.15%；开盘涨停买不进、开盘跌停卖不出、停牌不交易；等权持有。

用法：python backtest.py --data ../qlib_bin --elites runs/cogalpha/elites.json [--lgbm]
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from baseline_factors import BASELINE
from checker import compile_factor
from data import load_panel, make_label, load_calendar, read_bin
from runner import apply_factor

TRAIN_END, VALID = "2019-12-31", ("2020-01-01", "2020-12-31")
PERIODS = {"2021-2024（论文测试期）": ("2021-01-01", "2024-12-31"),
           "2025至今（全新样本外）": ("2025-01-01", "2099-12-31")}
TOPK, NDROP, BUY_FEE, SELL_FEE = 50, 5, 0.0005, 0.0015
STEP, GAP = 126, 12          # 每 126 个交易日重新训练；训练标签与预测日之间隔 12 天，防止标签泄漏

_P = None


def _init(data, index, cache):
    global _P
    _P = load_panel(data, index=index, end="2099-12-31", cache=cache)


def _compute(item):
    name, code = item
    if code in BASELINE:
        func = BASELINE[code]
    else:
        _, func, err = compile_factor(code)
        if err:
            return name, None
    try:
        return name, apply_factor(func, _P).astype("float32")
    except Exception:
        return name, None


def cs_rank(df):
    """每天截面排名，映射到 [-0.5, 0.5]；缺失记 0。负 IC 的因子交给模型学方向。"""
    return (df.groupby(level="datetime").rank(pct=True) - 0.5).fillna(0).astype("float32")


def daily_rank_ic(pred, lab):
    """每天的秩相关系数（向量化计算，比逐日 apply 快几十倍）。"""
    d = pd.concat([pred.rename("p"), lab.rename("y")], axis=1).dropna()
    g = d.groupby(level="datetime")
    p = g["p"].rank() - g["p"].rank().groupby(level="datetime").transform("mean")
    y = g["y"].rank() - g["y"].rank().groupby(level="datetime").transform("mean")
    num = (p * y).groupby(level="datetime").sum()
    den = np.sqrt((p ** 2).groupby(level="datetime").sum() * (y ** 2).groupby(level="datetime").sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------- 模型 ----------------------
def fit_predict(X, y, dates, model):
    """滚动训练：每 STEP 个交易日用之前全部数据重训一次，预测接下来的 STEP 天。"""
    all_days = pd.DatetimeIndex(sorted(set(dates)))
    test_days = all_days[all_days >= pd.Timestamp(VALID[0])]
    preds = []
    for i in range(0, len(test_days), STEP):
        block = test_days[i:i + STEP]
        cutoff = all_days[max(0, all_days.get_loc(block[0]) - GAP)]
        tr = (dates < cutoff) & y.notna().values
        te = np.isin(dates, block)
        if model == "ridge":
            m = Ridge(alpha=10.0).fit(X[tr], y[tr])
        else:
            import lightgbm as lgb    # 论文参数
            m = lgb.LGBMRegressor(learning_rate=0.0001, num_leaves=32, max_depth=12, n_estimators=1000,
                                  reg_alpha=1.0, reg_lambda=1.0, colsample_bytree=0.8, subsample=0.8,
                                  subsample_freq=1, n_jobs=32, verbose=-1).fit(X[tr], y[tr])
        preds.append(pd.Series(m.predict(X[te]), index=y.index[te]))
    return pd.concat(preds)


def synergy_select(F, y, dates, max_k=30, corr_cap=0.8):
    """贪心选因子：每次加入能让“2011-2019 训练、2020 验证”的组合 RankIC 提升最多的因子。"""
    tr = (dates <= pd.Timestamp(TRAIN_END)) & y.notna().values
    tr &= np.isin(dates, pd.DatetimeIndex(sorted(set(dates[tr])))[::3])     # 训练集隔 3 天取样，加速
    va = (dates >= pd.Timestamp(VALID[0])) & (dates <= pd.Timestamp(VALID[1])) & y.notna().values
    yv = y[va]
    C = np.corrcoef(F[tr][::50].T.values) if F.shape[1] > 1 else np.ones((1, 1))
    chosen, best = [], -1.0
    cols = list(F.columns)
    while len(chosen) < max_k:
        cand_best, cand = best, None
        for j, c in enumerate(cols):
            if c in chosen or any(abs(C[j, cols.index(s)]) > corr_cap for s in chosen):
                continue
            use = chosen + [c]
            m = Ridge(alpha=10.0).fit(F.loc[tr, use].values, y[tr])
            ric = daily_rank_ic(pd.Series(m.predict(F.loc[va, use].values), index=yv.index), yv).mean()
            if ric > cand_best + 0.0005:
                cand_best, cand = ric, c
        if cand is None:
            break
        chosen.append(cand)
        best = cand_best
        print(f"    组合贡献选择：加入 {cand}，验证集 RankIC = {best:.4f}", flush=True)
    return chosen


# ---------------------- 回测 ----------------------
def backtest(score, panel, bench_ret):
    o = panel["open"].unstack("instrument")
    c = panel["close"].unstack("instrument")
    v = panel["volume"].unstack("instrument")
    days = o.index
    gap_ret = o.shift(-1) / c - 1                       # T+1 开盘相对 T 收盘的涨跌（判断涨跌停）
    hold_ret = (o.shift(-2) / o.shift(-1) - 1)          # T+1 开盘买 → T+2 开盘
    star = [s for s in o.columns if s.startswith("SH68")]
    chinext = [s for s in o.columns if s.startswith("SZ30")]
    lim = pd.DataFrame(0.10, index=days, columns=o.columns)
    lim[star] = 0.20
    lim.loc[lim.index >= "2020-08-24", chinext] = 0.20
    can_buy = (gap_ret < lim - 0.002) & (v.shift(-1) > 0)
    can_sell = (gap_ret > -lim + 0.002) & (v.shift(-1) > 0)

    S = score.unstack("instrument").sort_index()
    hold, rows = [], []
    for t in S.index:
        if t not in days or pd.isna(hold_ret.loc[t]).all():
            continue
        s = S.loc[t].dropna()
        rank = s.sort_values(ascending=False)
        last = sorted(hold, key=lambda x: -s.get(x, -np.inf))
        # Qlib TopkDropout：候选 = 未持有中得分最高的若干只；卖出持仓里排名最后的 NDROP 只
        new_cand = [x for x in rank.index if x not in hold][: NDROP + TOPK - len(last)]
        comb = sorted(last + new_cand, key=lambda x: -s.get(x, -np.inf))
        sell = [x for x in last if x in comb[-NDROP:] or x not in s.index]
        sell = [x for x in sell if can_sell.at[t, x]]           # 跌停或停牌的卖不出，继续持有
        n_buy = len(sell) + TOPK - len(last)
        buy = [x for x in new_cand if can_buy.at[t, x]][:n_buy]
        hold = [x for x in last if x not in sell] + buy
        r = hold_ret.loc[t, hold].fillna(0).mean() if hold else 0.0
        cost = (len(buy) * BUY_FEE + len(sell) * SELL_FEE) / TOPK
        rows.append({"date": t, "ret": r - cost, "turnover": (len(buy) + len(sell)) / (2 * TOPK)})
    res = pd.DataFrame(rows).set_index("date").sort_index()
    res["bench"] = bench_ret.reindex(res.index).fillna(0)
    res["excess"] = res["ret"] - res["bench"]
    return res


def stats(res, ic):
    ex = res["excess"]
    nav = (1 + res["ret"]).cumprod()
    cum_ex = ex.cumsum()
    return {"年化收益": res["ret"].mean() * 252, "基准年化": res["bench"].mean() * 252,
            "年化超额": ex.mean() * 252, "信息比率IR": ex.mean() / ex.std() * np.sqrt(252),
            "最大回撤": (nav / nav.cummax() - 1).min(), "超额最大回撤": (cum_ex - cum_ex.cummax()).min(),
            "日均换手": res["turnover"].mean(), "预测RankIC": ic.mean(), "交易日数": len(res)}


# ---------------------- 主程序 ----------------------
BENCH = {"csi300": "sh000300", "csi500": "sh000905", "csi100": "sh000903"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--index", default="csi300")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--elites", default="runs/cogalpha/elites.json")
    ap.add_argument("--lgbm", action="store_true", help="另外用论文的 LightGBM 参数跑一遍（较慢）")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", default="runs/backtest")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = a.cache or f"./{a.index}_panel_full.pkl"
    panel = load_panel(a.data, index=a.index, end="2099-12-31", cache=cache)
    print(f"数据：{panel.index.get_level_values('datetime').max().date()} 为最后一个交易日", flush=True)
    label = make_label(panel)[panel["in_index"]]
    cal = load_calendar(Path(a.data))
    b = read_bin(Path(a.data) / f"features/{BENCH[a.index]}/open.day.bin", cal)
    bench_ret = (b.shift(-2) / b.shift(-1) - 1)                       # 与持仓收益同口径：开盘到开盘

    items = [(f"base_{k}", k) for k in BASELINE]
    elites = json.loads(Path(a.elites).read_text()) if Path(a.elites).exists() else []
    items += [(e["name"], e["code"]) for e in elites]
    print(f"计算因子：经典 {len(BASELINE)} 个，CogAlpha 精英 {len(elites)} 个", flush=True)
    with mp.get_context("fork").Pool(a.procs, _init, (a.data, a.index, cache)) as pool:
        got = dict(pool.map(_compute, items))
    F = cs_rank(pd.DataFrame({k: v for k, v in got.items() if v is not None}))
    y = label.reindex(F.index)
    y = (y.groupby(level="datetime").rank(pct=True) - 0.5)
    dates = F.index.get_level_values("datetime").values

    base_cols = [c for c in F.columns if c.startswith("base_")]
    ai_cols = [c for c in F.columns if not c.startswith("base_")]
    groups = {"经典因子(8个)": base_cols}
    if ai_cols:
        groups[f"CogAlpha全部精英({len(ai_cols)}个)"] = ai_cols
        print("按组合贡献挑选因子（只用 2020 验证集）……", flush=True)
        syn = synergy_select(F[ai_cols], y, dates)
        groups[f"CogAlpha组合贡献精选({len(syn)}个)"] = syn
        groups[f"经典+CogAlpha精选({8 + len(syn)}个)"] = base_cols + syn
    models = ["ridge"] + (["lgbm"] if a.lgbm else [])

    rows, curves, yearly = [], {}, {}
    for gname, cols in groups.items():
        for model in models:
            tag = f"{gname} | {model}"
            print(f"训练与回测：{tag}", flush=True)
            pred = fit_predict(F[cols].values, y, dates, model)
            res = backtest(pred, panel, bench_ret)
            ic = daily_rank_ic(pred, label.reindex(pred.index)).sort_index()
            curves[tag] = res["excess"]
            yearly[tag] = res["excess"].groupby(res.index.year).sum()
            for pname, (s, e) in PERIODS.items():
                r = res.loc[s:e]
                if len(r):
                    rows.append({"组合": gname, "模型": model, "区间": pname, **stats(r, ic.loc[s:e])})

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    fmt = tab.copy()
    for c in ["年化收益", "基准年化", "年化超额", "最大回撤", "超额最大回撤", "日均换手"]:
        fmt[c] = (fmt[c] * 100).map("{:.1f}%".format)
    for c in ["信息比率IR", "预测RankIC"]:
        fmt[c] = fmt[c].map("{:.3f}".format)
    print("\n================ 回测结果 ================\n" + fmt.to_string(index=False), flush=True)
    yt = pd.DataFrame(yearly).T * 100
    yt.to_csv(out / "yearly_excess.csv", encoding="utf-8-sig")
    print("\n每年超额收益（%，相对基准指数，已扣费）：\n" + yt.round(1).to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for (tag, ex), col in zip(curves.items(), colors):
        en = (tag.replace("经典+CogAlpha精选", "Classic + CogAlpha selected").replace("CogAlpha全部精英", "CogAlpha all elites")
              .replace("CogAlpha组合贡献精选", "CogAlpha selected (synergy)").replace("经典因子", "Classic factors")
              .replace("个", ""))
        ax.plot(ex.index, ex.cumsum() * 100, label=en, color=col, linewidth=2)
    for s in ["2021-01-01", "2025-01-01"]:
        ax.axvline(pd.Timestamp(s), color="#52514e", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_title("Cumulative excess return vs CSI 300 (after costs, %)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0")
    fig.tight_layout()
    fig.savefig(out / "excess_curve.png", dpi=130)
    print(f"\n结果已保存到 {out}/（summary.csv、yearly_excess.csv、excess_curve.png）", flush=True)


if __name__ == "__main__":
    main()
