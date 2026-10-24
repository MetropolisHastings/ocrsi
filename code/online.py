"""在线 RSI：用不断到来的新市场数据做确认（量化投资特有的 RSI 改进）。

第 r 轮对应决策年 t（t = 2020, 2021, ...）：
    选择窗口 W_s = [t-2, t-1]（改进算子和初筛只能用 t-1 年底以前的数据）
    确认窗口 W_c = t 年（每轮都是此前从未用于确认的新数据）
    部署检验   = t+1 年（用 t 年底以前的数据训练，预测 t+1 年；记录这一轮决策的真实效果）
因为每轮的确认数据都是全新的，A1 下不再需要 2^{r-1} 的分支并集界，每轮取 α_r = α / R 即可（见理论第 8 节）。
对照规则 naive-online：只看 [t-2, t] 的验证分数（确认年也拿来挑选），不做置信下界检验。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import orchestra
import rsiframe as R
from cogalpha import log
from orchestra import CONF_METRICS, METRICS


def year_metrics(ev, st, t):
    """用 t 年 12 月 15 日以前的数据训练，预测 t+1 年：RankIC、RankICIR、中性化 RankIC。"""
    X = ev.matrix(st).values
    p = ev.fit_pred(X, f"{t}-12-15", f"{t + 1}-01-01", f"{t + 1}-12-31")
    ric, icir = ev.ic(p)
    neu, _ = ev.ic(bt.neutralize(p, ev.S))
    return {"test_ric": ric, "test_icir": icir, "test_neu_ric": neu, "n": X.shape[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(R.METHODS), required=True)
    ap.add_argument("--rule", choices=["naive", "certified"], default="certified")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--min_gain", type=float, default=0.0003)
    ap.add_argument("--bt", action="store_true", help="每轮对部署状态和起点状态在 t+1 年做 TopK-Dropout 回测")
    ap.add_argument("--gp_pop", type=int, default=80)
    ap.add_argument("--gp_gens", type=int, default=6)
    ap.add_argument("--admit_mode", choices=["full", "partial"], default="full",
                    help="full：通过后整体替换；partial：通过后与当前部署模型按 --mix 混合")
    ap.add_argument("--mix", type=float, default=0.5, help="部分接受时新模型的权重")
    ap.add_argument("--targets", default=",".join(str(v[1]) for v in METRICS.values()))
    # 大模型智能体（--method orch）专用
    ap.add_argument("--seed_pool", default="runs/cogalpha/elites.json", help="orch 的起点因子池")
    ap.add_argument("--cache", default="./csi300_panel.pkl", help="CogAlpha 训练期面板缓存")
    ap.add_argument("--budget", type=int, default=400, help="orch 每轮大模型调用上限")
    ap.add_argument("--max_tasks", type=int, default=5)
    ap.add_argument("--max_gens", type=int, default=4)
    ap.add_argument("--mock", action="store_true", help="用模拟大模型测试流程，不花钱")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--full_cache", required=True)
    ap.add_argument("--libcache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.admit, a.conf_years, a.cert_on, a.rounds = "naive", 1, "rsi", 0      # Frame 只用来加载数据和改进算子
    a.decision_year = None
    fr = R.Frame(a)
    ev = fr.ev
    years = [int(y) for y in a.years.split(",")]
    rng = np.random.default_rng(a.seed)
    S = fr.m.init_state()
    S0 = dict(S)
    if a.bt:
        from data import load_panel, load_calendar, read_bin
        _panel = load_panel(a.data, index="csi300", end="2099-12-31", cache=a.full_cache)
        _cal = load_calendar(Path(a.data))
        _b = read_bin(Path(a.data) / "features/sh000300/open.day.bin", _cal)
        _bench = (_b.shift(-2) / _b.shift(-1) - 1)

        def bt_year(st, t):
            X = ev.matrix(st).values
            p = ev.fit_pred(X, f"{t}-12-15", f"{t + 1}-01-01", f"{t + 1}-12-31")
            res = bt.backtest(p, _panel, _bench)
            ex = res["excess"]
            return {"ann_ret": float(res["ret"].mean() * 252), "ann_bench": float(res["bench"].mean() * 252),
                    "ann_excess": float(ex.mean() * 252),
                    "ir": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
                    "turnover": float(res["turnover"].mean()), "days": int(len(res)),
                    "daily_excess": [round(float(v), 6) for v in ex.values],
                    "dates": [str(d.date()) for d in res.index]}
    base_static = ev.evaluate(S0)[1]                   # RSI 归一化用的常数：与主实验相同（2021-2024）
    t_ = [float(x) for x in a.targets.split(",")]
    fr.targets = {k: t_[i] for i, k in enumerate(METRICS)}
    w = [1.0 / fr.span(CONF_METRICS[k], base_static) for k in CONF_METRICS]
    keys = list(CONF_METRICS.values())
    rsi_units = lambda m1, m0: float(np.mean([(m1[k] - m0[k]) * wi for k, wi in zip(keys, w)]))
    alpha_r = a.alpha / len(years)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rule_name = a.rule if a.admit_mode == "full" else f"{a.rule}-partial"
    f = out / f"online_{a.method}_{rule_name}_s{a.seed}.json"
    rec = {"method": a.method, "rule": rule_name, "seed": a.seed, "alpha_r": alpha_r, "mix": a.mix, "rounds": []}
    D = S                                                # 实际部署的模型（部分接受时是混合）；S 是提出候选的基准

    def blend_with(Dold, new):
        parts = Dold["blend"] if "blend" in Dold else [[1.0, Dold]]
        return {"blend": [[w * (1 - a.mix), s] for w, s in parts] + [[a.mix, new]]}
    for r, t in enumerate(years, 1):
        t0 = time.time()
        # 本轮窗口
        orchestra.TRAIN_END_SEL = f"{t - 3}-12-15"
        ev.sel = (f"{t - 2}-01-01", f"{t if a.rule == 'naive' else t - 1}-12-31")
        orchestra.CONF, orchestra.TRAIN_END_CONF = (f"{t}-01-16", f"{t}-12-31"), f"{t - 1}-12-15"
        fr.fit_end = f"{t - 3}-12-15"
        fr.decision_year = t
        if a.method == "gp" and fr.m.g is not None:
            fr.m.g.set_window(fr.fit_end)
        elif a.method == "gp":
            pass                                         # 第一次 propose 时按 fr.fit_end 构建
        moves = fr.m.propose(S, r, rng)
        base_sc, _ = ev.sel_score(S)
        cur, kept, cur_sc = S, [], base_sc
        for mv in moves:
            cand = {"names": cur["names"] + mv["add"], "model": cur["model"]} if "add" in mv else mv["replace"]
            sc, _ = ev.sel_score(cand)
            if sc > cur_sc + a.min_gain:
                cur, cur_sc = cand, sc
                kept.append(mv["desc"])
        cert = None
        Dnew = (cur if a.admit_mode == "full" else blend_with(D, cur)) if kept else D
        if a.rule == "certified" and kept:
            cert = ev.confirm_rsi(D, Dnew, alpha_r, w, a.boot, a.block, a.seed * 100 + r)
        accepted = bool(kept) and (cert is None or cert["passed"])
        before = year_metrics(ev, D, t)                  # 部署年 t+1：不改 vs 改，两种状态都算（反事实）
        after = year_metrics(ev, Dnew, t) if kept else before
        new_S = cur if accepted else S
        new_D = Dnew if accepted else D
        realized = rsi_units(after, before)              # 这一批改动在部署年的真实效果（RSI 单位）
        row = {"round": r, "decision_year": t, "deploy_year": t + 1, "candidates": len(moves), "kept": kept,
               "accepted": accepted, "cert": cert, "sel_gain": cur_sc - base_sc,
               "deploy_before": before, "deploy_after_candidate": after,
               "realized_gain_if_accepted": realized, "realized_gain": realized if accepted else 0.0,
               "state_vs_S0": rsi_units(year_metrics(ev, new_D, t), year_metrics(ev, S0, t)),
               "minutes": round((time.time() - t0) / 60, 2)}
        if hasattr(fr.m, "feedback"):
            fr.m.feedback(r, kept, accepted, cert)
            row["llm_calls"] = getattr(fr.m, "calls_this_round", None)
        if a.bt:
            row["bt_deployed"] = bt_year(new_D, t)          # 本轮决策之后实际部署的状态
            row["bt_static"] = bt_year(S0, t)               # 从不改动：起点状态
            log(f"  回测 {t + 1}：部署状态年化超额 {row['bt_deployed']['ann_excess']:+.2%}，"
                f"起点状态 {row['bt_static']['ann_excess']:+.2%}")
        S, D = new_S, new_D
        rec["rounds"].append(row)
        log(f"第 {r} 轮（确认 {t}，部署 {t + 1}）：保留 {len(kept)}/{len(moves)}，"
            + (f"认证增益 {cert['gain_rsi']:+.4f} 下界 {cert['lcb_rsi']:+.4f} {'通过' if cert['passed'] else '未通过'}，" if cert else "")
            + f"{'接受' if accepted else '不接受'}；候选在部署年的真实增益 {realized:+.4f}（RSI 单位），"
            f"当前状态相对起点 {row['state_vs_S0']:+.4f}，{row['minutes']} 分钟")
        f.write_text(json.dumps({**rec, "state": S, "deployed": D}, indent=1, default=str))


if __name__ == "__main__":
    main()
