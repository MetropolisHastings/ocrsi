import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

import backtest as bt
import orchestra
import rsiframe as R
from baseline_factors import BASELINE
from cogalpha import _init_worker, eval_code
from llm import parse_functions


def call_llm(prompt):
    body = {
        "model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "thinking": {"type": "disabled"},
        "max_tokens": 1200,
    }
    req = urllib.request.Request(
        os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        },
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"].get("content", ""), data.get("usage", {})


def year_metrics(ev, state, year):
    X = ev.matrix(state).values
    pred = ev.fit_pred(
        X,
        f"{year}-12-15",
        f"{year + 1}-01-01",
        f"{year + 1}-12-31",
    )
    ric, icir = ev.ic(pred)
    neu, _ = ev.ic(bt.neutralize(pred, ev.S))
    return {
        "test_ric": ric,
        "test_icir": icir,
        "test_neu_ric": neu,
        "n": X.shape[1],
    }


def normalized_gain(after, before):
    targets = {
        "test_ric": 0.15,
        "test_icir": 1.0,
        "test_neu_ric": 0.10,
    }
    vals = []
    for k, target in targets.items():
        gap = target - before[k]
        vals.append((after[k] - before[k]) / gap if abs(gap) > 1e-12 else 0.0)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020,2021")
    ap.add_argument("--rule", choices=["naive", "certified"], default="certified")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--libcache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bt_out", default="runs/frame/online_bt")
    a = ap.parse_args()

    _init_worker(a.data, a.cache)

    class Args:
        method = "ridge"
        admit = "certified"
        conf_years = 1
        cert_on = "rsi"
        rounds = 0
        seed = a.seed
        alpha = 0.1
        boot = 2000
        block = 20
        min_gain = 0.0003
        gp_pop = 80
        gp_gens = 6
        targets = ",".join(str(v[1]) for v in orchestra.METRICS.values())
        procs = 2
        data = a.data
        full_cache = a.cache
        libcache = a.libcache
        out = a.out

    fr = R.Frame(Args())
    ev = fr.ev

    # ---- 回测（自动补丁）----
    from data import load_panel, load_calendar, read_bin
    _panel = load_panel(a.data, index="csi300", end="2099-12-31", cache=a.cache)
    _cal = load_calendar(Path(a.data))
    _b = read_bin(Path(a.data) / "features/sh000300/open.day.bin", _cal)
    _bench = (_b.shift(-2) / _b.shift(-1) - 1)

    def bt_year(st, t):
        X = ev.matrix(st).values
        p = ev.fit_pred(X, f"{t}-12-15", f"{t + 1}-01-01", f"{t + 1}-12-31")
        res = bt.backtest(p, _panel, _bench)
        ex = res["excess"]
        return {"ann_excess": float(ex.mean() * 252),
                "ir": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
                "daily_excess": [round(float(v), 6) for v in ex.values],
                "dates": [str(d.date()) for d in res.index]}
    all_records = []
    years = [int(x) for x in a.years.split(",")]

    state = {"names": list(BASELINE), "model": "ridge"}
    S0 = dict(state)
    all_rows = []
    total_calls = 0
    total_tokens = 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"开始决策年 {year}", flush=True)

        orchestra.TRAIN_END_SEL = f"{year - 3}-12-15"
        ev.sel = (
            f"{year - 2}-01-01",
            f"{year - 1 if a.rule == 'certified' else year}-12-31",
        )
        orchestra.CONF = (f"{year}-01-16", f"{year}-12-31")
        orchestra.TRAIN_END_CONF = f"{year - 1}-12-15"

        base_score, _ = ev.sel_score(state)
        candidates = []
        records = []

        prompt = f"""
你是量化因子研究员，现在是决策年 {year}。
只能使用单只股票 DataFrame 的五个字段：
open, high, low, close, volume。

请生成一个预测未来10个交易日收益的因子。
禁止 import，禁止使用 np 或 pd，禁止使用未来数据。
只能使用 df 的五个字段和 Series 自带方法。

严格输出：
<<function 1>>
def factor_cot_{year}(df):
    ...
    return series
<</function 1>>
"""

        for i in range(a.calls):
            text, usage = call_llm(prompt)
            total_calls += 1
            total_tokens += int(usage.get("total_tokens", 0))

            blocks = parse_functions(text)
            if not blocks:
                records.append({
                    "call": i,
                    "valid": False,
                    "error": "无法解析函数",
                    "usage": usage,
                })
                continue

            code = blocks[0]
            name = f"cot_{year}_s{a.seed}_{i}"
            start = code.find("def ")
            left = code.find("(", start)

            if start < 0 or left < 0:
                records.append({
                    "call": i,
                    "valid": False,
                    "error": "函数定义格式错误",
                    "usage": usage,
                })
                continue

            code = code[:start] + f"def {name}" + code[left:]
            result = eval_code(code)

            rec = {
                "call": i,
                "name": name,
                "code": code,
                "usage": usage,
            }
            rec.update(result)
            records.append(rec)

            if "error" not in result:
                candidates.append({"name": name, "code": code})

        ev.compute(candidates)

        best = state
        best_score = base_score
        kept = []

        for cand in candidates:
            trial = {
                "names": best["names"] + [cand["name"]],
                "model": "ridge",
            }
            score, _ = ev.sel_score(trial)
            if score > best_score + Args.min_gain:
                best = trial
                best_score = score
                kept.append(cand["name"])

        cert = None
        accepted = bool(kept)

        if accepted and a.rule == "certified":
            base_off = ev.evaluate(state)[1]
            weights = [
                1.0 / fr.span(orchestra.CONF_METRICS[k], base_off)
                for k in orchestra.CONF_METRICS
            ]
            cert = ev.confirm_rsi(
                state,
                best,
                a.alpha / len(years),
                weights,
                a.boot,
                a.block,
                a.seed * 100 + year,
            )
            accepted = bool(cert["passed"])

        before = year_metrics(ev, state, year)
        after = year_metrics(ev, best, year) if kept else before
        realized = normalized_gain(after, before)

        if accepted:
            state = best

        row = {
            "decision_year": year,
            "deploy_year": year + 1,
            "candidates": len(candidates),
            "kept": kept,
            "accepted": accepted,
            "cert": cert,
            "selection_score_before": base_score,
            "selection_score_after": best_score,
            "deploy_before": before,
            "deploy_after_candidate": after,
            "realized_gain": realized if accepted else 0.0,
            "calls_total": total_calls,
            "tokens_total": total_tokens,
        }
        all_rows.append(row)

        all_rows[-1]["bt_deployed"] = bt_year(state, year)       # 本轮决策后实际部署的状态
        all_rows[-1]["bt_static"] = bt_year(S0, year)            # 从不改动：起点状态
        all_rows[-1]["round"] = len(all_rows)
        all_records.extend([dict(r, year=year) for r in records])
        _bt = Path(a.bt_out); _bt.mkdir(parents=True, exist_ok=True)
        (_bt / f"online_cot_{a.rule}_s{a.seed}.json").write_text(json.dumps(
            {"method": "cot", "rule": a.rule, "seed": a.seed, "rounds": all_rows,
             "all_codes": all_records}, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "records.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (out / "online_result.json").write_text(
            json.dumps(
                {
                    "rule": a.rule,
                    "seed": a.seed,
                    "rounds": all_rows,
                    "calls_total": total_calls,
                    "tokens_total": total_tokens,
                    "state": state,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(
            f"决策年 {year} 完成：候选 {len(candidates)}，"
            f"保留 {len(kept)}，接受={accepted}，"
            f"部署期增益={realized:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
