"""单个 Agent 的最小进化流程：生成 → 质检 → 打分筛选 → 变异/交叉 → 再打分。

用法：
  python evolve.py --stage gen0 --agent AgentLiquidity --mode api
  python evolve.py --stage gen1 --agent AgentLiquidity --mode api
mode=manual 时从 responses/<agent>/ 读取事先写好的大模型回答。
每一步的提示词都存到 runs/<agent>/，方便检查大模型到底收到了什么。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel, make_label
from metrics import evaluate
from runner import apply_factor
from checker import compile_factor, run_check
from llm import chat, parse_functions
import prompts

TRAIN = ("2011-01-01", "2019-12-31")
TEST = ("2021-01-01", "2024-12-31")
KEYS = ["IC", "RankIC", "ICIR", "RankICIR", "MI"]
# 论文的最低门槛（合格 / 精英）；IC 类指标取绝对值，负 IC 因子反向使用即可
MIN_Q = {"IC": 0.005, "RankIC": 0.005, "ICIR": 0.05, "RankICIR": 0.05, "MI": 0.02}
MIN_E = {"IC": 0.01, "RankIC": 0.01, "ICIR": 0.1, "RankICIR": 0.1, "MI": 0.02}


def split(s: pd.Series, period):
    dt = s.index.get_level_values("datetime")
    return s[(dt >= period[0]) & (dt <= period[1])]


class Ctx:
    def __init__(self, args):
        self.panel = load_panel(args.data, cache=args.cache)
        self.label = make_label(self.panel)
        self.out = Path("runs") / args.agent
        self.out.mkdir(parents=True, exist_ok=True)
        self.resp = Path("responses") / args.agent
        self.state_file = self.out / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {"factors": []}
        self.args = args

    def save(self):
        self.state_file.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))

    def ask(self, tag: str, prompt: str) -> str:
        (self.out / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")
        text = chat(prompt, mode=self.args.mode, response_file=self.resp / f"{tag}.txt")
        (self.out / f"{tag}_response.txt").write_text(text, encoding="utf-8")
        return text

    def score(self, code: str, gen: int, origin: str) -> dict:
        rec = {"gen": gen, "origin": origin, "code": code}
        name, func, err = compile_factor(code)
        rec["name"] = name
        if err is None:
            err = run_check(func, self.panel)
        if err is None:
            f = apply_factor(func, self.panel)
            lab = self.label.reindex(f.index)
            tr = evaluate(split(f, TRAIN), split(lab, TRAIN))
            if tr["NaN率"] > 0.30:
                err = f"NaN 比例 {tr['NaN率']:.0%} 超过 30%"
            else:
                rec["train"] = tr
                # 测试集只用于最后展示，不参与筛选
                rec["test"] = evaluate(split(f, TEST), split(lab, TEST), with_mi=False)
        rec["error"] = err
        print(f"  [{origin}] {name}: {'通过' if err is None else '淘汰 - ' + err}")
        return rec


def select(cands: list[dict], pct: float = 65):
    """论文规则：5 项指标都超过本代 65 分位且过最低门槛 → 合格。"""
    ok = [c for c in cands if c.get("error") is None]
    if not ok:
        return [], []
    m = pd.DataFrame([{k: abs(c["train"][k]) for k in KEYS} for c in ok])
    cut = m.quantile(pct / 100)
    qualified, elite = [], []
    for c, (_, row) in zip(ok, m.iterrows()):
        if all(row[k] >= cut[k] and row[k] >= MIN_Q[k] for k in KEYS):
            qualified.append(c)
        if all(row[k] >= MIN_E[k] for k in KEYS):
            elite.append(c)
    return qualified, elite


def fmt(m):
    return ", ".join(f"{k}={m[k]:.4f}" for k in KEYS if k in m)


def stage_gen0(ctx):
    theme, ideas = prompts.AGENT_THEMES[ctx.args.agent]
    p = prompts.BASE_AGENT.format(theme=theme, n=ctx.args.n, existing="(none yet)",
                                  effective="(no history yet)", ineffective="(no history yet)",
                                  ideas=ideas)
    codes = parse_functions(ctx.ask("gen0", p))
    print(f"第 0 代：大模型返回 {len(codes)} 个因子")
    ctx.state["factors"] = [ctx.score(c, 0, "生成") for c in codes]
    ctx.save()


def parents(ctx, k=2):
    """父代：本代合格因子；不足 k 个时按 |RankICIR| 补齐（小样本下的折中）。"""
    pool = [f for f in ctx.state["factors"] if f.get("error") is None]
    qualified, _ = select(pool)
    rest = sorted([f for f in pool if f not in qualified],
                  key=lambda f: -abs(f["train"]["RankICIR"]))
    return (qualified + rest)[:k]


def stage_gen1(ctx):
    p1, p2 = parents(ctx)
    print(f"父代：{p1['name']}、{p2['name']}")
    kids = []
    for i, par in enumerate([p1, p2], 1):
        text = ctx.ask(f"gen1_mutation{i}", prompts.MUTATION.format(
            metrics=fmt(par["train"]), code=par["code"]))
        kids += [ctx.score(c, 1, f"变异自 {par['name']}") for c in parse_functions(text)]
    text = ctx.ask("gen1_crossover", prompts.CROSSOVER.format(
        metrics1=fmt(p1["train"]), code1=p1["code"], metrics2=fmt(p2["train"]), code2=p2["code"]))
    kids += [ctx.score(c, 1, "交叉") for c in parse_functions(text)]
    ctx.state["factors"] += kids
    ctx.save()


def report(ctx):
    rows = []
    for f in ctx.state["factors"]:
        if f.get("error"):
            rows.append({"代": f["gen"], "因子": f["name"], "来源": f["origin"], "结果": f["error"]})
            continue
        tr, te = f["train"], f["test"]
        rows.append({"代": f["gen"], "因子": f["name"], "来源": f["origin"],
                     "训练IC": tr["IC"], "训练RankIC": tr["RankIC"], "训练RankICIR": tr["RankICIR"],
                     "训练MI": tr["MI"], "测试IC": te["IC"], "测试RankIC": te["RankIC"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 48)
    print(df.round(4).to_string(index=False))
    df.to_csv(ctx.out / "report.csv", index=False, encoding="utf-8-sig")
    _, elite = select([f for f in ctx.state["factors"] if f.get("error") is None])
    print("\n达到精英门槛的因子：", [f["name"] for f in elite] or "无")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen0", "gen1", "report"], required=True)
    ap.add_argument("--agent", default="AgentLiquidity")
    ap.add_argument("--mode", choices=["api", "manual"], default="api")
    ap.add_argument("--n", type=int, default=6, help="第 0 代生成几个因子")
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--cache", default="./csi300_panel.pkl")
    a = ap.parse_args()
    c = Ctx(a)
    {"gen0": stage_gen0, "gen1": stage_gen1, "report": lambda x: None}[a.stage](c)
    report(c)
