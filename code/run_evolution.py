"""单 Agent 多代自动进化（CogAlpha 简化版）。

一条命令自动完成：
  第 0 代：大模型按主题生成一批因子
  每一代：选父代 → 变异 + 交叉生成子代 →（每 2 代）注入带反馈的新因子
  每个候选：代码修复 → 逻辑评审（不通过则改进一次再审）→ 运行检查 → 打分
筛选只看训练期（2011-2019），测试期（2021-2024）只用于最后展示。

用法（建议放后台）：
  nohup python -u run_evolution.py --data ../qlib_bin --gens 4 > evo.log 2>&1 &
  tail -f evo.log
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import prompts
from checker import compile_factor, run_check
from data import load_panel, make_label
from llm import chat, parse_functions
from metrics import evaluate
from runner import apply_factor

TRAIN = ("2011-01-01", "2019-12-31")
TEST = ("2021-01-01", "2024-12-31")
IC_KEYS = ["IC", "RankIC", "ICIR", "RankICIR"]
# 精英门槛（论文 B.4）。MI 只记录不卡门槛：我们的 MI 估计方法与论文不同，数值不可直接比。
ELITE = {"IC": 0.01, "RankIC": 0.01, "ICIR": 0.1, "RankICIR": 0.1}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def split(s, period):
    dt = s.index.get_level_values("datetime")
    return s[(dt >= period[0]) & (dt <= period[1])]


class Evolution:
    def __init__(self, a):
        self.a = a
        self.out = Path("runs") / f"{a.agent}_evo"
        self.out.mkdir(parents=True, exist_ok=True)
        self.panel = load_panel(a.data, cache=a.cache)
        self.label = make_label(self.panel)
        self.factors: list[dict] = []       # 所有候选（含被淘汰的）
        self.calls = 0

    # ---------- 大模型调用 ----------
    def ask(self, tag, prompt, temperature=None):
        self.calls += 1
        text = chat(prompt, temperature=temperature)
        d = self.out / "llm"
        d.mkdir(exist_ok=True)
        (d / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")
        (d / f"{tag}_response.txt").write_text(text, encoding="utf-8")
        return text

    # ---------- 质检：修复 → 评审 → 改进 ----------
    def quality_check(self, code, tag):
        """返回 (最终代码, 函数, 淘汰原因或 None, 质检记录)。"""
        notes = []
        for attempt in range(3):                       # 编译/运行出错时最多修 2 次
            name, func, err = compile_factor(code)
            if err is None:
                err = run_check(func, self.panel)
            if err is None:
                break
            notes.append(f"出错：{err}")
            if attempt == 2:
                return code, None, f"修复 2 次仍失败：{err}", notes
            fixed = parse_functions(self.ask(f"{tag}_repair{attempt}", prompts.CODE_REPAIR.format(
                error=err, code=code), temperature=0.8))
            if not fixed:
                return code, None, "修复后无法解析代码", notes
            code = fixed[0]
            notes.append("已自动修复")

        for attempt in range(2):                       # 评审不过：改进 1 次再审
            verdict = self.ask(f"{tag}_judge{attempt}", prompts.JUDGE.format(code=code), temperature=0.8)
            ok = re.search(r"VERDICT:\s*PASS", verdict, re.I) is not None
            issues = (re.search(r"ISSUES:\s*(.*)", verdict, re.S | re.I) or [None, verdict])[1].strip()
            if ok:
                notes.append("评审通过")
                return code, func, None, notes
            notes.append(f"评审不通过：{issues[:150]}")
            if attempt == 1:
                return code, None, "逻辑评审两次未通过", notes
            improved = parse_functions(self.ask(f"{tag}_improve", prompts.LOGIC_IMPROVE.format(
                issues=issues, code=code), temperature=0.8))
            if not improved:
                return code, None, "改进后无法解析代码", notes
            name, func, err = compile_factor(improved[0])
            if err is None:
                err = run_check(func, self.panel)
            if err is not None:
                return improved[0], None, f"改进后的代码出错：{err}", notes
            code = improved[0]
            notes.append("已按评审意见改进")

    # ---------- 打分 ----------
    def score(self, rec):
        f = apply_factor(rec.pop("_func"), self.panel)
        lab = self.label.reindex(f.index)
        tr = evaluate(split(f, TRAIN), split(lab, TRAIN))
        if tr["NaN率"] > 0.30:
            rec["error"] = f"NaN 比例 {tr['NaN率']:.0%} 超过 30%"
            return
        for o in self.alive():                         # 指标完全相同 = 与已有因子重复
            if abs(o["train"]["RankIC"] - tr["RankIC"]) < 1e-7 and abs(o["train"]["IC"] - tr["IC"]) < 1e-7:
                rec["error"] = f"与已有因子 {o['name']} 结果完全相同（重复）"
                return
        rec["train"] = tr
        rec["test"] = evaluate(split(f, TEST), split(lab, TEST), with_mi=False)

    def process(self, codes_with_origin, gen):
        """并发做质检（大模型调用），再逐个打分。"""
        jobs = [(c, o, f"g{gen}_{i}") for i, (c, o) in enumerate(codes_with_origin)]
        with ThreadPoolExecutor(self.a.workers) as ex:
            results = list(ex.map(lambda j: (j, self.quality_check(j[0], j[2])), jobs))
        names = {f["name"] for f in self.factors}
        for (code0, origin, tag), (code, func, err, notes) in results:
            name = compile_factor(code)[0] or "unparsed"
            if err is None and name in names:          # 大模型沿用了父代的名字：改名后保留
                new_name = f"{name}_g{gen}_{len(names)}"
                code = code.replace(name, new_name)
                name, func, err = compile_factor(code)
            rec = {"gen": gen, "origin": origin, "name": name, "code": code,
                   "qc": notes, "error": err}
            if rec["error"] is None:
                rec["_func"] = func
                self.score(rec)
            rec.pop("_func", None)
            names.add(name)
            status = "通过" if rec["error"] is None else "淘汰 - " + rec["error"]
            extra = f"  训练RankICIR={rec['train']['RankICIR']:+.3f}" if rec.get("train") else ""
            log(f"  [{origin}] {name}: {status}{extra}")
            self.factors.append(rec)
        self.save()

    # ---------- 选择 ----------
    def alive(self):
        return [f for f in self.factors if f.get("error") is None]

    def ranked(self):
        """综合得分：4 个 IC 类指标绝对值的平均排名（越高越好）。"""
        al = self.alive()
        if not al:
            return []
        m = pd.DataFrame([{k: abs(f["train"][k]) for k in IC_KEYS} for f in al])
        score = m.rank(pct=True).mean(axis=1)
        return [al[i] for i in score.sort_values(ascending=False).index]

    @staticmethod
    def brief(f):
        t = f["train"]
        return (f"{f['name']} (RankIC={t['RankIC']:+.4f}, RankICIR={t['RankICIR']:+.3f})")

    # ---------- 各代 ----------
    def generate_new(self, gen, n):
        theme, ideas = prompts.AGENT_THEMES[self.a.agent]
        r = self.ranked()
        eff = "\n".join(f"- {self.brief(f)}: worked; reuse its core idea." for f in r[:2]) or "(no history yet)"
        failed = [f for f in self.factors if f.get("error")]
        bad = (failed[-1:] + (r[-1:] if len(r) > 3 else []))[:2]
        ineff = "\n".join(f"- {f['name']}: {f['error'] or 'weak predictive power'}; avoid this pattern."
                          for f in bad) or "(no history yet)"
        existing = ", ".join(f["name"] for f in self.factors) or "(none yet)"
        p = prompts.BASE_AGENT.format(theme=theme, n=n, existing=existing,
                                      effective=eff, ineffective=ineff, ideas=ideas)
        codes = parse_functions(self.ask(f"g{gen}_generate", p))
        log(f"第 {gen} 代：大模型新生成 {len(codes)} 个因子")
        return [(c, "生成") for c in codes]

    def evolve(self, gen):
        parents = self.ranked()[: self.a.parents]
        log(f"第 {gen} 代父代：" + "、".join(p["name"] for p in parents))
        tasks = [("m", p) for p in parents] + \
                [("c", (parents[i], parents[i + 1])) for i in range(0, len(parents) - 1, 2)]

        def make(t):
            kind, x = t
            if kind == "m":
                txt = self.ask(f"g{gen}_mut_{x['name'][:40]}", prompts.MUTATION.format(
                    metrics=self.brief(x), code=x["code"]))
                return [(c, f"变异自 {x['name']}") for c in parse_functions(txt)[:1]]
            p1, p2 = x
            txt = self.ask(f"g{gen}_cross_{p1['name'][:20]}_{p2['name'][:20]}", prompts.CROSSOVER.format(
                metrics1=self.brief(p1), code1=p1["code"], metrics2=self.brief(p2), code2=p2["code"]))
            return [(c, f"交叉 {p1['name']} × {p2['name']}") for c in parse_functions(txt)[:1]]

        with ThreadPoolExecutor(self.a.workers) as ex:
            children = sum(ex.map(make, tasks), [])
        if gen % 2 == 0:                               # 论文：每 2 代注入新生成的因子
            children += self.generate_new(gen, self.a.inject)
        self.process(children, gen)

    # ---------- 保存与报告 ----------
    def save(self):
        (self.out / "state.json").write_text(json.dumps(self.factors, ensure_ascii=False, indent=1))

    def report(self):
        rows = []
        for f in self.factors:
            row = {"代": f["gen"], "因子": f["name"], "来源": f["origin"][:40]}
            if f.get("train"):
                row.update({"训练RankIC": f["train"]["RankIC"], "训练RankICIR": f["train"]["RankICIR"],
                            "训练MI": f["train"]["MI"], "测试RankIC": f["test"]["RankIC"],
                            "测试RankICIR": f["test"]["RankICIR"]})
            else:
                row["结果"] = f["error"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.out / "report.csv", index=False, encoding="utf-8-sig")
        pd.set_option("display.width", 250, "display.max_colwidth", 45)

        log("\n========== 每代汇总（只统计通过的因子）==========")
        ok = df.dropna(subset=["训练RankICIR"]) if "训练RankICIR" in df else df.iloc[0:0]
        summ = df.groupby("代").size().rename("候选数").to_frame()
        summ["通过数"] = ok.groupby("代").size()
        summ["训练|RankICIR|均值"] = ok.assign(a=ok["训练RankICIR"].abs()).groupby("代")["a"].mean()
        summ["训练|RankICIR|最高"] = ok.assign(a=ok["训练RankICIR"].abs()).groupby("代")["a"].max()
        print(summ.fillna(0).round(3).to_string(), flush=True)

        log("\n========== 排名前 10 的因子 ==========")
        top = [f["name"] for f in self.ranked()[:10]]
        print(df.set_index("因子").loc[top].round(4).to_string(), flush=True)
        elite = [f["name"] for f in self.alive()
                 if all(abs(f["train"][k]) >= v for k, v in ELITE.items())]
        log(f"达到精英门槛（IC/RankIC≥0.01，ICIR/RankICIR≥0.1）的因子：{len(elite)} 个")
        for n in elite:
            print("  ", n, flush=True)
        log(f"本次共调用大模型 {self.calls} 次；结果保存在 {self.out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="AgentLiquidity")
    ap.add_argument("--gens", type=int, default=4, help="第 0 代之后再进化几代")
    ap.add_argument("--n0", type=int, default=8, help="第 0 代生成几个因子")
    ap.add_argument("--parents", type=int, default=4, help="每代父代数量")
    ap.add_argument("--inject", type=int, default=4, help="每 2 代注入几个新因子")
    ap.add_argument("--workers", type=int, default=6, help="同时调用大模型的数量")
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--cache", default="./csi300_panel.pkl")
    a = ap.parse_args()

    t0 = time.time()
    ev = Evolution(a)
    log(f"开始：Agent={a.agent}，共 {a.gens + 1} 代")
    ev.process(ev.generate_new(0, a.n0), 0)
    for g in range(1, a.gens + 1):
        ev.evolve(g)
    ev.report()
    log(f"全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
