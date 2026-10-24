"""CogAlpha-Orchestra：在 CogAlpha 因子挖掘上加入“总指挥-专家-子代理”分层编排、文件即总线和 RSI 指数。

思路来自三篇论文：
  - CogAlpha：21 个主题 Agent 用大模型生成、进化、质检因子（这里降为子代理）；
  - AiScientist：总指挥只看“工作区地图 + 摘要”，专家通过共享文件协作（File-as-Bus）；
  - RSIGym：RSI 指数 = 各项指标“补上离目标差距的几成”的平均，逐轮衡量增益。

每一轮（round）：
  1. 诊断专家（大模型）：读因子池在训练期和 2020 验证期的表现、风格暴露、各主题统计，写 insights.md；
  2. 总指挥（大模型）：读工作区地图、诊断结论、预算，制定本轮计划 plan.md（派哪些主题、新挖还是进化、方向）；
  3. 设计专家：按计划调度主题子代理生成 / 进化因子，质检专家（代码检查、未来函数、逻辑评审）把关；
  4. 评估专家（程序）：把新精英加入因子池，计算验证期指标（写入 exp_log.md，大模型可见）
     和测试期指标（写入 hidden/，大模型看不到），并计算 RSI 指数。
下一轮从更新后的因子池出发，体现“递归”改进。起点（第 0 轮）是上一版 CogAlpha 的精英因子。

数据隔离：大模型只能看到 2011-2020；2021-2024 和 2025 至今只用于计算 RSI，不进入任何提示词。

用法（服务器后台）：
  nohup python -u orchestra.py --data ../qlib_bin --seed_pool runs/cogalpha/elites.json --run orch1 > orch1.log 2>&1 &
中断后重新运行同一条命令，会从最后完成的一轮继续。
"""
import argparse
import json
import multiprocessing as mp
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import backtest as bt
import cogalpha
import prompts
from agents import AGENTS
from cogalpha import CogAlpha, eval_code, log
from data import load_panel, make_label
from statistics import NormalDist

DEV = ("2020-01-01", "2020-12-31")          # 验证期：大模型可见
TEST = ("2021-01-01", "2024-12-31")         # 测试期：只用于 RSI
NEW = ("2025-01-01", "2099-12-31")          # 新样本：只用于 RSI
TRAIN_END_DEV, TRAIN_END_OFF = "2019-12-15", "2020-12-15"   # 留出约 10 个交易日，防止 10 日标签跨期
SEL = ("2018-01-01", "2020-12-31")         # 入池 / 剔除判断用的验证窗口（3 年，比只用 2020 更不容易过拟合）
TRAIN_END_SEL = "2017-12-15"
# 可认证模式（--admit certified）：2020 年成为确认窗口 W_c，大模型和挑选过程都看不到它的任何数字。
#   选择窗口 W_s = 2018-2019（逐年规则 T=2），大模型可见的验证期改为 2019，
#   确认窗口从 2020-01-16 开始，避开 2019 年末 10 日标签伸进 2020 年的那几天。
CERT_SEL = ("2018-01-01", "2019-12-31")
CERT_DEV, CERT_TRAIN_END_DEV = ("2019-01-01", "2019-12-31"), "2018-12-15"
CONF, TRAIN_END_CONF = ("2020-01-16", "2020-12-31"), "2019-12-15"
CONF_METRICS = {"ric": "test_ric", "icir": "test_icir", "neu": "test_neu_ric"}   # 确认检验的 M=3 项 → 对应 RSI 的测试期指标

# RSI 的四个“考试科目”：(中文名, 默认目标值)。目标值可用 --targets 修改
METRICS = {
    "test_ric": ("预测力：测试期合成 RankIC", 0.15),
    "test_icir": ("稳定性：测试期合成 RankICIR", 1.00),
    "test_neu_ric": ("纯选股：测试期中性化后 RankIC", 0.10),
    "new_ric": ("新样本：2025 至今合成 RankIC", 0.10),
}

# ============================ 提示词（总指挥、诊断专家） ============================
DIAGNOSE = """You are the DIAGNOSIS EXPERT of a quantitative factor-mining research team (CSI300 stocks,
daily OHLCV only, target = 10-day forward return). Data you may see: TRAIN 2011-2019 and VALIDATION 2020.
Later data is held out and you must not speculate about it.

Write the diagnosis of the current factor pool for the research lead. Be concrete and short (<= 300 words),
using exactly these four markdown sections:
## What works
## What fails
## Unexplored gaps
## Recommendations
In "What fails", pay attention to: factors whose sign or strength changes from train to validation (decay),
factors highly exposed to style (size / volatility / momentum / reversal), and themes that keep failing checks.
Recommendations must name agents from the list and say what kind of factor to build or avoid.

### Composite model on the current pool (Ridge, validation 2020)
{composite}

### Per-theme statistics of the pool
{themes}

### Strongest factors (by |validation RankIC|)
{top}

### Factors that decayed most from train to validation
{decayed}

### Previous round
{previous}
"""

PLAN = """You are the RESEARCH LEAD (orchestrator) of a quantitative factor-mining team. You decide what the
team does next; specialist agents do the actual work. Goal: improve the composite model built from the factor
pool, especially its stability and its stock-selection skill after removing style exposure, on unseen future data.

### Workspace map
{map}

### Latest diagnosis (from the diagnosis expert)
{insights}

### Available theme agents (sub-agents) and their record in the pool
{themes}

### Validation metrics of the composite model by round
{history}

### Budget
About {budget} LLM calls remain for this round. A task costs roughly 15 calls per generation.

Plan this round. Rules:
- At most {max_tasks} tasks; each task uses one agent from the list above.
- mode "evolve" improves that agent's existing pool factors; mode "new" explores fresh ideas.
- gens is the number of evolution generations, an integer from 1 to {max_gens}.
- focus is a concrete instruction (1-3 sentences) for the agent: what signal to build, what to avoid.
- New factors enter the pool ONLY if they raise the composite's validation score (raw + style-neutral RankIC)
  AND do not make any single year of 2018, 2019, 2020 worse, so signals that only work in one year are rejected;
  near-copies of pool factors (correlation > {dup}) are rejected. Prefer genuinely different signals.
- Set "stop": true only if you believe further rounds cannot help.

Return ONLY a JSON object, no other text:
{{"rationale": "<2-3 sentences>", "tasks": [{{"agent": "AgentLiquidity", "mode": "evolve", "focus": "...", "gens": 2}}], "stop": false}}
"""


# ============================ 评估专家（程序，不调大模型） ============================
class PoolEval:
    """在完整面板（含 2021 之后）上计算因子值、合成预测和各项指标。"""

    def __init__(self, data, procs, cache, certified=False):
        self.data, self.procs, self.cache = data, procs, cache
        self.sel, self.dev, self.train_end_dev = (CERT_SEL, CERT_DEV, CERT_TRAIN_END_DEV) if certified \
            else (SEL, DEV, TRAIN_END_DEV)
        self.panel = load_panel(data, index="csi300", end="2099-12-31", cache=cache)
        self.label = make_label(self.panel)[self.panel["in_index"]]
        self.idx = self.label.index
        self.dates = self.idx.get_level_values("datetime")
        self.y = self.label.groupby(level="datetime").rank(pct=True) - 0.5
        self.S = bt.style_exposure(self.panel, self.idx)
        self.vals = {}

    def compute(self, factors):
        todo = [(f["name"], f["code"]) for f in factors if f["name"] not in self.vals]
        if not todo:
            return
        with mp.get_context("fork").Pool(self.procs, bt._init, (self.data, "csi300", self.cache)) as p:
            for name, v in p.map(bt._compute, todo):
                if v is not None:
                    self.vals[name] = v

    def matrix(self, names):
        names = [n for n in names if n in self.vals]
        return bt.cs_rank(pd.DataFrame({n: self.vals[n] for n in names}).reindex(self.idx))

    def mask(self, start, end):
        return (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))

    def fit_pred(self, X, train_end, start, end):
        days = pd.DatetimeIndex(sorted(set(self.dates[self.dates <= pd.Timestamp(train_end)])))[::3]
        tr = np.isin(self.dates, days) & self.y.notna().values
        m = Ridge(alpha=10.0).fit(X[tr], self.y.values[tr])
        te = self.mask(start, end)
        return pd.Series(m.predict(X[te]), index=self.idx[te])

    def ic(self, pred):
        s = bt.daily_rank_ic(pred, self.label.reindex(pred.index)).sort_index()
        return float(s.mean()), float(s.mean() / s.std()) if len(s) > 1 and s.std() > 0 else 0.0

    def sel_score(self, names):
        """入池 / 剔除用的分数：2011-2017 训练、2018-2020 验证，合成 RankIC 与中性化 RankIC 的平均。
        同时返回 2018、2019、2020 每一年的分数，用来要求“每一年都不变差”，防止只适合某一年的因子入池。"""
        X = self.matrix(names).values
        pred = self.fit_pred(X, TRAIN_END_SEL, *self.sel)
        ric = bt.daily_rank_ic(pred, self.label.reindex(pred.index)).sort_index()
        neu = bt.daily_rank_ic(bt.neutralize(pred, self.S), self.label.reindex(pred.index)).sort_index()
        both = (ric + neu.reindex(ric.index)) / 2
        return float(both.mean()), both.groupby(both.index.year).mean().to_dict()

    def evaluate(self, names):
        """返回 (dev, official)：dev 给大模型看，official 只用于 RSI。"""
        X = self.matrix(names).values
        dev = self.fit_pred(X, self.train_end_dev, *self.dev)
        d_ric, d_icir = self.ic(dev)
        d_neu, _ = self.ic(bt.neutralize(dev, self.S))
        sty = {k: float(np.corrcoef(dev.values, self.S[k].reindex(dev.index).values)[0, 1]) for k in bt.STYLES}
        off = self.fit_pred(X, TRAIN_END_OFF, TEST[0], NEW[1])
        od = off.index.get_level_values("datetime")
        t = off[(od >= pd.Timestamp(TEST[0])) & (od <= pd.Timestamp(TEST[1]))]
        n = off[od >= pd.Timestamp(NEW[0])]
        t_ric, t_icir = self.ic(t)
        t_neu, _ = self.ic(bt.neutralize(t, self.S))
        n_ric, _ = self.ic(n)
        dev_m = {"dev_ric": d_ric, "dev_icir": d_icir, "dev_neu_ric": d_neu, "n_factors": X.shape[1],
                 **{f"style_{k}": v for k, v in sty.items()}}
        off_m = {"test_ric": t_ric, "test_icir": t_icir, "test_neu_ric": t_neu, "new_ric": n_ric,
                 "n_factors": X.shape[1]}
        return dev_m, off_m

    # ---------- 确认检验（定理第 5 节）：只返回置信下界，不给大模型看 ----------
    def conf_series(self, names):
        X = self.matrix(names).values
        pred = self.fit_pred(X, TRAIN_END_CONF, *CONF)
        lab = self.label.reindex(pred.index)
        ric = bt.daily_rank_ic(pred, lab).sort_index()
        neu = bt.daily_rank_ic(bt.neutralize(pred, self.S), lab).sort_index().reindex(ric.index)
        return pd.DataFrame({"ric": ric, "neu": neu}).dropna()

    @staticmethod
    def _conf_stats(o, n):
        icir = lambda x: x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
        return np.array([n[:, 0].mean() - o[:, 0].mean(), icir(n[:, 0]) - icir(o[:, 0]),
                         n[:, 1].mean() - o[:, 1].mean()])

    def confirm(self, old, new, alpha_r, B=2000, block=20, seed=0):
        """在确认窗口上比较 old 与 new 两个因子池：返回各项增益、分块 bootstrap 标准误和单侧置信下界。"""
        so, sn = self.conf_series(old), self.conf_series(new)
        idx = so.index.intersection(sn.index)
        o, n = so.loc[idx].values, sn.loc[idx].values
        T = len(idx)
        point = self._conf_stats(o, n)
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(T / block))
        boots = np.empty((B, 3))
        for b in range(B):                               # 移动分块 bootstrap：保留日序列的自相关
            st = rng.integers(0, T - block + 1, nb)
            ix = (st[:, None] + np.arange(block)).ravel()[:T]
            boots[b] = self._conf_stats(o[ix], n[ix])
        se = boots.std(axis=0, ddof=1)
        z = NormalDist().inv_cdf(1 - alpha_r)
        lcb = point - z * se
        keys = list(CONF_METRICS)
        return {"days": T, "alpha_r": alpha_r, "z": z, **{f"gain_{k}": float(point[i]) for i, k in enumerate(keys)},
                **{f"se_{k}": float(se[i]) for i, k in enumerate(keys)},
                **{f"lcb_{k}": float(lcb[i]) for i, k in enumerate(keys)}, "passed": bool((lcb > 0).all())}

    def factor_table(self, factors):
        """每个因子：训练期 RankICIR、验证期 RankIC、和四种风格的相关性（只用 2011-2020）。"""
        names = [f["name"] for f in factors if f["name"] in self.vals]
        F = self.matrix(names)
        dev = self.mask(*self.dev)
        tr_days = pd.DatetimeIndex(sorted(set(self.dates[self.mask("2011-01-01", "2019-12-31")])))[::20]
        sm = np.isin(self.dates, tr_days)
        lab_dev = self.label[dev]
        rows = []
        for f in factors:
            n = f["name"]
            if n not in F:
                continue
            v = F[n]
            ric_dev = bt.daily_rank_ic(v[dev], lab_dev).mean()
            sty = {k: float(np.corrcoef(v.values[sm], self.S[k].values[sm])[0, 1]) for k in bt.STYLES}
            rows.append({"name": n, "agent": f.get("agent", "Seed"), "train_ricir": f["train"]["RankICIR"],
                         "train_ric": f["train"]["RankIC"], "dev_ric": float(ric_dev),
                         **{f"c_{k}": s for k, s in sty.items()}})
        return pd.DataFrame(rows)


# ============================ 总指挥 + 专家 ============================
class Orchestra(CogAlpha):
    def __init__(self, a):
        super().__init__(a)
        self.history = self._load("history.json", [])
        self.ws = self.out / "workspace"
        self.hidden = self.out / "hidden"
        self.ws.mkdir(exist_ok=True)
        self.hidden.mkdir(exist_ok=True)
        self.state = self._load("state.json", {"round": 0, "dev": [], "official": [], "calls": [], "minutes": []})
        self.focus = ""
        self.budget_end = 10 ** 9
        t = [float(x) for x in a.targets.split(",")]
        self.targets = {k: t[i] for i, k in enumerate(METRICS)}
        log("加载完整面板，计算风格暴露……")
        self.cert = a.admit == "certified"
        if self.cert:
            log("可认证模式：选择窗口 2018-2019，确认窗口 2020（不对大模型开放），大模型可见验证期改为 2019")
            a.max_prune = 0
        self.state.setdefault("cert", [])
        self.ev = PoolEval(a.data, a.procs, a.full_cache, certified=self.cert)
        self.pool_f = self._load("pool.json", None)
        if self.pool_f is None:
            self.pool_f = self.seed_pool(a.seed_pool)
            self._save("pool.json", self.pool_f)
        log(f"因子池：{len(self.pool_f)} 个；登记去重信息……")
        for f, r in zip(self.pool_f, self.pool.map(eval_code, [f["code"] for f in self.pool_f])):
            if "sig" in r:
                self.sigs[f["name"]] = r["sig"]

    # ---------- 工具 ----------
    def duplicate_of(self, rec):
        """去重：与已有因子截面相关 > --dup（默认 0.7）视为重复，逼它挖真正不同的信号。"""
        if rec["name"] in self.sigs:
            return rec["name"]
        v = rec["_sig"]
        for name, s in self.sigs.items():
            if abs(np.corrcoef(v, s)[0, 1]) > self.a.dup:
                return name
        return None

    def seed_pool(self, path):
        seed = json.loads(Path(path).read_text())
        need = [s for s in seed if "train" not in s]
        if need:
            for s, r in zip(need, self.pool.map(eval_code, [s["code"] for s in need])):
                if r.get("train"):
                    s["train"], s["name"] = r["train"], r["name"]
        pool = [{"name": s["name"], "code": s["code"], "agent": s.get("agent", "Seed"), "gen": s.get("gen", 0),
                 "train": s["train"], "round": 0} for s in seed if s.get("train")]
        log(f"起点因子池来自 {path}：{len(pool)} 个")
        return pool

    def ask(self, tag, prompt, temperature=None):
        if self.calls >= self.budget_end:            # 本轮预算用完：不再调用
            return ""
        return super().ask(f"r{self.state['round'] + 1}_{tag}", prompt, temperature)

    def over(self):
        return self.calls >= self.budget_end

    def write(self, name, text, append=False):
        p = self.ws / name
        with open(p, "a" if append else "w", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")

    def read(self, name, limit=None):
        p = self.ws / name
        t = p.read_text(encoding="utf-8") if p.exists() else "(empty)"
        return t if limit is None else t[-limit:]

    def update_map(self):
        rounds = self.state["round"]
        self.write("map.md", f"""# Workspace map (round {rounds} completed)
- pool.json         : factor pool, {len(self.pool_f)} factors (each: name, code, agent, train metrics, round added)
- insights.md       : latest diagnosis by the diagnosis expert
- plan.md           : plans of all rounds (rationale + tasks)
- exp_log.md        : append-only log of tasks and VALIDATION results
- themes.md         : per-theme statistics of the pool (train / validation)
Held-out results (2021 onward) are stored outside the workspace and are never shown to agents.
""")

    # ---------- 专家与子代理调用时附加总指挥的方向 ----------
    def lead_note(self):
        ins = self.read("insights.md", 1200)
        return (f"\n\n### Direction from the research lead (follow it)\n{self.focus}"
                f"\n\n### Lessons from earlier experiments\n{ins}")

    def generate(self, agent, gen, n, alive):
        lvl, theme, ideas = AGENTS[agent]
        AGENTS[agent] = (lvl, theme, ideas + self.lead_note())
        try:
            return super().generate(agent, gen, n, alive)
        finally:
            AGENTS[agent] = (lvl, theme, ideas)

    def evolve_children(self, agent, gen, parents):
        orig = prompts.EVOLVE_BATCH
        prompts.EVOLVE_BATCH = orig + self.lead_note().replace("{", "{{").replace("}", "}}")
        try:
            return super().evolve_children(agent, gen, parents)
        finally:
            prompts.EVOLVE_BATCH = orig

    # ---------- 诊断专家 ----------
    def theme_stats(self, tab):
        rows = []
        for ag in AGENTS:
            sub = tab[tab["agent"] == ag]
            hist = [h for h in self.history if h.get("agent") == ag]
            rows.append({"agent": ag, "theme": AGENTS[ag][1][:60], "in_pool": len(sub),
                         "mean|train_RankICIR|": round(float(sub["train_ricir"].abs().mean()), 3) if len(sub) else None,
                         "mean|dev_RankIC|": round(float(sub["dev_ric"].abs().mean()), 4) if len(sub) else None,
                         "sign_kept": f"{int((np.sign(sub['train_ric']) == np.sign(sub['dev_ric'])).sum())}/{len(sub)}",
                         "tried_this_run": len(hist),
                         "failed_this_run": sum(1 for h in hist if h.get("error"))})
        return pd.DataFrame(rows)

    def diagnose(self, tab, dev):
        tab = tab.copy()
        tab["style_max"] = tab[[f"c_{k}" for k in bt.STYLES]].abs().max(axis=1)
        tab["decay"] = tab["train_ric"].abs() - np.where(np.sign(tab["train_ric"]) == np.sign(tab["dev_ric"]),
                                                          tab["dev_ric"].abs(), -tab["dev_ric"].abs())
        cols = ["name", "agent", "train_ricir", "dev_ric", "style_max"]
        th = self.theme_stats(tab)
        self.write("themes.md", "# Per-theme statistics\n" + th.to_string(index=False))
        comp = (f"validation RankIC={dev['dev_ric']:.4f}, RankICIR={dev['dev_icir']:.3f}, "
                f"style-neutral RankIC={dev['dev_neu_ric']:.4f}, factors={dev['n_factors']}\n"
                "correlation of composite prediction with styles: "
                + ", ".join(f"{k}={dev[f'style_{k}']:+.2f}" for k in bt.STYLES))
        prev = self.read("exp_log.md", 1500)
        top = tab.reindex(tab["dev_ric"].abs().sort_values(ascending=False).index).head(12)
        dec = tab.sort_values("decay", ascending=False).head(8)
        tmpl = DIAGNOSE if not self.cert else DIAGNOSE.replace("TRAIN 2011-2019 and VALIDATION 2020",
            "TRAIN 2011-2018 and VALIDATION 2019").replace("validation 2020", "validation 2019")
        p = tmpl.format(composite=comp, themes=th.to_string(index=False),
                            top=top[cols].round(4).to_string(index=False),
                            decayed=dec[cols + ["train_ric"]].round(4).to_string(index=False), previous=prev)
        txt = self.ask("diagnose", p, temperature=0.4) or "(diagnosis unavailable)"
        self.write("insights.md", f"# Diagnosis before round {self.state['round'] + 1}\n{txt}")
        self.write("insights_history.md", f"\n# Before round {self.state['round'] + 1}\n{txt}", append=True)
        return th

    # ---------- 总指挥 ----------
    def plan(self, th):
        a = self.a
        hist = "\n".join(f"round {i}: RankIC={d['dev_ric']:.4f}, RankICIR={d['dev_icir']:.3f}, "
                         f"neutral RankIC={d['dev_neu_ric']:.4f}, factors={d['n_factors']}"
                         for i, d in enumerate(self.state["dev"]))
        tmpl = PLAN if not self.cert else PLAN.replace(
            "AND do not make any single year of 2018, 2019, 2020 worse, so signals that only work in one year are rejected;",
            "AND do not make 2018 or 2019 worse, AND the whole batch then passes a one-shot statistical confirmation\n"
            "  on data you never see (you are told only pass/fail);")
        p = tmpl.format(map=self.read("map.md"), insights=self.read("insights.md", 2500),
                        themes=th[["agent", "theme", "in_pool", "mean|dev_RankIC|", "sign_kept",
                                   "failed_this_run"]].to_string(index=False),
                        history=hist, budget=max(0, self.budget_end - self.calls),
                        max_tasks=a.max_tasks, max_gens=a.max_gens, dup=a.dup)
        txt = self.ask("plan", p, temperature=0.4)
        plan = None
        m = re.search(r"\{.*\}", txt or "", re.S)
        if m:
            try:
                plan = json.loads(m.group(0))
            except json.JSONDecodeError:
                plan = None
        tasks = []
        for t in (plan or {}).get("tasks", []):
            if t.get("agent") in AGENTS and t.get("mode") in ("new", "evolve"):
                tasks.append({"agent": t["agent"], "mode": t["mode"], "focus": str(t.get("focus", ""))[:600],
                              "gens": int(min(max(int(t.get("gens", 1)), 1), a.max_gens))})
        if not tasks:                                    # 总指挥没给出有效计划：按验证期表现兜底
            log("  总指挥的计划无法解析，使用兜底计划")
            best = th.dropna(subset=["mean|dev_RankIC|"]).sort_values("mean|dev_RankIC|", ascending=False)
            tasks = [{"agent": g, "mode": "evolve", "focus": "Improve stability and reduce style exposure.",
                      "gens": a.max_gens} for g in best["agent"].head(a.max_tasks - 1)]
            fresh = th.sort_values("in_pool")["agent"].iloc[0]
            tasks.append({"agent": fresh, "mode": "new", "focus": "Explore new low-style-exposure signals.",
                          "gens": a.max_gens})
            plan = {"rationale": "fallback plan", "stop": False}
        plan["tasks"] = tasks[: a.max_tasks]
        r = self.state["round"] + 1
        self.write("plan.md", f"\n# Round {r}\nRationale: {plan.get('rationale', '')}\n"
                   + "\n".join(f"- {t['agent']} | {t['mode']} | gens={t['gens']} | {t['focus']}" for t in plan["tasks"]),
                   append=True)
        return plan

    # ---------- 设计专家：调度主题子代理 ----------
    def run_task(self, t):
        a = self.a
        agent, mode = t["agent"], t["mode"]
        self.focus = t["focus"]
        seeds = [f for f in self.pool_f if f.get("agent") == agent]
        alive = list(seeds) if mode == "evolve" else []
        n0 = a.n0 if mode == "new" or not seeds else max(2, a.n0 // 2)
        new = self.score_batch(self.generate(agent, 0, n0, alive), agent, 0)
        alive += new
        q, e = self.select(new)
        elites = list(e)
        if mode == "evolve":                               # 进化模式：池中已有因子也可当父代
            q = q + self.rank(seeds)[: a.parents]
        for gen in range(1, t["gens"] + 1):
            if self.over():
                log("  本轮预算用完，停止当前任务")
                break
            carried = self.rank(elites)[:2]
            parents = carried + [x for x in self.rank(q) if x not in carried]
            parents += [x for x in self.rank(alive) if x not in parents]
            parents = parents[: a.parents]
            if len(parents) < 2:
                break
            cands = self.evolve_children(agent, gen, parents)
            if gen % 2 == 0:
                cands += self.generate(agent, gen, a.inject, alive)
            new = self.score_batch(cands, agent, gen)
            alive += new
            q, e = self.select(new + carried)
            elites += [x for x in e if x not in elites]
        return elites

    # ---------- 评估专家：按组合贡献入池、剔除拖后腿的老因子 ----------
    def accept_and_prune(self, added, r):
        a = self.a
        keep = lambda e: {k: v for k, v in e.items() if k in ("name", "code", "agent", "gen", "train", "round")}
        self.ev.compute(self.pool_f + added)
        names = [f["name"] for f in self.pool_f]
        base, by = self.ev.sel_score(names)
        yr = lambda d: "/".join(f"{v:.3f}" for _, v in sorted(d.items()))
        log(f"  入池检查：当前组合验证分数 {base:.4f}，分年 {yr(by)}（{"/".join(str(y) for y in sorted(by))}，合成与中性化 RankIC 的平均）")
        if self.cert and a.prescreen == "pooled":       # 可认证模式：初筛可以大胆，安全由确认检验保证
            robust = lambda sc, d: sc > base + a.min_gain
        else:
            robust = lambda sc, d: sc > base + a.min_gain and all(d[y] >= by[y] for y in by)
        kept = []
        for e in sorted(added, key=lambda x: -abs(x["train"]["RankICIR"])):
            if e["name"] not in self.ev.vals:
                continue
            sc, d = self.ev.sel_score(names + [e["name"]])
            ok = robust(sc, d)
            log(f"    {'保留' if ok else '不要'} {e['name']}：{base:.4f} → {sc:.4f}，分年 {yr(by)} → {yr(d)}")
            if ok:
                names.append(e["name"])
                base, by = sc, d
                kept.append(e)
            else:
                self.sigs.pop(e["name"], None)
        cert = None
        if self.cert and kept:                          # 第二步：确认窗口上的一次性置信下界检验（整批）
            M = len(CONF_METRICS)
            alpha_r = a.alpha * 2.0 ** (-r) / (M * 2.0 ** (r - 1))
            old = [f["name"] for f in self.pool_f]
            cert = self.ev.confirm(old, old + [e["name"] for e in kept], alpha_r, a.boot, a.block, a.seed + r)
            log(f"  确认检验（2020，{cert['days']} 天，α'={alpha_r:.5f}，z={cert['z']:.2f}）："
                + "，".join(f"{k} 增益 {cert['gain_' + k]:+.4f} 下界 {cert['lcb_' + k]:+.4f}" for k in CONF_METRICS)
                + f" → {'通过，整批入池' if cert['passed'] else '未通过，整批放弃'}")
            if not cert["passed"]:
                for e in kept:
                    self.sigs.pop(e["name"], None)
                kept = []
        if self.cert:
            self.state["cert"].append({"round": r, "batch": len(kept) if cert is None or cert["passed"] else 0,
                                       **(cert or {"passed": False, "empty": True})})
            (self.hidden / "confirm.json").write_text(json.dumps(self.state["cert"], indent=1))
        self.pool_f += [keep(e) for e in kept]
        # 剔除：只考虑训练期到验证期方向反了或明显变弱的老因子，每轮最多剔除 --max_prune 个
        tab = self.ev.factor_table(self.pool_f)
        flip = tab[(np.sign(tab["train_ric"]) != np.sign(tab["dev_ric"])) | (tab["dev_ric"].abs() < 0.005)]
        pruned = []
        for n in flip.sort_values("dev_ric", key=abs)["name"]:
            if len(pruned) >= a.max_prune:
                break
            rest = [x for x in names if x != n]
            sc, d = self.ev.sel_score(rest)
            if robust(sc, d):
                log(f"    剔除 {n}：{base:.4f} → {sc:.4f}")
                names, base, by = rest, sc, d
                pruned.append(n)
        self.pool_f = [f for f in self.pool_f if f["name"] not in pruned]
        for n in pruned:
            self.sigs.pop(n, None)
        conf_note = "" if not self.cert else (" | held-out confirmation: " + ("not run (no candidates)" if cert is None
                     else "PASSED" if cert["passed"] else "FAILED (batch discarded)"))   # 只告诉大模型通过与否（假设 A1）
        self.write("exp_log.md", f"- round {r} | acceptance: kept {len(kept)}/{len(added)} new factors by composite "
                   f"contribution; pruned {len(pruned)} old factors: " + ", ".join(pruned[:10]) + conf_note, append=True)
        return kept, pruned

    # ---------- RSI ----------
    def span(self, k, base):
        s0, tgt = base[k], self.targets[k]
        if tgt <= s0:                                   # 起点已超过目标：目标上调到起点的 1.5 倍
            tgt = s0 + max(abs(s0) * 0.5, 1e-3)
        return tgt - s0

    def rsi(self, off, base, keys=None):
        gains = {k: (off[k] - base[k]) / self.span(k, base) for k in (keys or METRICS)}
        return float(np.mean(list(gains.values()))), gains

    def rsi_lcb(self, upto):
        """定理 (ii)：真实 RSI（3 项可认证指标）的下界 = 已入池各轮置信下界之和，按 RSI 归一化后取平均。"""
        base = self.state["official"][0]
        acc = {k: 0.0 for k in CONF_METRICS}
        for c in self.state.get("cert", []):
            if c["round"] <= upto and c.get("passed"):
                for k in CONF_METRICS:
                    acc[k] += c["lcb_" + k]
        return float(np.mean([acc[k] / self.span(CONF_METRICS[k], base) for k in CONF_METRICS]))

    # ---------- 主循环 ----------
    def run(self):
        a = self.a
        if not self.state["dev"]:
            log("===== 第 0 轮：评估起点因子池 =====")
            self.ev.compute(self.pool_f)
            dev, off = self.ev.evaluate([f["name"] for f in self.pool_f])
            self.state["dev"].append(dev)
            self.state["official"].append(off)
            self.state["calls"].append(0)
            self.state["minutes"].append(0)
            self._save("state.json", self.state)
            self.write("exp_log.md", f"# Experiment log\n## Round 0 (seed pool)\nvalidation: "
                       f"RankIC={dev['dev_ric']:.4f}, RankICIR={dev['dev_icir']:.3f}, "
                       f"neutral RankIC={dev['dev_neu_ric']:.4f}, factors={dev['n_factors']}")
            (self.hidden / "official.json").write_text(json.dumps(self.state["official"], indent=1))
            self.update_map()
        for r in range(self.state["round"] + 1, a.rounds + 1):
            t0, c0 = time.time(), self.calls
            self.budget_end = self.calls + a.budget
            log(f"===== 第 {r} 轮开始（本轮预算 {a.budget} 次调用） =====")
            self.ev.compute(self.pool_f)
            tab = self.ev.factor_table(self.pool_f)
            log("  诊断专家分析因子池……")
            th = self.diagnose(tab, self.state["dev"][-1])
            log("  总指挥制定计划……")
            plan = self.plan(th)
            for t in plan["tasks"]:
                log(f"  计划：{t['agent']} | {t['mode']} | {t['gens']} 代 | {t['focus'][:80]}")
            if plan.get("stop") and r > 1:
                log("  总指挥判断无需继续，提前结束")
                break
            added = []
            for t in plan["tasks"]:
                if self.over():
                    log("  本轮预算用完，跳过剩余任务")
                    break
                log(f"  --- 任务：{t['agent']}（{t['mode']}）---")
                es = self.run_task(t)
                fresh = [e for e in es if e["name"] not in {f["name"] for f in self.pool_f + added}]
                for e in fresh:
                    e.update({"round": r, "elite": True})
                    e.pop("_sig", None)
                added += fresh
                self.write("exp_log.md", f"- round {r} | {t['agent']} | {t['mode']} | new elites: {len(fresh)} | "
                           + ", ".join(f"{e['name']} (train RankICIR {e['train']['RankICIR']:+.3f})" for e in fresh[:6]),
                           append=True)
                log(f"  任务完成：新精英 {len(fresh)} 个，累计调用 {self.calls} 次")
            kept, pruned = self.accept_and_prune(added, r)
            log(f"  入池检查：候选 {len(added)} 个，按组合贡献保留 {len(kept)} 个；剔除老因子 {len(pruned)} 个")
            log(f"  评估专家：计算因子池（{len(self.pool_f)} 个）的验证期与测试期指标……")
            self.ev.compute(self.pool_f)
            dev, off = self.ev.evaluate([f["name"] for f in self.pool_f])
            self.state["dev"].append(dev)
            self.state["official"].append(off)
            self.state["calls"].append(self.calls - c0)
            self.state["minutes"].append(round((time.time() - t0) / 60, 1))
            self.state["round"] = r
            self.write("exp_log.md", f"## Round {r} result (validation {self.ev.dev[0][:4]}): RankIC={dev['dev_ric']:.4f}, "
                       f"RankICIR={dev['dev_icir']:.3f}, neutral RankIC={dev['dev_neu_ric']:.4f}, "
                       f"factors={dev['n_factors']} (+{len(kept)}, -{len(pruned)})", append=True)
            (self.hidden / "official.json").write_text(json.dumps(self.state["official"], indent=1))
            self._save("pool.json", self.pool_f)
            self._save("history.json", self.history)
            self._save("state.json", self.state)
            self.update_map()
            score, _ = self.rsi(off, self.state["official"][0])
            extra = ""
            if self.cert:
                s3, _ = self.rsi(off, self.state["official"][0], list(CONF_METRICS.values()))
                extra = f"；3 项可认证指标：实际 RSI {s3:+.4f}，理论下界 {self.rsi_lcb(r):+.4f}"
            log(f"===== 第 {r} 轮结束：新增 {len(kept)} 个、剔除 {len(pruned)} 个因子，用时 {self.state['minutes'][-1]} 分钟，"
                f"调用 {self.state['calls'][-1]} 次，RSI 指数 {score:+.4f}{extra} =====")
        self.report()

    def report(self):
        base = self.state["official"][0]
        rows = []
        for r, (off, dev) in enumerate(zip(self.state["official"], self.state["dev"])):
            score, gains = self.rsi(off, base)
            prev = self.state["official"][r - 1] if r else off
            step, _ = self.rsi(off, prev) if r else (0.0, None)
            rows.append({"轮次": r, "因子数": off["n_factors"],
                         **{METRICS[k][0].split("：")[0]: round(off[k], 4) for k in METRICS},
                         **{"增益_" + METRICS[k][0].split("：")[0]: round(g, 3) for k, g in gains.items()},
                         "RSI指数(相对第0轮)": round(score, 4), "本轮RSI(相对上一轮)": round(step, 4),
                         **({"RSI3实际": round(self.rsi(off, base, list(CONF_METRICS.values()))[0], 4),
                             "RSI3下界": round(self.rsi_lcb(r), 4)} if self.cert else {}),
                         "验证期RankIC": round(dev["dev_ric"], 4),
                         "调用次数": self.state["calls"][r], "用时(分)": self.state["minutes"][r],
                         # 成本维度（对应 Web Economics 的大模型成本）：每 100 次调用带来的 RSI 增益
                         "累计调用": sum(self.state["calls"][: r + 1]),
                         "RSI每百次调用": round(score / sum(self.state["calls"][: r + 1]) * 100, 5)
                         if sum(self.state["calls"][: r + 1]) else 0.0})
        df = pd.DataFrame(rows)
        df.to_csv(self.out / "rsi.csv", index=False, encoding="utf-8-sig")
        pd.set_option("display.width", 260)
        print("\n================ RSI 结果（测试期 2021-2024 与 2025 至今，大模型从未见过） ================", flush=True)
        print("目标值：" + "，".join(f"{METRICS[k][0]} = {v}" for k, v in self.targets.items()), flush=True)
        print(df.to_string(index=False), flush=True)
        if self.cert:
            print("可认证模式：RSI3 只含测试期 RankIC、RankICIR、中性化 RankIC 三项；定理保证（假设成立时，"
                  f"概率 ≥ {1 - self.a.alpha:.0%}）RSI3实际 逐轮不降且 ≥ RSI3下界。确认检验明细见 hidden/confirm.json。", flush=True)
        print("\n说明：增益 =（本轮值 − 第 0 轮值）/（目标值 − 第 0 轮值）；RSI 指数 = 四项增益的平均。"
              "\n“本轮RSI”以上一轮为起点计算，用来看每多一轮的边际进步是加速还是放缓。", flush=True)
        log(f"结果保存在 {self.out}/（rsi.csv、pool.json、workspace/、hidden/）")


# ============================ 本地测试用：模拟大模型 ============================
_MOCK = [0]


def mock_chat(prompt, temperature=None, **kw):
    if "Answer with exactly one line per factor" in prompt:
        return "\n".join(f"{i + 1}: PASS - none" for i in range(prompt.count("### Factor")))
    if "Return ONLY a JSON object" in prompt:
        return json.dumps({"rationale": "mock plan", "stop": False, "tasks": [
            {"agent": "AgentReversal", "mode": "new", "focus": "short-term reversal scaled by volatility", "gens": 1},
            {"agent": "AgentLiquidity", "mode": "evolve", "focus": "liquidity with low size exposure", "gens": 1}]})
    if "Write the diagnosis of the current factor pool" in prompt:
        return "## What works\n- mock\n## What fails\n- mock\n## Unexplored gaps\n- mock\n## Recommendations\n- mock"
    tpl = ["-df['close'].pct_change({w}) / (df['close'].pct_change().rolling({w}).std() + 1e-9)",
           "np.log(df['volume'] + 1).rolling({w}).mean() - np.log(df['volume'] + 1).rolling(5).mean()",
           "((df['high'] - df['low']) / (df['close'] + 1e-9)).rolling({w}).mean()",
           "df['close'].pct_change().rolling({w}).corr(np.log(df['volume'] + 1).diff())",
           "(df['close'] - df['low'].rolling({w}).min()) / (df['high'].rolling({w}).max() - df['low'].rolling({w}).min() + 1e-9)"]
    out = []
    for j in range(3):
        _MOCK[0] += 1
        w = random.choice([10, 20, 30, 40, 60])
        body = random.choice(tpl).format(w=w)
        out.append(f"<<function {j + 1}>>\ndef factor_mock_{_MOCK[0]}_{w}(df):\n"
                   f"    \"\"\"Mock factor for pipeline tests.\"\"\"\n"
                   f"    return {body}\n<</function {j + 1}>>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="orch1", help="结果目录名 runs/<run>")
    ap.add_argument("--seed_pool", default="runs/cogalpha/elites.json", help="起点因子池（上一版的精英因子）")
    ap.add_argument("--rounds", type=int, default=1, help="跑几轮（建议先 1 轮，再续跑到 3 轮）")
    ap.add_argument("--budget", type=int, default=400, help="每轮最多调用大模型的次数")
    ap.add_argument("--max_tasks", type=int, default=5, help="总指挥每轮最多派几个任务")
    ap.add_argument("--max_gens", type=int, default=4, help="每个任务最多进化几代")
    ap.add_argument("--targets", default=",".join(str(v[1]) for v in METRICS.values()),
                    help="RSI 四项目标值：测试期RankIC,RankICIR,中性化RankIC,2025至今RankIC")
    ap.add_argument("--dup", type=float, default=0.7, help="去重门槛：与已有因子截面相关超过它视为重复")
    ap.add_argument("--min_gain", type=float, default=0.0003, help="入池 / 剔除要求的最小验证分数提升")
    ap.add_argument("--max_prune", type=int, default=0, help="每轮最多剔除几个老因子（默认 0：不剔除）")
    ap.add_argument("--admit", choices=["robust", "certified"], default="robust",
                    help="入池规则：robust = orch3 的逐年规则；certified = 逐年初筛 + 2020 确认窗口上的置信下界检验")
    ap.add_argument("--prescreen", choices=["pooled", "yearwise"], default="pooled",
                    help="可认证模式下第一步初筛：pooled = 只要求总分提升；yearwise = 还要求每一年不变差")
    ap.add_argument("--alpha", type=float, default=0.1, help="可认证模式的总错误概率 α")
    ap.add_argument("--boot", type=int, default=2000, help="分块 bootstrap 次数")
    ap.add_argument("--block", type=int, default=20, help="分块 bootstrap 的块长（交易日）")
    ap.add_argument("--n0", type=int, default=8)
    ap.add_argument("--parents", type=int, default=6)
    ap.add_argument("--inject", type=int, default=3)
    ap.add_argument("--min_gens", type=int, default=99)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--cache", default="./csi300_panel.pkl", help="训练期面板缓存（与 cogalpha.py 相同）")
    ap.add_argument("--full_cache", default="./csi300_panel_full.pkl", help="完整面板缓存（与 backtest.py 相同）")
    ap.add_argument("--mock", action="store_true", help="用模拟大模型测试流程，不花钱")
    a = ap.parse_args()
    a.agents = "all"
    if a.mock:
        cogalpha.chat = mock_chat
    load_panel(a.data, cache=a.cache)
    Orchestra(a).run()


if __name__ == "__main__":
    main()
