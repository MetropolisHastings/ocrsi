"""CogAlpha 低成本复现版：21 个 Agent × 多代进化。

保留论文的全部机制：七层 21 个 Agent、五种引导风格、批量生成、代码修复、逻辑评审与改进、
变异 / 交叉 / 交叉后变异三种算子、65/80 分位筛选、精英保留、每 2 代注入新因子（带好坏反馈）、早停。
为降低成本做的改动（均可用参数调回论文规模）：
  - 规模：父代 8（论文 32）、子代 24（论文 96）、每个 Agent 最多 8 代（论文 24）；
  - 一次调用产出 3 个子代、一次调用评审一批因子；
  - 先做免费的规则检查和打分，只把达到最低门槛的因子送去大模型评审；
  - 与已有因子截面相关性 > 0.9 视为重复（避免一堆同质变种）。
筛选只用训练期 2011-2019；2020 留作合成时的验证集；2021 之后只在最终回测里看。

用法（服务器后台）：
  nohup python -u cogalpha.py --data ../qlib_bin > cogalpha.log 2>&1 &
中断后重新运行同一条命令会跳过已完成的 Agent。
"""
import argparse
import json
import multiprocessing as mp
import random
import re
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import prompts
from agents import AGENTS, GUIDANCE
from checker import compile_factor, run_check
from data import load_panel, make_label
from llm import chat, parse_functions
from metrics import evaluate
from runner import apply_factor

TRAIN = ("2011-01-01", "2019-12-31")
KEYS = ["IC", "RankIC", "ICIR", "RankICIR", "MI"]
IC_KEYS = ["IC", "RankIC", "ICIR", "RankICIR"]
MIN_Q = {"IC": 0.005, "RankIC": 0.005, "ICIR": 0.05, "RankICIR": 0.05}    # 论文合格门槛
MIN_E = {"IC": 0.01, "RankIC": 0.01, "ICIR": 0.1, "RankICIR": 0.1}        # 论文精英门槛
# 说明：MI 参与分位数排序，但不设绝对门槛（我们的 MI 估计方法与论文不同，数值不可直接比较）


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


# ====================== 多进程打分（子进程共享面板数据） ======================
_P = _L = _SAMPLE = None


def _init_worker(data, cache):
    global _P, _L, _SAMPLE
    _P = load_panel(data, cache=cache)
    _L = make_label(_P)
    dt = _P.index.get_level_values("datetime")
    days = pd.DatetimeIndex(sorted(set(dt[(dt >= TRAIN[0]) & (dt <= TRAIN[1])])))
    pick = days[np.linspace(0, len(days) - 1, 60).astype(int)]   # 60 个固定日期用于去重比较
    _SAMPLE = _P.index[np.isin(dt, pick) & _P["in_index"].values]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def eval_code(code, limit=180):
    """规则检查 + 训练期打分。返回 dict：成功时含 name/train/sig，失败时含 error。
    单个因子超过 limit 秒直接淘汰，防止一个极慢的因子卡住整批。"""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit)
    try:
        return _eval_code(code)
    except _Timeout:
        return {"name": compile_factor(code)[0], "error": f"计算太慢（超过 {limit} 秒）"}
    finally:
        signal.alarm(0)


def _eval_code(code):
    try:
        name, func, err = compile_factor(code)
        if err is None:
            err = run_check(func, _P)
        if err is not None:
            return {"name": name, "error": err}
        f = apply_factor(func, _P)
        dt = f.index.get_level_values("datetime")
        m = (dt >= TRAIN[0]) & (dt <= TRAIN[1])
        tr = evaluate(f[m], _L.reindex(f.index)[m])
        if tr["NaN率"] > 0.30:
            return {"name": name, "error": f"NaN 比例 {tr['NaN率']:.0%} 超过 30%"}
        if not np.isfinite(tr["RankIC"]):
            return {"name": name, "error": "因子几乎为常数，无法计算 IC"}
        s = f.reindex(_SAMPLE)
        sig = s.groupby(level="datetime").rank(pct=True).fillna(0.5).astype("float32").values
        return {"name": name, "train": {k: float(v) for k, v in tr.items()}, "sig": sig}
    except _Timeout:
        raise
    except Exception as e:  # 防御：任何意外都不能让主程序崩溃
        return {"name": None, "error": f"打分出错: {type(e).__name__}: {e}"}


# ====================== 主流程 ======================
class CogAlpha:
    def __init__(self, a):
        self.a = a
        self.out = Path("runs") / a.run
        (self.out / "llm").mkdir(parents=True, exist_ok=True)
        self.pool = mp.get_context("fork").Pool(a.procs, _init_worker, (a.data, a.cache))
        self.calls = 0
        self.elites = self._load("elites.json", [])          # 全局精英池（所有 Agent）
        self.sigs = {}                                         # 名字 -> 去重用的截面排名向量
        self.rng = random.Random(a.seed)

    # ---------- 工具 ----------
    def _load(self, name, default):
        p = self.out / name
        return json.loads(p.read_text()) if p.exists() else default

    def _save(self, name, obj):
        (self.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))

    def ask(self, tag, prompt, temperature=None):
        self.calls += 1
        for attempt in range(3):
            try:
                text = chat(prompt, temperature=temperature)
                break
            except Exception as e:                          # 网络抖动：重试
                log(f"    大模型调用失败（{e}），{5 * (attempt + 1)} 秒后重试")
                time.sleep(5 * (attempt + 1))
        else:
            return ""
        safe = re.sub(r"[^\w]+", "_", tag)[:80]
        (self.out / "llm" / f"{safe}_prompt.txt").write_text(prompt, encoding="utf-8")
        (self.out / "llm" / f"{safe}_response.txt").write_text(text, encoding="utf-8")
        return text

    def pmap(self, fn, items):
        with ThreadPoolExecutor(self.a.workers) as ex:
            return list(ex.map(fn, items))

    @staticmethod
    def brief(f):
        t = f["train"]
        return f"{f['name']}: RankIC={t['RankIC']:+.4f}, RankICIR={t['RankICIR']:+.3f}, MI={t['MI']:.4f}"

    # ---------- 质检与打分 ----------
    def score_batch(self, cands, agent, gen):
        """cands: [(code, origin)]。返回通过的记录列表；全部记录写入 history。"""
        if not cands:
            return []
        res = self.pool.map(eval_code, [c for c, _ in cands])
        # 1) 代码修复：运行出错的交给修复 Agent，修一次再测
        broken = [i for i, r in enumerate(res) if r.get("error") and not r["error"].startswith("NaN")]
        if broken:
            fixed = self.pmap(lambda i: parse_functions(self.ask(
                f"{agent}_g{gen}_repair{i}", prompts.CODE_REPAIR.format(error=res[i]["error"], code=cands[i][0]),
                temperature=0.8))[:1], broken)
            redo = [(i, f[0]) for i, f in zip(broken, fixed) if f]
            for (i, code), r in zip(redo, self.pool.map(eval_code, [c for _, c in redo])):
                cands[i] = (code, cands[i][1] + "（已修复）")
                res[i] = r
        # 2) 去重 + 最低门槛（免费）
        recs = []
        for (code, origin), r in zip(cands, res):
            rec = {"agent": agent, "gen": gen, "origin": origin, "name": r.get("name"), "code": code}
            if r.get("error"):
                rec["error"] = r["error"]
            else:
                rec["train"] = r["train"]
                rec["_sig"] = r["sig"]
                if not all(abs(rec["train"][k]) >= v for k, v in MIN_Q.items()):
                    rec["error"] = "未达到最低门槛"
                else:
                    dup = self.duplicate_of(rec)
                    if dup:
                        rec["error"] = f"与 {dup} 高度相关（>0.9），视为重复"
                    else:
                        self.sigs[rec["name"]] = rec["_sig"]       # 先登记，防止同批次重复
            recs.append(rec)
        # 3) 逻辑评审（只评审过了门槛的），不通过的改进一次后重新打分
        ok = [r for r in recs if not r.get("error")]
        verdicts = {}
        for chunk in [ok[i:i + 6] for i in range(0, len(ok), 6)]:
            text = "\n\n".join(f"### Factor {j + 1}\n{r['code']}" for j, r in enumerate(chunk))
            ans = self.ask(f"{agent}_g{gen}_judge_{chunk[0]['name']}", prompts.JUDGE_BATCH.format(factors=text),
                           temperature=0.8)
            for j, r in enumerate(chunk):
                m = re.search(rf"^\s*{j + 1}\s*[:：]\s*(PASS|FAIL)\s*[-–—:]?\s*(.*)$", ans, re.M | re.I)
                verdicts[id(r)] = (m.group(1).upper(), m.group(2).strip()) if m else ("PASS", "未解析到评审结果")
        failed = [r for r in ok if verdicts[id(r)][0] == "FAIL"]
        improved = self.pmap(lambda r: parse_functions(self.ask(
            f"{agent}_g{gen}_improve_{r['name']}", prompts.LOGIC_IMPROVE.format(
                issues=verdicts[id(r)][1], code=r["code"]), temperature=0.8))[:1], failed)
        redo = [(r, f[0]) for r, f in zip(failed, improved) if f]
        for r in failed:
            r["error"] = f"逻辑评审未通过：{verdicts[id(r)][1][:120]}"
            self.sigs.pop(r["name"], None)
        for (r, code), new in zip(redo, self.pool.map(eval_code, [c for _, c in redo])):
            nr = {"agent": agent, "gen": gen, "origin": r["origin"] + "（按评审改进）",
                  "name": new.get("name"), "code": code}
            if new.get("error"):
                nr["error"] = "改进后：" + new["error"]
            else:
                nr["train"], nr["_sig"] = new["train"], new["sig"]
                dup = None
                if not all(abs(nr["train"][k]) >= v for k, v in MIN_Q.items()):
                    nr["error"] = "改进后未达到最低门槛"
                elif (dup := self.duplicate_of(nr)):
                    nr["error"] = f"改进后与 {dup} 重复"
                else:
                    self.sigs[nr["name"]] = nr["_sig"]
            recs.append(nr)
        # 4) 登记
        passed = []
        for r in recs:
            r.pop("_sig", None)
            if not r.get("error"):
                passed.append(r)
            status = "通过" if not r.get("error") else "淘汰 - " + r["error"]
            extra = f"  RankICIR={r['train']['RankICIR']:+.3f}" if r.get("train") and not r.get("error") else ""
            log(f"    [{r['origin'][:38]}] {r['name']}: {status}{extra}")
        self.history += recs
        return passed

    def duplicate_of(self, rec):
        if rec["name"] in self.sigs:
            return rec["name"]
        v = rec["_sig"]
        for name, s in self.sigs.items():
            if abs(np.corrcoef(v, s)[0, 1]) > 0.9:
                return name
        return None

    # ---------- 筛选（论文 65/80 分位规则） ----------
    def select(self, cands):
        """合格：综合分位 ≥ 65%；精英：综合分位 ≥ 80% 且过精英门槛。
        综合分位 = 5 项指标（取绝对值）各自分位数的平均。论文要求 5 项“同时”过分位，
        在每代只有二三十个候选的小规模下几乎没有因子能满足，所以改为平均分位。"""
        if not cands:
            return [], []
        m = pd.DataFrame([{k: abs(c["train"][k]) for k in KEYS} for c in cands])
        score = m.rank(pct=True).mean(axis=1)
        qualified, elite = [], []
        for c, (i, row) in zip(cands, m.iterrows()):
            if score[i] >= score.quantile(0.65):
                qualified.append(c)
            if score[i] >= score.quantile(0.80) and all(row[k] >= v for k, v in MIN_E.items()):
                elite.append(c)
        return qualified, elite

    @staticmethod
    def rank(fs):
        if not fs:
            return []
        m = pd.DataFrame([{k: abs(f["train"][k]) for k in KEYS} for f in fs])
        order = m.rank(pct=True).mean(axis=1).sort_values(ascending=False).index
        return [fs[i] for i in order]

    # ---------- 生成 ----------
    def generate(self, agent, gen, n, alive):
        level, theme, ideas = AGENTS[agent]
        styles = list(GUIDANCE)
        r = self.rank(alive)
        eff = "\n".join(f"- {self.brief(f)} -> worked; reuse its core idea." for f in r[:2]) or "(no history yet)"
        bad = [h for h in self.history if h["agent"] == agent and h.get("error")][-2:]
        ineff = "\n".join(f"- {h['name']}: {h['error'][:100]} -> avoid this pattern." for h in bad) or "(no history yet)"
        existing = ", ".join(f["name"] for f in alive) or "(none yet)"
        half = [n - n // 2, n // 2] if n > 1 else [n]
        def one(k):
            style = styles[(gen * 2 + k) % len(styles)]
            p = prompts.BASE_AGENT.format(theme=theme, n=half[k], existing=existing, effective=eff,
                                          ineffective=ineff, ideas=ideas + f"\nGuidance style: {GUIDANCE[style]}")
            return [(c, f"生成[{style}]") for c in parse_functions(self.ask(f"{agent}_g{gen}_gen{k}", p))[:half[k]]]
        return sum(self.pmap(one, range(len(half))), [])

    def evolve_children(self, agent, gen, parents):
        def one(i):
            a = parents[i]
            b = self.rng.choice([p for p in parents if p is not a] or [a])
            txt = self.ask(f"{agent}_g{gen}_evo_{a['name']}", prompts.EVOLVE_BATCH.format(
                metrics_a=self.brief(a), code_a=a["code"], metrics_b=self.brief(b), code_b=b["code"]))
            fs = parse_functions(txt)[:3]
            tags = [f"变异 {a['name']}", f"交叉 {a['name']}×{b['name']}", f"交叉+变异 {a['name']}×{b['name']}"]
            return list(zip(fs, tags))
        return sum(self.pmap(one, range(len(parents))), [])

    # ---------- 单个 Agent 的进化 ----------
    def run_agent(self, agent):
        a = self.a
        log(f"===== {agent}（第 {AGENTS[agent][0]} 层）开始 =====")
        alive, agent_elites, best_hist = [], [], []
        new = self.score_batch(self.generate(agent, 0, a.n0, alive), agent, 0)
        alive += new
        q, e = self.select(new)
        agent_elites += e
        for gen in range(1, a.gens + 1):
            carried = self.rank(agent_elites)[:2]                       # 论文：前 2 名精英直接进入下一代
            parents = carried + [x for x in self.rank(q) if x not in carried]
            parents += [x for x in self.rank(alive) if x not in parents]
            parents = parents[: a.parents]
            if len(parents) < 2:
                log(f"  {agent}：可用父代不足，提前结束")
                break
            log(f"  第 {gen} 代：父代 {len(parents)} 个，最好的是 {self.brief(parents[0])}")
            cands = self.evolve_children(agent, gen, parents)
            if gen % 2 == 0:                                            # 论文：每 2 代注入新因子
                cands += self.generate(agent, gen, a.inject, alive)
            new = self.score_batch(cands, agent, gen)
            alive += new
            q, e = self.select(new + carried)
            agent_elites += [x for x in e if x not in agent_elites]
            best = max([abs(x["train"]["RankICIR"]) for x in agent_elites] or [0])
            best_hist.append(best)
            log(f"  第 {gen} 代结束：新通过 {len(new)} 个，本 Agent 精英 {len(agent_elites)} 个，"
                f"最佳 |RankICIR|={best:.3f}，累计调用 {self.calls} 次")
            # 论文：提升 ≤ 0.001 时早停（这里看最近两代，且至少跑 min_gens 代）
            if gen >= a.min_gens and len(best_hist) >= 3 and best_hist[-1] - best_hist[-3] <= 0.001:
                log(f"  {agent}：最近两代没有提升，早停")
                break
        for x in agent_elites:
            x["elite"] = True
        self.elites += [{k: v for k, v in x.items()} for x in agent_elites]
        return alive

    def run(self):
        done = self._load("done.json", [])
        self.history = self._load("history.json", [])
        # 恢复已完成 Agent 的去重信息
        if self.elites:
            res = self.pool.map(eval_code, [e["code"] for e in self.elites])
            for e, r in zip(self.elites, res):
                if "sig" in r:
                    self.sigs[e["name"]] = r["sig"]
        names = list(AGENTS) if self.a.agents == "all" else self.a.agents.split(",")
        t0 = time.time()
        for ag in names:
            if ag in done:
                log(f"{ag} 已完成，跳过")
                continue
            t = time.time()
            self.run_agent(ag)
            done.append(ag)
            self._save("elites.json", self.elites)
            self._save("history.json", self.history)
            self._save("done.json", done)
            log(f"===== {ag} 完成，用时 {(time.time() - t) / 60:.1f} 分钟；全局精英 {len(self.elites)} 个 =====")
        self.report()
        log(f"全部完成：用时 {(time.time() - t0) / 60:.1f} 分钟，本次调用大模型 {self.calls} 次")

    def report(self):
        rows = [{"Agent": e["agent"], "代": e["gen"], "因子": e["name"], **{k: e["train"][k] for k in KEYS}}
                for e in self.elites]
        if not rows:
            log("没有精英因子")
            return
        df = pd.DataFrame(rows)
        df.to_csv(self.out / "elites.csv", index=False, encoding="utf-8-sig")
        h = pd.DataFrame(self.history)
        summ = h.groupby("agent").agg(候选=("name", "size"),
                                      通过=("error", lambda s: int(s.isna().sum())))
        summ["精英"] = df.groupby("Agent").size()
        pd.set_option("display.width", 220)
        print("\n各 Agent 汇总：\n" + summ.fillna(0).astype(int).to_string(), flush=True)
        print("\n训练期 |RankICIR| 前 15 的精英因子：", flush=True)
        df["a"] = df["RankICIR"].abs()
        print(df.sort_values("a", ascending=False).drop(columns="a").head(15).round(4).to_string(index=False),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="all", help="all 或逗号分隔的 Agent 名")
    ap.add_argument("--run", default="cogalpha", help="结果目录名 runs/<run>")
    ap.add_argument("--n0", type=int, default=12, help="每个 Agent 第 0 代生成的因子数")
    ap.add_argument("--gens", type=int, default=8, help="每个 Agent 最多进化几代（论文 24）")
    ap.add_argument("--min_gens", type=int, default=4, help="至少进化几代才允许早停")
    ap.add_argument("--parents", type=int, default=8, help="父代数量（论文 32）；子代 = 3 × 父代")
    ap.add_argument("--inject", type=int, default=4, help="每 2 代注入的新因子数")
    ap.add_argument("--workers", type=int, default=8, help="同时调用大模型的数量")
    ap.add_argument("--procs", type=int, default=24, help="并行打分的进程数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--cache", default="./csi300_panel.pkl")
    a = ap.parse_args()
    load_panel(a.data, cache=a.cache)          # 先建好缓存，子进程直接读
    CogAlpha(a).run()


if __name__ == "__main__":
    main()
