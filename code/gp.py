"""遗传规划（GP）因子挖掘：私募最常用的公式化因子挖掘方法之一。

公式树 → 一段 pandas 表达式字符串。同一个字符串既能在“日期 × 股票”的宽表上整体计算（用来算适应度，快），
也能放进 def f(df) 里按单只股票计算（和其他方法的因子走同一条评估路径），两者数学上完全一致。
适应度：训练期（截至 2017 年）每 5 个交易日抽一天，单因子 |RankIC|。"""
import numpy as np
import pandas as pd

LEAVES = ["o", "h", "l", "c", "v", "r"]
WINS = [5, 10, 20, 40, 60]
UNARY = ["neg", "abs", "slog"]
BINARY = ["add", "sub", "mul", "div"]
TS1 = ["ts_mean", "ts_std", "ts_delta", "ts_delay", "ts_max", "ts_min", "ts_z"]
TS2 = ["ts_corr"]


def to_str(t):
    """公式树 → pandas 表达式。"""
    if isinstance(t, str):
        return t
    op = t[0]
    if op in UNARY:
        x = to_str(t[1])
        return {"neg": f"(-({x}))", "abs": f"({x}).abs()",
                "slog": f"(np.sign({x}) * np.log1p(({x}).abs()))"}[op]
    if op in BINARY:
        a, b = to_str(t[1]), to_str(t[2])
        return {"add": f"(({a}) + ({b}))", "sub": f"(({a}) - ({b}))", "mul": f"(({a}) * ({b}))",
                "div": f"(({a}) / ({b}).where(({b}).abs() > 1e-9))"}[op]
    w = t[-1]
    m = max(2, w // 2)
    x = to_str(t[1])
    if op == "ts_corr":
        return f"({x}).rolling({w}, min_periods={m}).corr({to_str(t[2])})"
    return {"ts_mean": f"({x}).rolling({w}, min_periods={m}).mean()",
            "ts_std": f"({x}).rolling({w}, min_periods={m}).std()",
            "ts_delta": f"({x}).diff({w})",
            "ts_delay": f"({x}).shift({w})",
            "ts_max": f"({x}).rolling({w}, min_periods={m}).max()",
            "ts_min": f"({x}).rolling({w}, min_periods={m}).min()",
            "ts_z": f"((({x}) - ({x}).rolling({w}, min_periods={m}).mean()) / ({x}).rolling({w}, min_periods={m}).std())"}[op]


def to_code(name, t):
    return (f"def {name}(df):\n    \"\"\"GP formula: {to_str(t)[:200]}\"\"\"\n"
            '    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]\n'
            '    r = c / c.shift(1) - 1\n'
            f"    return {to_str(t)}\n")


def size(t):
    return 1 if isinstance(t, str) else 1 + sum(size(x) for x in t[1:] if not isinstance(x, int))


class GP:
    def __init__(self, panel, label, rng, fit_end="2017-12-15", step=5, pop=80, gens=6, max_depth=4):
        w = lambda col: panel[col].unstack("instrument").astype("float64")
        self.env = {k: w(col) for k, col in zip("ohlcv", ["open", "high", "low", "close", "volume"])}
        self.env["r"] = self.env["c"] / self.env["c"].shift(1) - 1
        self.env["np"] = np
        dates = self.env["c"].index
        inidx = panel["in_index"].unstack("instrument").reindex(index=dates, columns=self.env["c"].columns)
        y = label.unstack("instrument").reindex(index=dates, columns=self.env["c"].columns)
        self._inidx, self._yw, self._step = inidx, y, step
        self.set_window(fit_end)
        self.rng, self.pop, self.gens, self.max_depth = rng, pop, gens, max_depth

    def set_window(self, fit_end):
        """适应度只用 2011 年到 fit_end 的数据（在线 RSI 每轮向前滚动）。"""
        dates = self._yw.index
        rows = dates[(dates >= pd.Timestamp("2011-01-01")) & (dates <= pd.Timestamp(fit_end))][::self._step]
        self.rows = rows
        self.ymask = self._inidx.loc[rows].fillna(False).values.astype(bool)
        self.y = self._yw.loc[rows].values
        self.cache = {}

    # ---------- 随机树 ----------
    def rand_tree(self, d):
        rng = self.rng
        if d <= 1 or rng.random() < 0.25:
            return LEAVES[rng.integers(len(LEAVES))]
        k = rng.random()
        if k < 0.15:
            return (UNARY[rng.integers(len(UNARY))], self.rand_tree(d - 1))
        if k < 0.45:
            return (BINARY[rng.integers(len(BINARY))], self.rand_tree(d - 1), self.rand_tree(d - 1))
        if k < 0.9:
            return (TS1[rng.integers(len(TS1))], self.rand_tree(d - 1), int(WINS[rng.integers(len(WINS))]))
        return ("ts_corr", self.rand_tree(d - 1), self.rand_tree(d - 1), int(WINS[rng.integers(len(WINS))]))

    def nodes(self, t, path=()):
        yield path, t
        if not isinstance(t, str):
            for i, x in enumerate(t[1:], 1):
                if not isinstance(x, int):
                    yield from self.nodes(x, path + (i,))

    def replace(self, t, path, new):
        if not path:
            return new
        t = list(t)
        t[path[0]] = self.replace(t[path[0]], path[1:], new)
        return tuple(t)

    def mutate(self, t):
        ps = list(self.nodes(t))
        path, sub = ps[self.rng.integers(len(ps))]
        if not isinstance(sub, str) and isinstance(sub[-1], int) and self.rng.random() < 0.5:
            return self.replace(t, path, sub[:-1] + (int(WINS[self.rng.integers(len(WINS))]),))
        return self.replace(t, path, self.rand_tree(3))

    def cross(self, a, b):
        pa, pb = list(self.nodes(a)), list(self.nodes(b))
        path, _ = pa[self.rng.integers(len(pa))]
        _, sub = pb[self.rng.integers(len(pb))]
        return self.replace(a, path, sub)

    # ---------- 适应度 ----------
    def values(self, t):
        with np.errstate(all="ignore"):
            out = eval(to_str(t), {"__builtins__": {}}, self.env)     # 表达式由本程序生成，不含外部输入
        return out

    def fitness(self, t):
        s = to_str(t)
        if s in self.cache:
            return self.cache[s]
        fit, vec = -1.0, None
        if size(t) <= 25 and not isinstance(t, str):
            try:
                f = self.values(t).reindex(self.rows).values
                f = np.where(np.isfinite(f), f, np.nan)
                m = self.ymask & ~np.isnan(f) & ~np.isnan(self.y)
                if m.sum() > 0.5 * self.ymask.sum():
                    ok = m.sum(axis=1) >= 50                          # 向量化的逐日秩相关
                    a = pd.DataFrame(np.where(m, f, np.nan)[ok]).rank(axis=1).values
                    b = pd.DataFrame(np.where(m, self.y, np.nan)[ok]).rank(axis=1).values
                    a = a - np.nanmean(a, axis=1, keepdims=True)
                    b = b - np.nanmean(b, axis=1, keepdims=True)
                    den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
                    ics = (np.nansum(a * b, axis=1) / np.where(den > 0, den, np.nan))
                    ics = ics[np.isfinite(ics)]
                    if len(ics) > 20:
                        fit = abs(float(np.mean(ics)))
                        fr = np.where(m, f, np.nan)                   # 用于去重的截面排名签名
                        vec = pd.DataFrame(fr[::4]).rank(axis=1, pct=True).values.ravel()
            except Exception:
                pass
        self.cache[s] = (fit, vec)
        return fit, vec

    def run(self, k):
        """进化 gens 代，返回适应度最高、彼此相关性 < 0.8 的 k 个公式。"""
        P = [self.rand_tree(self.max_depth) for _ in range(self.pop)]
        for g in range(self.gens):
            fits = [self.fitness(t)[0] for t in P]
            order = np.argsort(fits)[::-1]
            elite = [P[i] for i in order[: self.pop // 10]]
            def pick():
                i, j = self.rng.integers(len(P), size=2)
                return P[i] if fits[i] >= fits[j] else P[j]
            nxt = list(elite)
            while len(nxt) < self.pop:
                u = self.rng.random()
                child = self.cross(pick(), pick()) if u < 0.6 else self.mutate(pick()) if u < 0.9 \
                    else self.rand_tree(self.max_depth)
                nxt.append(child)
            P = nxt
        scored = sorted({to_str(t): t for t in P + list(self._all())}.values(),
                        key=lambda t: -self.fitness(t)[0])
        out, sigs = [], []
        for t in scored:
            fit, vec = self.fitness(t)
            if fit <= 0 or vec is None:
                continue
            ok = True
            for s in sigs:
                mm = ~np.isnan(vec) & ~np.isnan(s)
                if mm.sum() > 100 and abs(np.corrcoef(vec[mm], s[mm])[0, 1]) > 0.8:
                    ok = False
                    break
            if ok:
                out.append((t, fit))
                sigs.append(vec)
            if len(out) >= k:
                break
        return out

    def _all(self):
        return []
