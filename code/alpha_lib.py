"""类 Alpha158 特征库：按 Qlib Alpha158 的算子与窗口写成单股票因子代码（102 个）。
公募、私募最常见的“传统量价因子库”做法：K 线形态 + 价格比 + 18 种滚动算子 × 5 个窗口。
每个因子都是一段 def f(df) 代码，和大模型挖出的因子走同一条计算路径。"""

W = [5, 10, 20, 30, 60]

KBAR = {
    "KMID": '(c - o) / o',
    "KLEN": '(h - l) / o',
    "KMID2": '(c - o) / (h - l + 1e-12)',
    "KUP": '(h - np.maximum(o, c)) / o',
    "KUP2": '(h - np.maximum(o, c)) / (h - l + 1e-12)',
    "KLOW": '(np.minimum(o, c) - l) / o',
    "KLOW2": '(np.minimum(o, c) - l) / (h - l + 1e-12)',
    "KSFT": '(2 * c - h - l) / o',
    "KSFT2": '(2 * c - h - l) / (h - l + 1e-12)',
    "OPEN0": 'o / c',
    "HIGH0": 'h / c',
    "LOW0": 'l / c',
}

ROLL = {
    "ROC": 'c.shift({w}) / c',
    "MA": 'c.rolling({w}, min_periods={m}).mean() / c',
    "STD": 'c.rolling({w}, min_periods={m}).std() / c',
    "BETA": 'c.rolling({w}, min_periods={m}).cov(t) / t.rolling({w}, min_periods={m}).var() / c',
    "RSQR": 'c.rolling({w}, min_periods={m}).corr(t) ** 2',
    "MAX": 'h.rolling({w}, min_periods={m}).max() / c',
    "MIN": 'l.rolling({w}, min_periods={m}).min() / c',
    "QTLU": 'c.rolling({w}, min_periods={m}).quantile(0.8) / c',
    "QTLD": 'c.rolling({w}, min_periods={m}).quantile(0.2) / c',
    "RSV": '(c - l.rolling({w}, min_periods={m}).min()) / (h.rolling({w}, min_periods={m}).max() - l.rolling({w}, min_periods={m}).min() + 1e-12)',
    "CORR": 'c.rolling({w}, min_periods={m}).corr(np.log(v + 1))',
    "CORD": '(c / c.shift(1)).rolling({w}, min_periods={m}).corr(np.log(v / v.shift(1) + 1))',
    "CNTP": '(r > 0).astype(float).rolling({w}, min_periods={m}).mean()',
    "SUMP": 'r.clip(lower=0).rolling({w}, min_periods={m}).sum() / (r.abs().rolling({w}, min_periods={m}).sum() + 1e-12)',
    "VMA": 'v.rolling({w}, min_periods={m}).mean() / (v + 1e-12)',
    "VSTD": 'v.rolling({w}, min_periods={m}).std() / (v + 1e-12)',
    "WVMA": '(r.abs() * v).rolling({w}, min_periods={m}).std() / ((r.abs() * v).rolling({w}, min_periods={m}).mean() + 1e-12)',
    "VSUMP": 'v.diff().clip(lower=0).rolling({w}, min_periods={m}).sum() / (v.diff().abs().rolling({w}, min_periods={m}).sum() + 1e-12)',
}

HEAD = ('    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]\n'
        '    r = c / c.shift(1) - 1\n'
        '    t = pd.Series(np.arange(len(c), dtype=float), index=c.index)\n')


def _code(name, expr):
    return f"def lib_{name}(df):\n    \"\"\"Alpha158-style feature {name}.\"\"\"\n{HEAD}    return {expr}\n"


def library():
    """返回 {名字: 代码}，共 12 + 18×5 = 102 个。"""
    out = {f"lib_{k}": _code(k, e) for k, e in KBAR.items()}
    for k, e in ROLL.items():
        for w in W:
            out[f"lib_{k}{w}"] = _code(f"{k}{w}", e.format(w=w, m=max(2, w // 2)))
    return out


def family(name):
    """lib_ROC20 → ROC；K 线类 → KBAR。"""
    s = name[4:].rstrip("0123456789")
    return s if s in ROLL else "KBAR"
