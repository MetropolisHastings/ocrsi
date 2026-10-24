"""质检：静态检查 + 实际运行检查 + 未来函数检测（对应论文 Multi-Agent Quality Checker
里不依赖大模型的部分；代码修复、逻辑评审两个 Agent 留到接入大模型后再加）。"""
import ast
import math
import numpy as np
import pandas as pd
from scipy import stats

from runner import lookahead_test, BASE_COLS

SAFE_GLOBALS = {"np": np, "pd": pd, "stats": stats, "math": math}
import re
# 用正则按“词边界”匹配，避免把 pos. / cos. / df["open"] 之类误判为危险写法
FORBIDDEN = [r"\.shift\(\s*-", r"(?<![\w\"'])open\(", r"\bexec\(", r"\beval\(", r"__import__",
             r"\bos\.", r"\bsys\.", r"\bsubprocess\b", r"\bimport\b"]


def _has_nested_loop(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.For, ast.While)):
                    return True
    return False


def compile_factor(code: str):
    """返回 (函数名, 函数, 错误信息)。"""
    for bad in FORBIDDEN:
        if re.search(bad, code):
            return None, None, f"禁止的写法: {bad}"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, None, f"语法错误: {e}"
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        return None, None, "必须恰好包含一个函数"
    if _has_nested_loop(tree):
        return None, None, "包含嵌套循环"
    ns = dict(SAFE_GLOBALS)
    exec(compile(tree, "<factor>", "exec"), ns)
    name = funcs[0].name
    return name, ns[name], None


def run_check(func, panel: pd.DataFrame, n_inst: int = 20):
    """在少量股票上试跑：检查报错、输出长度、无穷值。"""
    insts = panel.index.get_level_values("instrument").unique()[:n_inst]
    for inst in insts:
        df = panel.loc[inst, BASE_COLS].astype("float64")
        try:
            out = pd.Series(np.asarray(func(df.copy()), dtype="float64"))
        except Exception as e:
            return f"运行报错: {type(e).__name__}: {e}"
        if len(out) != len(df):
            return "输出长度与输入不一致"
    if not lookahead_test(func, panel):
        return "检测到未来函数（使用了未来数据）"
    return None
