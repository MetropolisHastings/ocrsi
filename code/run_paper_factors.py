"""第一步验证：用论文给出的因子跑通“数据 → 因子 → 指标”整个流程。

用法：python run_paper_factors.py --data ./qlib_bin
"""
import argparse
import time
import pandas as pd

from data import load_panel, make_label
from metrics import evaluate
from runner import apply_factor, lookahead_test
from factors_paper import PAPER_FACTORS

SPLITS = {"训练 2011-2019": ("2011-01-01", "2019-12-31"),
          "测试 2021-2024": ("2021-01-01", "2024-12-31")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--cache", default="./csi300_panel.pkl")
    args = ap.parse_args()

    t = time.time()
    panel = load_panel(args.data, "csi300", "2010-01-01", "2024-12-31", cache=args.cache)
    label = make_label(panel, horizon=10)
    print(f"面板：{panel.index.get_level_values(0).nunique()} 只股票，"
          f"{len(panel)} 行，用时 {time.time()-t:.0f}s")

    rows = []
    for name, func in PAPER_FACTORS.items():
        ok = lookahead_test(func, panel)
        f = apply_factor(func, panel)
        for split, (s, e) in SPLITS.items():
            dt = f.index.get_level_values("datetime")
            m = (dt >= s) & (dt <= e)
            res = evaluate(f[m], label.reindex(f.index)[m])
            rows.append({"因子": name, "区间": split, "无未来函数": ok, **res})
        print(f"完成 {name}")

    out = pd.DataFrame(rows).set_index(["因子", "区间"])
    pd.set_option("display.width", 200)
    print(out.round(4))
    out.to_csv("paper_factors_result.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
