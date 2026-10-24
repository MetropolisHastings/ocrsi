"""把 RD-Agent 每个 (决策年, 种子) 运行中生成的因子导出成因子值，供 online.py 的 rdagent 方法读取。
做法：找到工作目录下所有 factor.py，在全量数据 daily_pv.h5 上重新执行，读取 result.h5，
并做一次“未来函数”检查：把数据截断到某个日期再算一遍，截断前的因子值必须完全一致，否则丢弃。
在 rdagent4qlib 环境里运行：
  conda run -n rdagent4qlib python rdagent_export.py --runs <workdir>/rd_runs --out runs/rdagent"""
import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CUT = "2019-06-28"          # 未来函数检查的截断日


def run_factor(code: Path, data: pd.DataFrame, timeout=900):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(code, tmp / "factor.py")
        data.to_hdf(tmp / "daily_pv.h5", key="data")
        r = subprocess.run(["python", "factor.py"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        out = tmp / "result.h5"
        if r.returncode != 0 or not out.exists():
            return None
        v = pd.read_hdf(out)
        if isinstance(v, pd.DataFrame):
            if v.shape[1] != 1:
                return None
            v = v.iloc[:, 0]
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="<workdir>/rd_work/git_ignore_folder/"
                                      "factor_implementation_source_data/daily_pv.h5")
    a = ap.parse_args()
    full = pd.read_hdf(a.data)
    cut = full.loc[:CUT]
    for run in sorted(Path(a.runs).glob("y*_s*")):
        dst = Path(a.out) / run.name
        if (dst / "DONE").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        seen, kept = set(), 0
        codes = sorted(run.glob("git_ignore_folder/RD-Agent_workspace/*/factor.py"))
        for code in codes:
            h = hashlib.md5(code.read_bytes()).hexdigest()[:10]
            if h in seen:
                continue
            seen.add(h)
            try:
                v = run_factor(code, full)
                if v is None or v.notna().mean() < 0.3:
                    continue
                vc = run_factor(code, cut)                       # 未来函数检查
                if vc is None:
                    continue
                common = vc.dropna().index.intersection(v.index)
                if len(common) == 0 or not np.allclose(v.loc[common].values, vc.loc[common].values,
                                                       rtol=1e-6, atol=1e-9, equal_nan=True):
                    print(f"  丢弃（疑似用到未来数据）：{code.parent.name}")
                    continue
                v.index = v.index.set_names(["datetime", "instrument"])
                v.to_pickle(dst / f"rd_{h}.pkl")
                shutil.copy(code, dst / f"rd_{h}.py")
                kept += 1
            except Exception as e:                               # 单个因子失败不影响其他因子
                print(f"  失败 {code.parent.name}: {str(e)[:80]}")
        (dst / "DONE").write_text(str(kept))
        print(f"{run.name}: 共 {len(codes)} 个 factor.py，去重后有效且无未来函数 {kept} 个")


if __name__ == "__main__":
    main()
