"""给 online.py 加 --bt 选项：每轮在部署年 t+1 上，对“实际部署的状态”和“从不改动的起点状态”
各做一次 TopK-Dropout 回测（Top50、每天最多换 5 只、含手续费与涨跌停），结果写进每轮记录。
用法：python patch_online_bt.py online.py     （会先备份为 online.py.before_bt）"""
import shutil
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "online.py")
s = p.read_text()
if "--bt" in s:
    print("已经打过补丁，未做改动"); sys.exit(0)
shutil.copy(p, p.with_suffix(".py.before_bt"))


def ins(anchor, new, after=True):
    global s
    assert s.count(anchor) == 1, f"锚点不唯一或不存在：{anchor!r}"
    s = s.replace(anchor, anchor + new if after else new + anchor)


ins('    ap.add_argument("--min_gain", type=float, default=0.0003)\n',
    '    ap.add_argument("--bt", action="store_true", help="每轮对部署状态和起点状态在 t+1 年做 TopK-Dropout 回测")\n')

ins('    S0 = dict(S)\n', '''    if a.bt:
        from data import load_panel, load_calendar, read_bin
        _panel = load_panel(a.data, index="csi300", end="2099-12-31", cache=a.full_cache)
        _cal = load_calendar(Path(a.data))
        _b = read_bin(Path(a.data) / "features/sh000300/open.day.bin", _cal)
        _bench = (_b.shift(-2) / _b.shift(-1) - 1)

        def bt_year(st, t):
            X = ev.matrix(st).values
            p = ev.fit_pred(X, f"{t}-12-15", f"{t + 1}-01-01", f"{t + 1}-12-31")
            res = bt.backtest(p, _panel, _bench)
            ex = res["excess"]
            return {"ann_ret": float(res["ret"].mean() * 252), "ann_bench": float(res["bench"].mean() * 252),
                    "ann_excess": float(ex.mean() * 252),
                    "ir": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
                    "turnover": float(res["turnover"].mean()), "days": int(len(res)),
                    "daily_excess": [round(float(v), 6) for v in ex.values],
                    "dates": [str(d.date()) for d in res.index]}
''')

ins('        S, D = new_S, new_D\n', '''        if a.bt:
            row["bt_deployed"] = bt_year(new_D, t)          # 本轮决策之后实际部署的状态
            row["bt_static"] = bt_year(S0, t)               # 从不改动：起点状态
            log(f"  回测 {t + 1}：部署状态年化超额 {row['bt_deployed']['ann_excess']:+.2%}，"
                f"起点状态 {row['bt_static']['ann_excess']:+.2%}")
''', after=False)

p.write_text(s)
print("补丁完成，备份在", p.with_suffix(".py.before_bt"))
