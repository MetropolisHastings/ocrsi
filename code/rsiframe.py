"""统一框架：任何因子 / 模型方法 + RSI + 可认证入池。

每一种方法都被看成同一个循环（第 r 轮）：
    状态 S_r（因子集合 + 合成模型）
    → 改进算子 P（方法自带）提出一批候选改动 Δ_1..Δ_K
    → 入池规则 A（与方法无关）：在选择窗口上贪心接受能提高分数的改动；
       可认证模式下再在确认窗口上做一次分块 bootstrap 置信下界检验，整批通过才生效
    → S_{r+1}，在测试期（2021-2024）和新样本期（2025 至今）上计算 RSI。

方法（改进算子）：
    classic  传统因子库：从类 Alpha158 特征库里按训练期单因子表现挑候选，加入线性合成（Ridge）
    ml       机器学习：梯度提升树（LightGBM 同类） 直接用全部特征预测，每轮提出若干超参数 / 特征族组合
    gp       遗传规划：每轮进化一批公式因子，加入线性合成（Ridge）
（大模型智能体方法见 orchestra.py，用的是同一个 PoolEval 和同一套入池、RSI 定义。）

用法：python rsiframe.py --method gp --admit certified --seed 0 --rounds 3 --full_cache panel_full.pkl
"""
import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_lib
import backtest as bt
import orchestra
import gp as gpmod
import prompts
from baseline_factors import BASELINE
from cogalpha import log
from orchestra import PoolEval, METRICS, CONF_METRICS

# 梯度提升树（与 LightGBM 同类的直方图 GBDT；本机无 LightGBM 时用 scikit-learn 的 HistGradientBoosting）
LGB0 = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
            max_features=0.8, l2_regularization=1.0)
GRID = dict(max_iter=[100, 200, 400], learning_rate=[0.02, 0.05, 0.1], max_leaf_nodes=[15, 31, 63, 127],
            min_samples_leaf=[50, 200, 800], max_features=[0.5, 0.8, 1.0], l2_regularization=[0.0, 1.0, 10.0])
# 其他模型类别（统一框架的“机器学习 / 深度学习”分支）：状态里的 model 带 kind 字段
MODEL0 = {
    "ridge": {"kind": "ridge", "alpha": 10.0},
    "mlp": {"kind": "mlp", "hidden": [128, 64], "l2": 1e-4, "lr": 1e-3, "epochs": 10},
    "transformer": {"kind": "transformer", "d_model": 32, "nlayers": 2, "nhead": 4, "lr": 1e-3, "epochs": 4,
                    "seq_len": 20, "dropout": 0.1},
    "gat": {"kind": "gat", "hidden": 32, "layers": 1, "lr": 1e-3, "epochs": 4, "seq_len": 20, "dropout": 0.1},
    "tra": {"kind": "tra", "hidden": 32, "heads": 3, "tau": 1.0, "rho": 0.1, "lr": 1e-3, "epochs": 4, "seq_len": 20},
    "adarnn": {"kind": "adarnn", "hidden": 32, "periods": 2, "lam": 0.5, "lr": 1e-3, "epochs": 4, "seq_len": 20,
               "dropout": 0.1},
    "lstm": {"kind": "lstm", "hidden": 64, "layers": 2, "lr": 1e-3, "epochs": 4, "seq_len": 20, "dropout": 0.1},
    "gru": {"kind": "gru", "hidden": 64, "layers": 2, "lr": 1e-3, "epochs": 4, "seq_len": 20, "dropout": 0.1},
    "catboost": {"kind": "catboost", "iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
                 "subsample": 0.8, "rsm": 0.8},
    "rf": {"kind": "rf", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 200, "max_features": 0.3},
    "alstm": {"kind": "alstm", "hidden": 64, "layers": 2, "lr": 1e-3, "epochs": 4, "seq_len": 20, "dropout": 0.1},
    "dens": {"kind": "dens", "num_models": 4, "decay": 0.5, "bins": 10, "keep": 0.8,
             "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200},
    "lasso": {"kind": "lasso", "alpha": 1e-4, "max_iter": 2000},
    "elasticnet": {"kind": "elasticnet", "alpha": 1e-4, "l1_ratio": 0.5, "max_iter": 2000},
    "lightgbm": {"kind": "lightgbm", "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
                 "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "xgboost": {"kind": "xgboost", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
}
GRIDS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
    "mlp": {"hidden": [[64], [128], [128, 64], [256, 128]], "l2": [1e-5, 1e-4, 1e-3, 1e-2],
            "lr": [3e-4, 1e-3, 3e-3], "epochs": [5, 10, 20]},
    "transformer": {"d_model": [16, 32, 64], "nlayers": [1, 2, 3], "nhead": [2, 4], "lr": [3e-4, 1e-3],
                    "epochs": [2, 4, 8], "seq_len": [10, 20, 40], "dropout": [0.0, 0.1, 0.3]},
    "gat": {"hidden": [16, 32, 64], "layers": [1, 2], "lr": [3e-4, 1e-3], "epochs": [2, 4, 8],
            "seq_len": [10, 20, 40], "dropout": [0.0, 0.1, 0.3]},
    "tra": {"hidden": [16, 32, 64], "heads": [2, 3, 5], "tau": [0.1, 1.0], "rho": [0.1, 1.0], "lr": [3e-4, 1e-3],
            "epochs": [2, 4, 8], "seq_len": [10, 20, 40]},
    "adarnn": {"hidden": [16, 32, 64], "periods": [2, 3], "lam": [0.1, 0.5, 1.0], "lr": [3e-4, 1e-3],
               "epochs": [2, 4, 8], "seq_len": [10, 20, 40], "dropout": [0.0, 0.1, 0.3]},
    "lstm": {"hidden": [32, 64, 128], "layers": [1, 2], "lr": [3e-4, 1e-3], "epochs": [2, 4, 8],
             "seq_len": [10, 20, 40], "dropout": [0.0, 0.1, 0.3]},
    "gru": {"hidden": [32, 64, 128], "layers": [1, 2], "lr": [3e-4, 1e-3], "epochs": [2, 4, 8],
            "seq_len": [10, 20, 40], "dropout": [0.0, 0.1, 0.3]},
    "catboost": {"iterations": [150, 300, 600], "depth": [4, 6, 8], "learning_rate": [0.02, 0.05, 0.1],
                 "l2_leaf_reg": [1.0, 3.0, 10.0], "subsample": [0.6, 0.8, 1.0], "rsm": [0.5, 0.8, 1.0]},
    "rf": {"n_estimators": [100, 200, 400], "max_depth": [6, 8, 12, 16], "min_samples_leaf": [50, 200, 800],
           "max_features": [0.1, 0.3, 0.5]},
    "alstm": {"hidden": [32, 64, 128], "layers": [1, 2], "lr": [3e-4, 1e-3], "epochs": [2, 4, 8],
              "seq_len": [10, 20, 40], "dropout": [0.0, 0.1, 0.3]},
    "dens": {"num_models": [3, 4, 6], "decay": [0.3, 0.5, 0.8], "bins": [5, 10, 20], "keep": [0.6, 0.8, 1.0],
             "n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63],
             "min_child_samples": [50, 200, 800]},
    "lasso": {"alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "max_iter": [1000, 2000, 5000]},
    "elasticnet": {"alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                   "max_iter": [1000, 2000, 5000]},
    "lightgbm": {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63, 127],
                 "min_child_samples": [50, 200, 800], "subsample": [0.6, 0.8, 1.0],
                 "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]},
    "xgboost": {"n_estimators": [100, 200, 400], "max_depth": [3, 4, 6, 8], "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]},
}
DEEP = ("transformer", "gat", "tra", "adarnn", "lstm", "gru", "alstm")
LIB = alpha_lib.library()


class DoubleEnsemble:
    """DoubleEnsemble（Zhang et al., ICDM 2020）的简化实现：K 个 LightGBM 子模型依次训练，
    (1) 样本重加权：按当前集成在各样本上的误差分箱，难度适中的箱权重大、过易 / 过难（多为噪声）的箱权重小；
    (2) 特征选择：用置换重要性保留前 keep 比例的特征。预测为各子模型平均。"""

    def __init__(self, num_models=4, decay=0.5, bins=10, keep=0.8, **lgb_params):
        self.K, self.decay, self.bins, self.keep, self.lp = int(num_models), decay, int(bins), keep, lgb_params

    def _lgb(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(**self.lp, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 n_jobs=3, verbose=-1, random_state=len(self.models))

    def fit(self, X, y):
        n, d = X.shape
        w = np.ones(n)
        feats = np.arange(d)
        self.models = []
        ens = np.zeros(n)
        rng = np.random.default_rng(0)
        for k in range(self.K):
            m = self._lgb()
            m.fit(X[:, feats], y, sample_weight=w)
            self.models.append((m, feats))
            ens = ens + (m.predict(X[:, feats]) - ens) / (k + 1)
            if k == self.K - 1:
                break
            loss = (ens - y) ** 2                          # 样本重加权
            q = pd.qcut(pd.Series(loss).rank(method="first"), self.bins, labels=False).values
            mid = (self.bins - 1) / 2
            wb = 1.0 / (1.0 + self.decay * np.abs(np.arange(self.bins) - mid))
            w = wb[q] * self.bins / wb.sum()
            sub = rng.choice(n, size=min(n, 20000), replace=False)   # 置换重要性做特征选择
            Xs, ys = X[sub], y[sub]
            base = np.mean((ens[sub] - ys) ** 2)
            imp = np.empty(d)
            for j in range(d):
                Xp = Xs.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                imp[j] = np.mean((self._predict(Xp) - ys) ** 2) - base
            feats = np.sort(np.argsort(-imp)[:max(5, int(self.keep * d))])
        return self

    def _predict(self, X):
        return np.mean([m.predict(X[:, f]) for m, f in self.models], axis=0)

    def predict(self, X):
        return self._predict(X)


# ============================ 评估（与 orchestra 完全相同的指标，只是状态里多了“合成模型”） ============================
class FrameEval(PoolEval):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rk = {}
        self._model = "ridge"

    def compute(self, factors):
        super().compute(factors)
        for f in factors:                                 # 立刻转成截面排名的 float32 数组，省内存
            n = f["name"]
            v = self.vals.get(n)
            if isinstance(v, pd.Series):
                self.add_values(n, v)

    def add_values(self, n, v):
        self.rk[n] = bt.cs_rank(v.reindex(self.idx).to_frame()).iloc[:, 0].values.astype("float32")
        self.vals[n] = 1

    def confirm_rsi(self, old, new, alpha_r, w, B=2000, block=20, seed=0):
        """对 RSI 指数本身做认证：把 3 项增益按 RSI 的归一化权重合成一个统计量，只检验一次（M = 1）。"""
        so, sn = self.conf_series(old), self.conf_series(new)
        idx = so.index.intersection(sn.index)
        o, n = so.loc[idx].values, sn.loc[idx].values
        T = len(idx)
        w = np.asarray(w, float)
        comp = self._conf_stats(o, n)
        point = float(np.dot(comp, w) / len(w))
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(T / block))
        boots = np.empty(B)
        for b in range(B):
            st = rng.integers(0, T - block + 1, nb)
            ix = (st[:, None] + np.arange(block)).ravel()[:T]
            boots[b] = np.dot(self._conf_stats(o[ix], n[ix]), w) / len(w)
        se = float(boots.std(ddof=1))
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - alpha_r)
        keys = list(CONF_METRICS)
        return {"days": T, "alpha_r": alpha_r, "z": z, "gain_rsi": point, "se_rsi": se, "lcb_rsi": point - z * se,
                **{f"gain_{k}": float(comp[i]) for i, k in enumerate(keys)}, "passed": bool(point - z * se > 0)}

    def matrix(self, st):
        if isinstance(st, dict) and "blend" in st:          # 部分接受：多个状态按权重混合预测
            self._blend = st["blend"]
            return pd.DataFrame(np.zeros((len(self.idx), 1), dtype="float32"), index=self.idx, columns=["blend"])
        self._blend = None
        names = st["names"] if isinstance(st, dict) else st
        self._model = st.get("model", "ridge") if isinstance(st, dict) else "ridge"
        self._key = json.dumps({"n": sorted(names), "m": self._model}, sort_keys=True, default=str)
        if isinstance(self._model, dict) and self._model.get("kind") in DEEP:
            return pd.DataFrame(np.zeros((len(self.idx), 1), dtype="float32"), index=self.idx, columns=["seq"])
        names = [n for n in names if n in self.rk]
        return pd.DataFrame(np.column_stack([self.rk[n] for n in names]), index=self.idx, columns=names)

    def fit_pred(self, X, train_end, start, end):
        bl = getattr(self, "_blend", None)
        if bl:                                             # 各子状态的截面百分位排名按权重加总
            self._blend = None
            out = None
            for wgt, sub in bl:
                p = self.fit_pred(self.matrix(sub).values, train_end, start, end)
                rk = p.groupby(level="datetime").rank(pct=True) * float(wgt)
                out = rk if out is None else out.add(rk, fill_value=0.0)
            return out
        if self._model == "ridge":
            return super().fit_pred(X, train_end, start, end)
        key = (getattr(self, "_key", None), train_end, start, end)   # 同一状态、同一窗口只训练一次（深度模型很贵）
        memo = self.__dict__.setdefault("_memo", {})
        if key in memo:
            return memo[key]
        days = pd.DatetimeIndex(sorted(set(self.dates[self.dates <= pd.Timestamp(train_end)])))[::3]
        tr = np.isin(self.dates, days) & self.y.notna().values
        te = self.mask(start, end)
        p = dict(self._model)
        kind = p.pop("kind", "gbdt")
        if kind in DEEP:
            import deep
            pred = deep.fit_predict(self, p, tr, te, kind=kind)
        else:
            if kind == "ridge":
                from sklearn.linear_model import Ridge
                m = Ridge(alpha=p["alpha"])
            elif kind == "mlp":
                from sklearn.neural_network import MLPRegressor
                m = MLPRegressor(hidden_layer_sizes=tuple(p["hidden"]), alpha=p["l2"], learning_rate_init=p["lr"],
                                 max_iter=p["epochs"], batch_size=2048, early_stopping=False, random_state=0)
            elif kind == "dens":
                m = DoubleEnsemble(**p)
            elif kind == "catboost":
                from catboost import CatBoostRegressor
                m = CatBoostRegressor(**p, bootstrap_type="Bernoulli", thread_count=3, verbose=0, random_seed=0)
            elif kind == "rf":
                from sklearn.ensemble import RandomForestRegressor
                m = RandomForestRegressor(**p, n_jobs=3, random_state=0)
            elif kind == "lasso":
                from sklearn.linear_model import Lasso
                m = Lasso(alpha=p["alpha"], max_iter=p["max_iter"])
            elif kind == "elasticnet":
                from sklearn.linear_model import ElasticNet
                m = ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=p["max_iter"])
            elif kind == "lightgbm":
                import lightgbm as lgb
                m = lgb.LGBMRegressor(**p, subsample_freq=1, n_jobs=3, verbose=-1, random_state=0)
            elif kind == "xgboost":
                import xgboost as xgb
                m = xgb.XGBRegressor(**p, tree_method="hist", n_jobs=3, random_state=0)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(**p, early_stopping=False, random_state=0)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[tr], self.y.values[tr])
            pred = m.predict(X[te])
        out = pd.Series(np.asarray(pred, dtype="float64"), index=self.idx[te])
        if len(memo) > 300:
            memo.clear()
        memo[key] = out
        return out


# ============================ 改进算子（各方法自带） ============================
class Classic:
    """传统做法：研究员从现成因子库里按单因子表现挑一批候选，逐个看能否提升组合。"""
    K = 20

    def __init__(self, fr):
        self.fr = fr
        self.ic = fr.lib_train_ic()

    def init_state(self):
        return {"names": list(BASELINE), "model": "ridge"}

    def propose(self, st, r, rng):
        left = [n for n in LIB if n not in st["names"]]
        pick = list(rng.choice(left, size=min(self.K, len(left)), replace=False))
        pick.sort(key=lambda n: -abs(self.ic[n]))
        return [{"add": [n], "desc": n} for n in pick]


class ML:
    """机器学习：梯度提升树用全部类 Alpha158 特征直接预测；每轮在超参数和特征族上提出 K 个候选配置。"""
    K = 8
    KIND = "gbdt"

    def __init__(self, fr):
        self.fr = fr

    def init_state(self):
        return {"names": list(LIB), "model": dict(LGB0) if self.KIND == "gbdt" else dict(MODEL0[self.KIND])}

    def propose(self, st, r, rng):
        fams = sorted({alpha_lib.family(n) for n in LIB})
        grid = GRID if self.KIND == "gbdt" else GRIDS[self.KIND]
        moves = []
        for _ in range(self.K):
            p = dict(st["model"])
            for key in rng.choice(list(grid), size=min(2, len(grid)), replace=False):
                v = grid[key][rng.integers(len(grid[key]))]
                p[key] = v if isinstance(v, list) else type(grid[key][0])(v)
            names = list(st["names"])
            if self.KIND not in DEEP and rng.random() < 0.5:
                fam = fams[rng.integers(len(fams))]
                has = [n for n in names if alpha_lib.family(n) == fam]
                if has and len(names) - len(has) >= 40:
                    names = [n for n in names if n not in has]
                else:
                    names = sorted(set(names) | {n for n in LIB if alpha_lib.family(n) == fam})
            moves.append({"replace": {"names": names, "model": p},
                          "desc": f"{len(names)} features, " + ",".join(f"{k}={p[k]}" for k in grid)})
        return moves


class RidgeM(ML):
    """Ridge 回归（线性模型）：全部类 Alpha158 特征，每轮调正则强度、增删特征族。"""
    K, KIND = 8, "ridge"


class MLPM(ML):
    """多层感知机（最基础的深度模型）：每轮调网络宽度 / 深度、正则、学习率、训练轮数，增删特征族。"""
    K, KIND = 6, "mlp"


class TransformerM(ML):
    """Transformer：输入每只股票过去 seq_len 天的开高低收量（按当天收盘价 / 成交量归一，类 Alpha360），
    每轮调结构与训练超参数。需要 PyTorch，建议在服务器 GPU 上跑。"""
    K, KIND = 4, "transformer"


class GATM(ML):
    """GATs：GRU 编码 + 每天截面上的图注意力（股票之间的关系由注意力自动学习）。"""
    K, KIND = 4, "gat"


class TRAM(ML):
    """TRA：多个预测头 + 路由器，最优传输分配样本，适应多种市场状态。"""
    K, KIND = 4, "tra"


class AdaRNNM(ML):
    """AdaRNN：按分布差异切分训练期，并对齐各段隐状态，减轻分布漂移。"""
    K, KIND = 4, "adarnn"


class LSTMM(ML):
    """LSTM：输入过去 seq_len 天的开高低收量（类 Alpha360），Qlib 基准里最常用的时序模型。"""
    K, KIND = 4, "lstm"


class GRUM(ML):
    """GRU：结构比 LSTM 简单的循环网络，输入同 LSTM。"""
    K, KIND = 4, "gru"


class CatBoostM(ML):
    """CatBoost：对称树的梯度提升，每轮调迭代次数、树深、学习率、L2、行/列采样，增删特征族。"""
    K, KIND = 8, "catboost"


class RFM(ML):
    """随机森林：每轮调树数量、树深、最小叶样本、特征采样比例，增删特征族。"""
    K, KIND = 8, "rf"


class ALSTMM(ML):
    """ALSTM：LSTM + 时间注意力，Qlib 基准模型。"""
    K, KIND = 4, "alstm"


class DoubleEnsembleM(ML):
    """DoubleEnsemble：样本重加权 + 特征选择的 LightGBM 集成，Qlib 基准模型。"""
    K, KIND = 6, "dens"


class LassoM(ML):
    """Lasso：L1 正则线性模型，每轮调 alpha、迭代次数，增删特征族。"""
    K, KIND = 8, "lasso"


class ElasticNetM(ML):
    """ElasticNet：L1 + L2 正则线性模型，每轮调 alpha、l1_ratio、迭代次数，增删特征族。"""
    K, KIND = 8, "elasticnet"


class LightGBMM(ML):
    """LightGBM（真实库）：每轮调树数量、学习率、叶子数、最小叶样本、行/列采样、L2，增删特征族。"""
    K, KIND = 8, "lightgbm"


class XGBoostM(ML):
    """XGBoost（真实库）：每轮调树数量、树深、学习率、行/列采样、L2，增删特征族。"""
    K, KIND = 8, "xgboost"


class RDAgentMethod:
    """RD-Agent(Q) 因子模式：每个决策年 t 单独运行 RD-Agent（窗口 ≤ t-1），由 rdagent_export.py 导出因子值到
    runs/rdagent/y{t}_s{seed}/rd_*.pkl。本方法读取这些因子作为候选，筛选、认证、部署与其他方法完全相同。"""
    K = 20

    def __init__(self, fr):
        import os
        self.fr = fr
        self.dir = Path(os.environ.get("RDAGENT_FACTORS", "runs/rdagent"))

    def init_state(self):
        return {"names": list(BASELINE), "model": "ridge"}

    def propose(self, st, r, rng):
        ev, t = self.fr.ev, self.fr.decision_year
        d = self.dir / f"y{t}_s{self.fr.a.seed}"
        files = sorted(d.glob("rd_*.pkl"))
        if not files:
            log(f"[rdagent] {d} 下没有导出的因子，本轮无候选")
            return []
        dates = ev.idx.get_level_values("datetime")
        train = np.asarray(dates <= pd.Timestamp(self.fr.fit_end))
        scored = []
        for f in files:
            name = f"rd{t}_{f.stem[3:]}"
            if name not in ev.rk:
                v = pd.read_pickle(f)
                if list(v.index.names) != list(ev.idx.names):
                    v = v.reorder_levels(ev.idx.names)
                ins0 = str(ev.idx.get_level_values("instrument")[0])   # 股票代码大小写与框架一致
                fix = str.lower if ins0.islower() else str.upper
                v.index = v.index.set_levels(v.index.levels[v.index.names.index("instrument")].map(fix),
                                             level="instrument")
                ev.add_values(name, v.groupby(level=list(range(v.index.nlevels))).last())
            s = pd.Series(ev.rk[name], index=ev.idx)[train]
            ic, _ = ev.ic(s)
            scored.append((abs(ic) if np.isfinite(ic) else 0.0, name))
        scored.sort(reverse=True)
        return [{"add": [n], "desc": n} for _, n in scored[: self.K] if n not in st["names"]]


class GPMethod:
    """遗传规划：每轮从随机公式出发进化，取适应度最高且互不相近的 K 个公式作为候选。"""
    K = 20

    def __init__(self, fr):
        self.fr = fr
        self.g = None

    def init_state(self):
        return {"names": list(BASELINE), "model": "ridge"}

    def propose(self, st, r, rng):
        if self.g is None:
            log("  构建 GP 宽表……")
            self.g = gpmod.GP(self.fr.ev.panel, self.fr.ev.label, rng, fit_end=self.fr.fit_end, pop=self.fr.a.gp_pop, gens=self.fr.a.gp_gens)
        self.g.rng = rng
        best = self.g.run(self.K)
        moves = []
        for i, (t, fit) in enumerate(best):
            name = f"gp_s{self.fr.a.seed}_r{r}_{i}"
            w = self.g.values(t).replace([np.inf, -np.inf], np.nan)
            try:
                vals = w.stack(future_stack=True)
            except TypeError:                              # pandas < 2.1
                vals = w.stack(dropna=False)
            vals.index = vals.index.swaplevel()
            vals.index.names = ["instrument", "datetime"]
            self.fr.ev.add_values(name, vals.astype("float64"))
            self.fr.codes[name] = gpmod.to_code(name, t)
            moves.append({"add": [name], "desc": f"{name} fit={fit:.4f} {gpmod.to_str(t)[:120]}"})
        return moves


class Null(Classic):
    """无效因子（检验安全性）：把类 Alpha158 特征在每天的截面上随机打乱，信号被完全破坏，
    但分布、缺失位置都不变。任何被接受的批次都是“误入池”。"""

    def __init__(self, fr):
        self.fr = fr
        ev = fr.ev
        rng = np.random.default_rng(12345)
        groups = pd.Series(np.arange(len(ev.idx))).groupby(ev.dates).indices
        self.ic = {}
        for n in LIB:
            src = ev.rk[n]
            v = src.copy()
            for ix in groups.values():
                v[ix] = src[rng.permutation(ix)]
            ev.rk["null_" + n] = v
            ev.vals["null_" + n] = 1
            self.ic["null_" + n] = 0.0

    def propose(self, st, r, rng):
        left = [n for n in self.ic if n not in st["names"]]
        pick = list(rng.choice(left, size=min(self.K, len(left)), replace=False))
        return [{"add": [n], "desc": n} for n in pick]


class OrchMethod:
    """大模型智能体编排（CogAlpha + 诊断专家 / 总指挥 / 主题子代理，见 orchestra.py）作为一个改进算子。
    每轮：诊断专家看当前因子池在 t-1 年（大模型可见的最近一年）的表现 → 总指挥制定计划 → 子代理按计划挖掘 / 进化，
    产出的新精英因子作为候选交给统一框架的入池规则。主题子代理的单因子打分沿用 CogAlpha 的 2011-2019 训练期，
    不会用到 t-1 年之后的数据。"""

    def __init__(self, fr):
        import argparse
        import cogalpha
        import orchestra as O
        a = fr.a
        oa = argparse.Namespace(
            run=f"online_orch_{a.rule}_s{a.seed}", seed_pool=a.seed_pool, rounds=6, budget=a.budget,
            max_tasks=a.max_tasks, max_gens=a.max_gens, targets=a.targets, dup=0.7, min_gain=a.min_gain,
            max_prune=0, admit="robust", prescreen="pooled", alpha=a.alpha, boot=a.boot, block=a.block,
            n0=8, parents=6, inject=3, min_gens=99, workers=8, procs=a.procs, seed=a.seed, data=a.data,
            cache=a.cache, full_cache=a.full_cache, mock=a.mock, agents="all")
        if a.mock:
            cogalpha.chat = O.mock_chat
        O.load_panel(a.data, cache=a.cache)
        import shutil
        shutil.rmtree(Path("runs") / oa.run, ignore_errors=True)   # 每次从头开始，避免读到上次的因子池
        self.O, self.fr = O, fr
        self.orch = O.Orchestra(oa)
        self.factors = {f["name"]: f for f in self.orch.pool_f}
        fr.ev.compute(self.orch.pool_f)
        self.orch.state.update({"dev": [], "round": 0})
        self.orch.update_map()

    def init_state(self):
        return {"names": [f["name"] for f in self.orch.pool_f], "model": "ridge"}

    def _dev_window(self, t):
        ev = self.orch.ev
        ev.dev, ev.train_end_dev = (f"{t - 1}-01-01", f"{t - 1}-12-31"), f"{t - 2}-12-15"

    def propose(self, st, r, rng):
        o, O = self.orch, self.O
        t = self.fr.decision_year
        o.pool_f = [self.factors[n] for n in st["names"] if n in self.factors]
        o.state["round"] = r - 1
        o.budget_end = o.calls + self.fr.a.budget
        self._dev_window(t)
        O.DIAGNOSE_YEAR = t
        o.ev.compute(o.pool_f)
        dev, _ = o.ev.evaluate([f["name"] for f in o.pool_f])
        o.state["dev"].append(dev)
        tab = o.ev.factor_table(o.pool_f)
        orig = O.DIAGNOSE
        O.DIAGNOSE = orig.replace("TRAIN 2011-2019 and VALIDATION 2020", f"TRAIN 2011-{t - 2} and VALIDATION {t - 1}") \
                         .replace("validation 2020", f"validation {t - 1}")
        try:
            log(f"  [orch] 诊断专家分析因子池（验证年 {t - 1}）……")
            th = o.diagnose(tab, dev)
            log("  [orch] 总指挥制定计划……")
            plan = o.plan(th)
        finally:
            O.DIAGNOSE = orig
        added = []
        for task in plan["tasks"]:
            if o.over():
                break
            log(f"  [orch] 任务 {task['agent']} | {task['mode']} | {task['gens']} 代")
            for e in o.run_task(task):
                if e["name"] in self.factors or any(e["name"] == x["name"] for x in added):
                    continue
                e = {k: v for k, v in e.items() if k in ("name", "code", "agent", "gen", "train")}
                e["round"] = r
                added.append(e)
        for e in added:
            self.factors[e["name"]] = e
        self.fr.ev.compute(added)
        added.sort(key=lambda x: -abs(x["train"]["RankICIR"]))
        self.calls_this_round = o.calls - (o.budget_end - self.fr.a.budget)
        log(f"  [orch] 本轮候选 {len(added)} 个，大模型调用 {self.calls_this_round} 次")
        return [{"add": [e["name"]], "desc": e["name"]} for e in added if e["name"] in self.fr.ev.rk]

    def feedback(self, r, kept, accepted, cert):
        """只把“保留了几个、确认是否通过”写回工作区（大模型看不到任何确认期或测试期的数字）。"""
        note = "" if cert is None else (" | held-out confirmation: " + ("PASSED" if cert["passed"] else "FAILED (batch discarded)"))
        self.orch.write("exp_log.md", f"- round {r} | kept {len(kept)} candidates by validation score; "
                        f"{'accepted' if accepted else 'not accepted'}{note}", append=True)
        self.orch.update_map()


REFLECT_GEN = """You are a quantitative researcher mining alpha factors for China A-share CSI300 stocks (daily data).
Goal: factors that predict next-day cross-sectional returns and add information beyond the current pool.

### Current factor pool
{pool}

### Lessons from your previous rounds (your own reflections and outcomes)
{memory}

Please generate {n} new, original factor functions that differ from the pool.

""" + prompts.COMMON_RULES

REFLECT_REV = """You are a quantitative researcher mining alpha factors for China A-share CSI300 stocks (daily data).
Below are the factors you wrote in this round, with their results on the TRAINING period only
(daily cross-sectional RankIC and RankICIR; the sign does not matter, the model learns the direction;
|RankICIR| >= 0.3 is good, < 0.1 is weak). Failed factors show the error.

{results}

Step 1. Reflect: in at most 5 bullet points, say what worked, what failed and why, and what to try next.
Put the reflection between <<reflection>> and <</reflection>>.
Step 2. Based on your reflection, write {n} revised or new factor functions: fix the failed ones, strengthen
or replace the weak ones, and explore variations of the strong ones that are not near-copies.

""" + prompts.COMMON_RULES


class ReflectMethod:
    """反思式因子挖掘（Reflexion 风格的单智能体）：大模型先写一批因子公式 → 程序在训练期（≤ fit_end）打分 →
    大模型读到分数和报错后写“反思”，据此修改 / 重写因子，重复 T 次；本轮所有通过检查的因子按训练期 |RankICIR|
    取前 K 个作为候选。跨轮记忆只包含它自己的反思和“保留了几个、确认是否通过”，看不到确认期或部署期的任何数字。"""
    K, N, T = 20, 8, 3

    def __init__(self, fr):
        import cogalpha
        import orchestra as O
        self.fr, self.cg = fr, cogalpha
        if fr.a.mock:
            cogalpha.chat = O.mock_chat
        self.info, self.memory, self.calls = {}, [], 0

    def init_state(self):
        return {"names": list(BASELINE), "model": "ridge"}

    def _chat(self, prompt):
        self.calls += 1
        for k in range(3):
            try:
                return self.cg.chat(prompt)
            except Exception as e:                              # 网络错误重试，三次都失败就返回空
                log(f"  [reflect] 大模型调用失败（{k + 1}/3）：{str(e)[:80]}")
                time.sleep(10)
        return ""

    def _pool_text(self, st):
        base = [n for n in st["names"] if n in BASELINE]
        lines = [f"- baseline factors: {', '.join(base)}"] if base else []
        for n in st["names"]:
            if n in self.info:
                i = self.info[n]
                lines.append(f"- {i['fname']}: {i['doc']} (train RankIC {i['ric']:+.4f}, RankICIR {i['icir']:+.3f})")
        return "\n".join(lines) or "(empty)"

    def _evaluate(self, codes, r, it, train):
        """检查 + 计算 + 训练期打分。返回 [(结果字典)]，失败的带 error。"""
        from checker import compile_factor, run_check
        ev, out, todo = self.fr.ev, [], []
        for j, code in enumerate(codes):
            fname, func, err = compile_factor(code)
            if err is None:
                err = run_check(func, ev.panel)
            name = f"refl_s{self.fr.a.seed}_r{r}_{it}_{j}"
            doc = (func.__doc__ or "").strip().split("\n")[0][:150] if func else ""
            out.append({"name": name, "fname": fname or "?", "code": code, "doc": doc, "error": err})
            if err is None:
                todo.append({"name": name, "code": code})
        if todo:
            ev.compute(todo)
        for o in out:
            if o["error"] is None and o["name"] not in ev.rk:
                o["error"] = "computation failed on the full panel"
            if o["error"] is None:
                ric, icir = ev.ic(pd.Series(ev.rk[o["name"]], index=ev.idx)[train])
                if not np.isfinite(ric):
                    o["error"] = "factor is (almost) constant"
                else:
                    o["ric"], o["icir"] = float(ric), float(icir)
        return out

    def propose(self, st, r, rng):
        from llm import parse_functions
        ev = self.fr.ev
        train = np.asarray(ev.idx.get_level_values("datetime") <= pd.Timestamp(self.fr.fit_end))
        c0 = self.calls
        mem = "\n".join(self.memory[-6:]) or "(first round, no lessons yet)"
        text = self._chat(REFLECT_GEN.format(pool=self._pool_text(st), memory=mem, n=self.N))
        done, refl = [], ""
        for it in range(self.T):
            res = self._evaluate(parse_functions(text), r, it, train)
            done += res
            ok = [o for o in res if "ric" in o]
            log(f"  [reflect] 第 {it + 1}/{self.T} 次：写出 {len(res)} 个，通过检查 {len(ok)} 个，"
                + (f"最好 |RankICIR| {max(abs(o['icir']) for o in ok):.3f}" if ok else "无有效因子"))
            if it == self.T - 1:
                break
            rep = "\n\n".join(
                f"### {o['fname']}: " + (f"RankIC {o['ric']:+.4f}, RankICIR {o['icir']:+.3f}" if "ric" in o
                                          else f"FAILED: {o['error']}") + f"\n{o['code']}" for o in res) or "(no parsable factor)"
            text = self._chat(REFLECT_REV.format(results=rep, n=self.N))
            m = re.search(r"<<\s*reflection\s*>>(.*?)<<\s*/\s*reflection\s*>>", text, flags=re.S)
            if m:
                refl = m.group(1).strip()
        if refl:
            self.memory.append(f"- round {r} reflection: " + " ".join(refl.split())[:600])
        good = sorted([o for o in done if "ric" in o], key=lambda o: -abs(o["icir"]))
        for o in good:
            self.info[o["name"]] = o
            self.fr.codes[o["name"]] = o["code"]
        self.calls_this_round = self.calls - c0
        log(f"  [reflect] 本轮有效因子 {len(good)} 个，大模型调用 {self.calls_this_round} 次")
        return [{"add": [o["name"]], "desc": f"{o['name']} {o['fname']} icir={o['icir']:+.3f}"}
                for o in good[: self.K] if o["name"] not in st["names"]]

    def feedback(self, r, kept, accepted, cert):
        note = "" if cert is None else (", held-out confirmation " + ("PASSED" if cert["passed"] else "FAILED"))
        self.memory.append(f"- round {r} outcome: {len(kept)} of your factors improved the validation score; "
                           f"batch {'accepted into the pool' if accepted else 'not accepted'}{note}")


METHODS = {"classic": Classic, "ml": ML, "gp": GPMethod, "null": Null, "orch": OrchMethod, "reflect": ReflectMethod,
           "ridge": RidgeM, "mlp": MLPM, "transformer": TransformerM,
           "gat": GATM, "tra": TRAM, "adarnn": AdaRNNM,
           "lasso": LassoM, "elasticnet": ElasticNetM, "lightgbm": LightGBMM, "xgboost": XGBoostM,
           "lstm": LSTMM, "gru": GRUM, "catboost": CatBoostM, "rf": RFM,
           "alstm": ALSTMM, "doubleensemble": DoubleEnsembleM, "rdagent": RDAgentMethod}


# ============================ 统一循环 ============================
class Frame:
    def __init__(self, a):
        self.a = a
        self.cert = a.admit == "certified"
        self.fit_end = "2017-12-15"
        if self.cert and a.conf_years == 2:                # 推论：确认窗口加长一倍，检验力更高
            orchestra.CONF, orchestra.TRAIN_END_CONF = ("2019-01-16", "2020-12-31"), "2018-12-15"
            orchestra.CERT_SEL, orchestra.TRAIN_END_SEL = ("2017-01-01", "2018-12-31"), "2016-12-15"
            orchestra.CERT_DEV, orchestra.CERT_TRAIN_END_DEV = ("2018-01-01", "2018-12-31"), "2017-12-15"
            self.fit_end = "2016-12-15"
        self.tag = (f"{a.admit}{'-rsi' if self.cert and a.cert_on == 'rsi' else ''}"
                    f"{'2y' if self.cert and a.conf_years == 2 else ''}")
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.file = self.out / f"{a.method}_{self.tag}_s{a.seed}.json"
        t = [float(x) for x in a.targets.split(",")]
        self.targets = {k: t[i] for i, k in enumerate(METRICS)}
        log(f"加载面板（{'可认证' if self.cert else '只看验证分'}模式）……")
        self.ev = FrameEval(a.data, a.procs, a.full_cache, certified=self.cert)
        self.codes = {}
        self.load_lib(a.method)
        self.m = METHODS[a.method](self)

    def load_lib(self, method):
        need = list(LIB) if method in ("classic", "ml", "null", "ridge", "mlp") else []
        need += list(BASELINE)
        cache = Path(self.a.libcache)
        if cache.exists():
            z = np.load(cache)
            for n in z.files:
                self.ev.rk[n] = z[n]
                self.ev.vals[n] = 1
        todo = [n for n in need if n not in self.ev.rk]
        if todo:
            log(f"计算 {len(todo)} 个库因子（只需一次，结果缓存到 {cache}）……")
            for i in range(0, len(todo), 12):
                part = todo[i:i + 12]
                self.ev.compute([{"name": n, "code": BASELINE.get(n) and n or LIB[n]} for n in part])
                log(f"  {min(i + 12, len(todo))}/{len(todo)}")
            np.savez(cache, **self.ev.rk)

    def lib_train_ic(self):
        path = Path(self.a.libcache).with_suffix(f".ic{self.fit_end[:4]}.json")
        if path.exists():
            return json.loads(path.read_text())
        m = self.ev.mask("2011-01-01", self.fit_end)
        days = pd.DatetimeIndex(sorted(set(self.ev.dates[m])))[::5]
        sm = np.isin(self.ev.dates, days)
        lab = self.ev.label[sm]
        ic = {}
        for n in LIB:
            s = pd.Series(self.ev.rk[n][sm], index=self.ev.idx[sm])
            ic[n] = float(bt.daily_rank_ic(s, lab).mean())
        path.write_text(json.dumps(ic))
        return ic

    def span(self, k, base):
        s0, tgt = base[k], self.targets[k]
        if tgt <= s0:
            tgt = s0 + max(abs(s0) * 0.5, 1e-3)
        return tgt - s0

    def rsi(self, off, base, keys=None):
        return float(np.mean([(off[k] - base[k]) / self.span(k, base) for k in (keys or METRICS)]))

    def run(self):
        a = self.a
        rng = np.random.default_rng(a.seed)
        S = self.m.init_state()
        t0 = time.time()
        dev, off = self.ev.evaluate(S)
        sel0, _ = self.ev.sel_score(S)
        rec = {"method": a.method, "admit": self.tag, "seed": a.seed, "alpha": a.alpha,
               "rounds": [{"round": 0, "dev": dev, "off": off, "sel": sel0, "n": len(S["names"]),
                           "rsi": 0.0, "rsi3": 0.0, "lcb_rsi3": 0.0, "minutes": round((time.time() - t0) / 60, 2)}]}
        base = off
        lcb_acc = {k: 0.0 for k in CONF_METRICS}
        lcb_rsi = 0.0
        log(f"第 0 轮：{len(S['names'])} 个特征，测试期 RankIC {off['test_ric']:.4f}，选择窗口分数 {sel0:.4f}")
        for r in range(1, a.rounds + 1):
            t0 = time.time()
            moves = self.m.propose(S, r, rng)
            base_sc, _ = self.ev.sel_score(S)
            cur, kept, cur_sc = S, [], base_sc
            for mv in moves:                               # 入池规则 A 第一步：贪心，只看选择窗口分数
                cand = {"names": cur["names"] + mv["add"], "model": cur["model"]} if "add" in mv else mv["replace"]
                sc, _ = self.ev.sel_score(cand)
                if sc > cur_sc + a.min_gain:
                    cur, cur_sc = cand, sc
                    kept.append(mv["desc"])
            cert = None
            if self.cert and kept and a.cert_on == "rsi":   # 第二步（RSI 版）：直接认证 RSI 指数，M = 1
                alpha_r = a.alpha * 2.0 ** (-r) / (2.0 ** (r - 1))
                w = [1.0 / self.span(CONF_METRICS[k], base) for k in CONF_METRICS]
                cert = self.ev.confirm_rsi(S, cur, alpha_r, w, a.boot, a.block, a.seed * 100 + r)
                if cert["passed"]:
                    lcb_rsi += cert["lcb_rsi"]
            elif self.cert and kept:                        # 第二步：确认窗口上的置信下界（整批，逐项）
                M = len(CONF_METRICS)
                alpha_r = a.alpha * 2.0 ** (-r) / (M * 2.0 ** (r - 1))
                cert = self.ev.confirm(S, cur, alpha_r, a.boot, a.block, a.seed * 100 + r)
                if cert["passed"]:
                    for k in CONF_METRICS:
                        lcb_acc[k] += cert["lcb_" + k]
            accepted = bool(kept) and (cert is None or cert["passed"])
            if accepted:
                S = cur
            dev, off = self.ev.evaluate(S)
            rsi3 = self.rsi(off, base, list(CONF_METRICS.values()))
            lcb3 = lcb_rsi if a.cert_on == "rsi" else \
                float(np.mean([lcb_acc[k] / self.span(CONF_METRICS[k], base) for k in CONF_METRICS]))
            row = {"round": r, "dev": dev, "off": off, "sel": cur_sc if accepted else base_sc,
                   "sel_gain_proposed": cur_sc - base_sc, "candidates": len(moves), "kept": kept,
                   "accepted": accepted, "cert": cert, "n": len(S["names"]),
                   "rsi": self.rsi(off, base), "rsi3": rsi3, "lcb_rsi3": lcb3,
                   "minutes": round((time.time() - t0) / 60, 2)}
            rec["rounds"].append(row)
            log(f"第 {r} 轮：候选 {len(moves)}，初筛保留 {len(kept)}，选择窗口分数 {base_sc:.4f}→{cur_sc:.4f}"
                + (f"，确认检验{'通过' if cert['passed'] else '未通过'}" if cert else "")
                + f"；测试期 RankIC {off['test_ric']:.4f}，RSI {row['rsi']:+.4f}，RSI3 {rsi3:+.4f}（下界 {lcb3:+.4f}），"
                f"{row['minutes']} 分钟")
            self.file.write_text(json.dumps({**rec, "state": S, "codes": self.codes}, indent=1, default=str))
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHODS), required=True)
    ap.add_argument("--conf_years", type=int, choices=[1, 2], default=1,
                    help="确认窗口长度：1 = 2020；2 = 2019-2020（选择窗口相应前移到 2017-2018）")
    ap.add_argument("--admit", choices=["naive", "certified"], default="certified",
                    help="naive = 只看验证分数（2018-2020）；certified = 2018-2019 初筛 + 2020 确认窗口置信下界")
    ap.add_argument("--cert_on", choices=["each", "rsi"], default="each",
                    help="each = 3 项指标逐项认证；rsi = 直接认证 RSI 指数（合成统计量，检验力更高）")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--min_gain", type=float, default=0.0003)
    ap.add_argument("--gp_pop", type=int, default=80)
    ap.add_argument("--gp_gens", type=int, default=6)
    ap.add_argument("--targets", default=",".join(str(v[1]) for v in METRICS.values()))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--data", default="./qlib_bin")
    ap.add_argument("--full_cache", default="./csi300_panel_full.pkl")
    ap.add_argument("--libcache", default="./runs/frame/lib_values.npz")
    ap.add_argument("--out", default="./runs/frame")
    a = ap.parse_args()
    Frame(a).run()


if __name__ == "__main__":
    main()
