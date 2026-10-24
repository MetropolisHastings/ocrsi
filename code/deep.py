"""深度序列模型：给统一框架用的训练 / 预测函数（Transformer、GATs、TRA、AdaRNN）。

输入（类 Qlib Alpha360）：每只股票过去 seq_len 天的 开/高/低/收 ÷ 当天收盘价 − 1、log(成交量 ÷ 当天成交量)，
共 5 个特征 × seq_len 天；缺失记 0。标签与其他方法相同：未来 10 日收益的截面排名。
需要 PyTorch；有 GPU 时自动用 GPU。

    Transformer  编码器取最后一个时间步
    GATs         GRU 编码 → 每天截面上的图注意力（Qlib GATs 的做法：注意力加权的邻居表示 + 自身表示）
    TRA          GRU 编码 → K 个预测头 + 路由器；训练时用 Sinkhorn 最优传输按各头损失分配样本（Lin et al., KDD 2021）
                 简化：路由器只用隐状态，不用各头的历史误差
    AdaRNN       时序分布刻画：把训练期切成分布差异最大的几段；时序分布匹配：训练时让各段的隐状态均值对齐
                 （Du et al., CIKM 2021；对齐距离简化为隐状态均值的平方距离）

自检（不需要真实数据，几秒钟）：python deep.py --selftest
"""
import itertools

import numpy as np
import pandas as pd

_CACHE = {}
DEEP_KINDS = ("transformer", "gat", "tra", "adarnn")


def _wide(ev):
    """把面板变成 日期 × 股票 的宽表，并记下 ev.idx 每一行在宽表里的位置（每个 ev 只算一次）。"""
    key = id(ev)
    if key in _CACHE:
        return _CACHE[key]
    P = ev.panel
    W = {c: P[c].unstack("instrument").astype("float32") for c in ["open", "high", "low", "close", "volume"]}
    dates, insts = W["close"].index, W["close"].columns
    di = dates.get_indexer(ev.idx.get_level_values("datetime"))
    ii = insts.get_indexer(ev.idx.get_level_values("instrument"))
    arr = {c: W[c].values for c in W}
    _CACHE[key] = (arr, di, ii)
    return _CACHE[key]


def seq_features(ev, rows, T):
    """rows：ev.idx 上的布尔掩码。返回 (n, T, 5) 的 float32 数组，时间从早到晚，最后一步是当天。"""
    arr, di, ii = _wide(ev)
    d, i = di[rows], ii[rows]
    lag = np.arange(T - 1, -1, -1)
    D = d[:, None] - lag[None, :]
    ok = D >= 0
    D = np.where(ok, D, 0)
    I = np.broadcast_to(i[:, None], D.shape)
    c0 = arr["close"][d, i][:, None]
    v0 = arr["volume"][d, i][:, None]
    feats = []
    with np.errstate(all="ignore"):
        for c in ["open", "high", "low", "close"]:
            feats.append(arr[c][D, I] / c0 - 1.0)
        feats.append(np.log((arr["volume"][D, I] + 1.0) / (v0 + 1.0)))
    X = np.stack(feats, axis=-1).astype("float32")
    X[~ok] = 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -5, 5)


def _tdc_split(ev, tr, K):
    """AdaRNN 的时序分布刻画（简化）：在训练日上，从十分位候选切点里选 K-1 个，
    使各段“每日截面均值特征”的段均值两两距离之和最大。返回每个训练样本的段编号。"""
    arr, di, _ = _wide(ev)
    days = np.unique(di[tr])
    c = arr["close"][days]
    r = np.nanmean(np.log(c[1:] / c[:-1]), axis=1)
    vol = np.nanstd(np.log(c[1:] / c[:-1]), axis=1)
    f = np.nan_to_num(np.c_[r, vol])
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    n = len(f)
    cand = [int(n * q / 10) for q in range(1, 10)]
    best, best_cut = -1.0, None
    for cut in itertools.combinations(cand, K - 1):
        seg = np.split(f, cut)
        if min(len(s) for s in seg) < 20:
            continue
        mu = [s.mean(0) for s in seg]
        dsum = sum(np.sum((a - b) ** 2) for a, b in itertools.combinations(mu, 2))
        if dsum > best:
            best, best_cut = dsum, cut
    edges = [days[1:][c] for c in (best_cut or [])]
    return np.searchsorted(np.array(edges), di[tr], side="right")


def fit_predict(ev, p, tr, te, kind="transformer", seed=0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = int(p["seq_len"])
    drop = float(p.get("dropout", 0.0))

    class GRUEnc(nn.Module):
        def __init__(self, h, layers=1):
            super().__init__()
            self.rnn = nn.GRU(5, h, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0.0)

        def forward(self, x):
            return self.rnn(x)[0][:, -1]

    class TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = int(p["d_model"])
            self.inp = nn.Linear(5, d)
            self.pos = nn.Parameter(torch.zeros(1, T, d))
            layer = nn.TransformerEncoderLayer(d, int(p["nhead"]), dim_feedforward=4 * d, dropout=drop,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, int(p["nlayers"]))
            self.out = nn.Linear(d, 1)

        def forward(self, x):
            return self.out(self.enc(self.inp(x) + self.pos)[:, -1]).squeeze(-1)

    class GATNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = int(p["hidden"])
            self.enc = GRUEnc(h, int(p["layers"]))
            self.W = nn.Linear(h, h)
            self.a1, self.a2 = nn.Linear(h, 1), nn.Linear(h, 1)
            self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(h, 1))

        def forward(self, x):                            # x：同一天的全部股票 (N, T, 5)
            z = self.enc(x)
            wz = self.W(z)
            e = nn.functional.leaky_relu(self.a1(wz) + self.a2(wz).T, 0.2)   # (N, N)
            att = torch.softmax(e, dim=1)
            return self.out(att @ wz + z).squeeze(-1)

    class TRANet(nn.Module):
        def __init__(self):
            super().__init__()
            h, k = int(p["hidden"]), int(p["heads"])
            self.enc = GRUEnc(h)
            self.heads = nn.Linear(h, k)
            self.router = nn.Linear(h, k)

        def forward(self, x):
            z = self.enc(x)
            return self.heads(z), self.router(z)

    class AdaNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = int(p["hidden"])
            self.enc = GRUEnc(h, 2)
            self.out = nn.Linear(h, 1)

        def forward(self, x):
            z = self.enc(x)
            return self.out(z).squeeze(-1), z

    class RNNNet(nn.Module):                           # LSTM / GRU：Qlib 基准里最常用的时序模型
        def __init__(self):
            super().__init__()
            h, L = int(p["hidden"]), int(p["layers"])
            cell = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = cell(5, h, num_layers=L, batch_first=True, dropout=drop if L > 1 else 0.0)
            self.out = nn.Linear(h, 1)

        def forward(self, x):
            return self.out(self.rnn(x)[0][:, -1]).squeeze(-1)

    class ALSTMNet(nn.Module):                         # ALSTM：LSTM + 时间注意力（Qlib 基准）
        def __init__(self):
            super().__init__()
            h, L = int(p["hidden"]), int(p["layers"])
            self.rnn = nn.LSTM(5, h, num_layers=L, batch_first=True, dropout=drop if L > 1 else 0.0)
            self.att = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1, bias=False))
            self.out = nn.Linear(2 * h, 1)

        def forward(self, x):
            H = self.rnn(x)[0]                             # (B, T, h)
            a = torch.softmax(self.att(H), dim=1)          # 每个时间步的注意力权重
            return self.out(torch.cat([(a * H).sum(1), H[:, -1]], 1)).squeeze(-1)

    def sinkhorn(Q, iters=3):
        Q = torch.exp(Q - Q.max())
        for _ in range(iters):
            Q = Q / Q.sum(0, keepdim=True)
            Q = Q / Q.sum(1, keepdim=True)
        return Q

    net = {"transformer": TransformerNet, "gat": GATNet, "tra": TRANet, "adarnn": AdaNet, "lstm": RNNNet, "gru": RNNNet, "alstm": ALSTMNet}[kind]().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=float(p["lr"]))
    Xtr = torch.from_numpy(seq_features(ev, tr, T))
    ytr = torch.from_numpy(ev.y.values[tr].astype("float32"))
    epochs = int(p["epochs"])

    def step(loss):
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()

    net.train()
    if kind == "gat":                                   # 每天一个批次（图建在当天的截面上）
        dtr = np.asarray(ev.dates[tr])
        groups = [np.where(dtr == d)[0] for d in np.unique(dtr)]
        for _ in range(epochs):
            for gi in np.random.permutation(len(groups)):
                g = torch.from_numpy(groups[gi])
                xb, yb = Xtr[g].to(dev), ytr[g].to(dev)
                step(((net(xb) - yb) ** 2).mean())
    else:
        seg = torch.from_numpy(_tdc_split(ev, tr, int(p["periods"]))).long() if kind == "adarnn" else None
        n, bs = len(ytr), 2048
        for _ in range(epochs):
            perm = torch.randperm(n)
            for k in range(0, n, bs):
                b = perm[k:k + bs]
                xb, yb = Xtr[b].to(dev), ytr[b].to(dev)
                if kind in ("transformer", "lstm", "gru", "alstm"):
                    loss = ((net(xb) - yb) ** 2).mean()
                elif kind == "tra":
                    preds, logits = net(xb)
                    L = (preds - yb[:, None]) ** 2
                    P = sinkhorn(-L.detach() / float(p["tau"]))          # 最优传输：样本均衡地分给损失小的头
                    loss = (P * L).sum(1).mean() - float(p["rho"]) * (P * torch.log_softmax(logits, 1)).sum(1).mean()
                else:
                    pred, z = net(xb)
                    loss = ((pred - yb) ** 2).mean()
                    sb = seg[b].to(dev)
                    mus = [z[sb == s].mean(0) for s in torch.unique(sb) if (sb == s).sum() > 8]
                    if len(mus) > 1:
                        loss = loss + float(p["lam"]) * sum(((a - c) ** 2).sum() for a, c in itertools.combinations(mus, 2))
                step(loss)
    net.eval()
    idx = np.where(te)[0]
    out = np.empty(len(idx), dtype="float32")
    with torch.no_grad():
        if kind == "gat":
            dte = np.asarray(ev.dates[te])
            for d in np.unique(dte):
                pos = np.where(dte == d)[0]
                m = np.zeros(len(te), dtype=bool)
                m[idx[pos]] = True
                out[pos] = net(torch.from_numpy(seq_features(ev, m, T)).to(dev)).float().cpu().numpy()
        else:
            for k in range(0, len(idx), 20000):
                m = np.zeros(len(te), dtype=bool)
                m[idx[k:k + 20000]] = True
                xb = torch.from_numpy(seq_features(ev, m, T)).to(dev)
                if kind == "tra":
                    preds, logits = net(xb)
                    yhat = (torch.softmax(logits, 1) * preds).sum(1)
                elif kind == "adarnn":
                    yhat = net(xb)[0]
                else:
                    yhat = net(xb)
                out[k:k + 20000] = yhat.float().cpu().numpy()
    return out


def _selftest():
    """用 30 只股票 × 400 天的随机数据把四个模型各训练 1 轮，检查能跑通、输出没有缺失值。"""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=400)
    insts = [f"S{i:03d}" for i in range(30)]
    ix = pd.MultiIndex.from_product([insts, dates], names=["instrument", "datetime"])
    c = np.exp(np.cumsum(rng.normal(0, 0.02, (30, 400)), axis=1)).ravel() * 10
    panel = pd.DataFrame({"open": c * (1 + rng.normal(0, .005, c.size)), "high": c * 1.01, "low": c * 0.99,
                          "close": c, "volume": rng.lognormal(10, 1, c.size)}, index=ix)

    class EV:
        pass
    ev = EV()
    ev.panel, ev.idx = panel, ix
    ev.dates = ix.get_level_values("datetime")
    ev.y = pd.Series(rng.uniform(-.5, .5, len(ix)), index=ix)
    tr = np.asarray(ev.dates < dates[300]) & (np.arange(len(ix)) % 3 == 0)
    te = np.asarray(ev.dates >= dates[320])
    base = {"seq_len": 10, "lr": 1e-3, "epochs": 1, "dropout": 0.1}
    cfg = {"transformer": {"d_model": 16, "nhead": 2, "nlayers": 1}, "gat": {"hidden": 16, "layers": 1},
           "tra": {"hidden": 16, "heads": 3, "tau": 1.0, "rho": 0.1}, "adarnn": {"hidden": 16, "periods": 2, "lam": 0.5},
           "lstm": {"hidden": 16, "layers": 1}, "gru": {"hidden": 16, "layers": 2},
           "alstm": {"hidden": 16, "layers": 1}}
    for k, extra in cfg.items():
        out = fit_predict(ev, {**base, **extra}, tr, te, kind=k)
        print(f"{k:12s} 输出 {out.shape}，缺失 {int(np.isnan(out).sum())}，标准差 {out.std():.4f}  OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
