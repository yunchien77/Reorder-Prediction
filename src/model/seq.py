"""Sequence models (LSTM / GRU / Transformer encoder) on each (user, product) pair's purchase history.  PyTorch, GPU.

Input sequence of a pair = the user's last T=24 PRIOR orders (oldest -> newest; missing early steps are padded with valid = 0), per step:
    product features : relative cart position of the product in that order (0 = not bought)  +  bought flag
    order features   : valid, days since the previous order (/30), sin/cos of day-of-week and hour, log basket size
Nothing about the labelled order is used (the target day / hour are only in the tabular branch, as in the 54 features).
Two kinds of model:
    hybrid  : sequence encoder output  +  MLP branch on the 54 tabular features  ->  MLP head        (name: lstm_hybrid, transformer_hybrid ...)
    seq-only: sequence encoder output only (can the network learn the hand-made lag / decay / gap features by itself?)   (name: lstm_seq ...)
Training as in model/mlp.py: BCE, Adam, up to 10 epochs, keeps the epoch with the best val AUPRC.
    cd src && python -m model.seq --kind lstm gru transformer --folds 1 2 3 4 5 [--seq-only]"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

import common
from external.names import RAW
from model import mlp

T, CTX = 24, 7
P = 49689


def history():
    o = pd.read_csv(RAW / "orders.csv", usecols=["order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"])
    o = o[o.eval_set == "prior"].copy()                                                # prior orders only
    o["back"] = o.groupby("user_id").order_number.transform("max") - o.order_number    # 0 = last prior order
    op = pd.read_csv(RAW / "order_products__prior.csv", usecols=["order_id", "product_id", "add_to_cart_order"])
    bs = op.groupby("order_id").size().rename("bsize")
    o = o.join(bs, on="order_id")
    op = op.merge(o[["order_id", "user_id", "back", "bsize"]], on="order_id")
    op["rel"] = (op.add_to_cart_order / op.bsize).astype(np.float32)
    return o[o.back < T], op[op.back < T]


class Hist:
    """Per-user order context [U, T, 7] and per-pair product sequences [N, T] (float16 relative cart position, 0 = not bought)."""

    def __init__(self, users):
        self.uidx = np.full(users.max() + 1, -1, np.int64); self.uidx[users] = np.arange(len(users))
        o, op = history()
        o = o[o.user_id.isin(users)]
        ctx = np.zeros((len(users), T, CTX), np.float32)
        u, t = self.uidx[o.user_id.values], T - 1 - o.back.values
        ang = lambda x, m: 2 * np.pi * x.values / m
        ctx[u, t] = np.stack([np.ones(len(o)), o.days_since_prior_order.fillna(0).values / 30, np.sin(ang(o.order_dow, 7)), np.cos(ang(o.order_dow, 7)),
                              np.sin(ang(o.order_hour_of_day, 24)), np.cos(ang(o.order_hour_of_day, 24)), np.log1p(o.bsize.values) / 4], 1)
        self.ctx = ctx
        op = op[op.user_id.isin(users)]
        self.keys = []
        for b in range(T):
            x = op[op.back == b]
            k = self.uidx[x.user_id.values] * P + x.product_id.values.astype(np.int64)
            order = np.argsort(k)
            self.keys.append((k[order], x.rel.values[order]))

    def pairs(self, user_id, product_id):
        pu = self.uidx[user_id]; q = pu * P + product_id.astype(np.int64)
        out = np.zeros((len(q), T), np.float16)
        for b, (k, rel) in enumerate(self.keys):
            if len(k):
                pos = np.minimum(np.searchsorted(k, q), len(k) - 1)
                out[:, T - 1 - b] = np.where(k[pos] == q, rel[pos], 0)
        return pu, out


class Net(nn.Module):
    def __init__(self, kind, d_tab, hidden=64):
        super().__init__()
        self.kind = kind
        din = 2 + CTX
        if kind == "transformer":
            self.inp = nn.Linear(din, hidden); self.pos = nn.Parameter(torch.zeros(1, T, hidden))
            layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, 0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False); self.dim = 2 * hidden
        else:
            self.enc = (nn.LSTM if kind == "lstm" else nn.GRU)(din, hidden, batch_first=True); self.dim = hidden
        self.tab = None
        if d_tab:
            self.tab = nn.Sequential(nn.Linear(d_tab, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(self.dim + (64 if d_tab else 0), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, tab=None):
        if self.kind == "transformer":
            z = self.enc(self.inp(seq) + self.pos); z = torch.cat([z[:, -1], z.mean(1)], 1)
        else:
            z = self.enc(seq)[0][:, -1]
        if self.tab is not None:
            z = torch.cat([z, self.tab(tab)], 1)
        return self.head(z).squeeze(1)


def run(fold, kind, seq_only, H, epochs=10):
    dev = "cuda"
    torch.manual_seed(fold); np.random.seed(fold)
    D = {s: common.load(fold, s, common.FEATURES + ["product_id"]) for s in ("train", "val", "test")}
    pu, S = {}, {}
    for s, d in D.items():
        pu[s], S[s] = H.pairs(d.user_id.values, d.product_id.values)
        pu[s] = torch.from_numpy(pu[s]).to(dev); S[s] = torch.from_numpy(S[s]).to(dev)
    tab = None if seq_only else {k: v.to(dev) for k, v in zip(("train", "val", "test"), mlp.prep(*(D[s][common.FEATURES] for s in ("train", "val", "test"))))}
    ctx = torch.from_numpy(H.ctx).to(dev)
    y = torch.tensor(D["train"].label.values, dtype=torch.float32, device=dev)
    net = Net(kind, 0 if seq_only else tab["train"].shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3 if kind == "transformer" else 2e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.85)
    lossf = nn.BCEWithLogitsLoss()

    def batch(s, idx):
        rel = S[s][idx].float()
        x = torch.cat([rel.unsqueeze(-1), (rel > 0).float().unsqueeze(-1), ctx[pu[s][idx]]], -1)
        return x, (None if seq_only else tab[s][idx])

    def predict(s):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(S[s]), 50_000):
                idx = torch.arange(i, min(i + 50_000, len(S[s])), device=dev)
                out.append(torch.sigmoid(net(*batch(s, idx))).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(perm), 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad(); loss = lossf(net(*batch("train", idx)), y[idx]); loss.backward(); opt.step()
        sched.step()
        sc = average_precision_score(D["val"].label.values, predict("val"))
        print(f"  epoch {ep + 1} val AUPRC {sc:.4f}", flush=True)
        if sc > best:
            best, bad = sc, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    net.load_state_dict(best_state)
    return D, predict("val"), predict("test"), dict(best_val_auprc=round(float(best), 4), epochs_run=ep + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", nargs="+", default=["lstm", "gru", "transformer"], choices=["lstm", "gru", "transformer"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seq-only", action="store_true", help="sequence encoder only, no tabular features")
    a = ap.parse_args()
    (common.RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    users = pd.read_csv(common.SPLITS / "fold1.csv").user_id.values
    t0 = time.time(); H = Hist(users); print(f"history built ({time.time() - t0:.0f}s)", flush=True)
    for fold in a.folds:
        for kind in a.kind:
            t0 = time.time(); name = f"{kind}_{'seq' if a.seq_only else 'hybrid'}"
            D, sv, st, info = run(fold, kind, a.seq_only, H)
            row = dict(model=name, n_features=0 if a.seq_only else len(common.FEATURES), fold=fold, **common.evaluate(D["val"], D["test"], sv, st), **info, seconds=round(time.time() - t0))
            for part, sc in (("val", sv), ("test", st)):
                np.save(common.RESULTS / "preds" / f"{name}_fold{fold}_{part}.npy", np.asarray(sc, dtype=np.float32))
            common.save_row(row); common.report(row)


if __name__ == "__main__":
    main()
