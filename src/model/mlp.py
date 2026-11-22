"""MLP on the tabular features (PyTorch, GPU).  NaN -> train median + bought_ge2/bought_ge3 flags, standardised; heavy-tailed columns get signed log1p first.
3 hidden layers (256-128-64), BatchNorm + dropout 0.1, Adam, up to 15 epochs, keeps the epoch with the best val AUPRC.
Needs PyTorch in the active environment.   cd src && python -m model.mlp --folds 1 2 3 4 5"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import common


def prep(Xtr, Xva, Xte):
    Xtr, Xva, Xte = common.add_flags(Xtr), common.add_flags(Xva), common.add_flags(Xte)
    med = Xtr.median()
    A = [x.fillna(med).values.astype(np.float32) for x in (Xtr, Xva, Xte)]
    skew = (np.abs(A[0]).max(0) > 50)                     # large-range columns (counts, days, popularity ...)
    A = [np.where(skew, np.sign(a) * np.log1p(np.abs(a)), a) for a in A]
    mu, sd = A[0].mean(0), A[0].std(0) + 1e-6
    return [torch.tensor((a - mu) / sd, dtype=torch.float32) for a in A]


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, pos_weight=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    tr, va, te = prep(Xtr, Xva, Xte)
    y = torch.tensor(ytr, dtype=torch.float32)
    d = tr.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.85)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=dev))

    def predict(X):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 200_000):
                out.append(torch.sigmoid(net(X[i:i + 200_000].to(dev))).squeeze(1).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(15):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad(); loss = lossf(net(tr[idx].to(dev)).squeeze(1), y[idx].to(dev)); loss.backward(); opt.step()
        sched.step()
        s = average_precision_score(yva, predict(va))
        if s > best:
            best, bad = s, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    net.load_state_dict(best_state)
    return predict(va), predict(te), dict(best_val_auprc=round(float(best), 4), epochs_run=ep + 1)


if __name__ == "__main__":
    common.main("mlp", fit_predict)
