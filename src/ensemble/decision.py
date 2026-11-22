"""Better order-level F1 without retraining: decide per user how many items to recommend, from saved probabilities.
Compares on test (users with >= 1 positive):
  global     : one probability threshold for everybody (chosen on val)            <- what common.py reports
  thr+top1   : same threshold, but every user gets at least the top-1 item
  exp-F1     : per user, choose k that maximises the expected F1  2*sum(p_1..p_k) / (k + sum(all p))   (needs calibrated probabilities)
   cd src && python -m ensemble.decision --model xgb --folds 1 2 3 4 5"""
import argparse
import numpy as np
import pandas as pd
import common


def select(df, p, mode, thr):
    tie = np.random.RandomState(0).rand(len(df))
    order = np.lexsort((tie, -p, df.user_id.values))
    u = df.user_id.values[order]; ps = p[order]
    start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]; n = np.diff(np.r_[start, len(u)])
    rank = np.arange(len(u)) - np.repeat(start, n)
    if mode == "global":
        sel = ps >= thr
    elif mode == "thr+top1":
        sel = (ps >= thr) | (rank == 0)
    else:                                                   # exp-F1
        cs = np.cumsum(ps); csum = cs - np.repeat(cs[start] - ps[start], n)
        tot = np.repeat(np.add.reduceat(ps, start), n)
        ef1 = 2 * csum / (rank + 1 + tot)
        best = np.repeat(np.maximum.reduceat(ef1, start), n)
        pos = np.where(ef1 >= best, np.arange(len(u)), len(u) + 1)
        kbest = np.repeat(np.minimum.reduceat(pos, start), n) - np.repeat(start, n)
        sel = rank <= kbest
    y = df.label.values[order].astype(np.int64)
    npos = np.add.reduceat(y, start); npred = np.add.reduceat(sel.astype(np.int64), start); tp = np.add.reduceat(y * sel, start)
    keep = npos > 0
    f1_order = float((2 * tp / np.maximum(npred + npos, 1))[keep].mean())
    P = tp.sum() / max(npred.sum(), 1); R = tp.sum() / npos.sum()
    return dict(f1_order=f1_order, precision=float(P), recall=float(R), f1=float(2 * P * R / max(P + R, 1e-12)), avg_items=float(npred[keep].mean()))


def run(model, fold):
    val = common.load(fold, "val", ["label"])        # labels and user ids from features/ (row order = the order of the saved predictions)
    test = common.load(fold, "test", ["label"])
    pv = np.load(common.RESULTS / "preds" / f"{model}_fold{fold}_val.npy").astype(np.float64)
    pt = np.load(common.RESULTS / "preds" / f"{model}_fold{fold}_test.npy").astype(np.float64)
    thr = common.best_f1_threshold(val.label.values, pv)
    rows = []
    for mode in ("global", "thr+top1", "exp-F1"):
        r = dict(model=model, fold=fold, decision=mode, **select(test, pt, mode, thr)); rows.append(r)
        print(f"{model} fold {fold}  {mode:9s} F1-order {r['f1_order']:.4f}  pair F1 {r['f1']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}  items/user {r['avg_items']:.1f}")
    f = common.RESULTS / "postprocess.csv"
    new = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f); new = pd.concat([old[~((old.model == model) & (old.fold == fold))], new])
    new.to_csv(f, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb"); ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    for fold in a.folds:
        run(a.model, fold)
