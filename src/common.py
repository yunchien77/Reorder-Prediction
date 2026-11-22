"""Shared pieces for every model script: data loading, the 54-feature list, the 7 metrics, result logging, CLI.

A model file only has to define   fit_predict(Xtr, ytr, Xva, yva, Xte, seed) -> (val_scores, test_scores, info_dict)
and call                          common.main("name", fit_predict)        (run as  cd src && python -m model.xgb --folds 1 2 3 4 5)

Data    : features/fold{k}.parquet   (one table per CV fold with a `split` column; built by feature/build.py)
Metrics : AUPRC, Precision, Recall, F1 (threshold = F1-optimal on val), F1 (order-level), Recall@10, NDCG@10 (+ AUROC).
          Order-level F1 / Recall@10 / NDCG@10 are averaged over users with >= 1 positive among their candidates.
Output  : results/results.csv (one row per model x fold; re-running replaces the row),
          results/preds/{model}_fold{k}_test.npy (val / test scores in the row order of val.csv / test.csv)
"""
import argparse
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEATURES_DIR = ROOT / "features"                            # features/fold{k}.parquet (built by feature/build.py)
SPLITS = FEATURES_DIR / "split"                             # features/split/fold{k}.csv (built by data/split.py)
RESULTS = ROOT / "results"
K = 10

INTERACTION = ["up_times_purchased", "up_order_rate", "up_orders_since_last", "up_days_since_last_prior",
               "decay_freq_b05", "decay_freq_b08", "decay_freq_b09", "lag1", "lag2", "lag3", "in_last_3", "in_last_5",
               "up_streak", "gap_mean", "gap_std", "since_over_gap", "first_on", "up_rate_since_first", "up_rank_in_user"]
PRODUCT = ["product_reorder_rate", "product_popularity", "aisle_reorder_rate", "department_reorder_rate", "rep_score"]
USER = ["user_total_orders", "user_distinct_products", "user_avg_basket_size", "user_reorder_ratio",
        "user_mean_gap_days", "user_aisle_share", "user_dept_share"]
EXT_CART = ["up_avg_cart_pos", "up_avg_cart_rel", "product_avg_cart_pos"]              # cart position
EXT_AISLE = ["aisle_orders_since_last", "dept_orders_since_last"]                      # aisle / department recency
EXT_PGAP = ["product_mean_gap", "gap_ratio_to_product"]                                # product-level cadence
TIME = ["target_dow", "target_hour_bin", "user_dow_share", "user_hour_share", "up_dow_purchase_count", "up_hour_purchase_count", "up_dow_share", "up_hour_share",
        "up_dow_diff", "up_hour_diff", "prod_dow_lift", "aisle_dow_lift", "dept_dow_lift", "prod_hour_lift", "aisle_hour_lift", "dept_hour_lift"]
FEATURES = INTERACTION + PRODUCT + USER + EXT_CART + EXT_AISLE + EXT_PGAP + TIME      # the 54 features (G1-G12)


# ---------------------------------------------------------------- metrics
def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def user_metrics(df, score, thr, k=K):
    """Mean over users with >= 1 positive of: order-level F1, Recall@k, NDCG@k."""
    tie = np.random.RandomState(0).rand(len(df))               # random tie-break, so ties do not favour low product ids
    order = np.lexsort((tie, -score, df.user_id.values))       # by user, then score descending
    u = df.user_id.values[order]; y = df.label.values[order].astype(np.int64); s = score[order]
    start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
    n = np.diff(np.r_[start, len(u)])
    rank = np.arange(len(u)) - np.repeat(start, n)             # 0 = best-scored pair of the user
    npos = np.add.reduceat(y, start)
    npred = np.add.reduceat((s >= thr).astype(np.int64), start)
    tp = np.add.reduceat(y * (s >= thr), start)
    top = rank < k
    hits = np.add.reduceat(y * top, start)
    dcg = np.add.reduceat(y * top / np.log2(rank + 2), start)
    idcg = np.cumsum(1.0 / np.log2(np.arange(k) + 2))[np.minimum(npos, k).clip(min=1) - 1]
    keep = npos > 0
    return dict(f1_order=float((2 * tp / np.maximum(npred + npos, 1))[keep].mean()),
                recall_at_10=float((hits / np.maximum(npos, 1))[keep].mean()),
                ndcg_at_10=float((dcg / idcg)[keep].mean()), users_eval=int(keep.sum()))


def evaluate(val, test, s_val, s_test):
    thr = best_f1_threshold(val.label.values, s_val)
    pred = s_test >= thr; y = test.label.values.astype(bool)
    tp = int((pred & y).sum()); p = tp / max(pred.sum(), 1); r = tp / y.sum()
    out = dict(auprc=average_precision_score(y, s_test), precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12),
               auroc=roc_auc_score(y, s_test), threshold=thr, val_auprc=average_precision_score(val.label.values, s_val))
    out.update(user_metrics(test, s_test, thr))
    return out


# ---------------------------------------------------------------- helpers
def add_flags(X):
    """bought_ge2 / bought_ge3: 1 if the product was bought >= 2 / >= 3 times (= lag2 / lag3 present). For models that need filled NaNs."""
    X = X.copy()
    if "lag2" in X:
        X["bought_ge2"] = X["lag2"].notna().astype(np.float32); X["bought_ge3"] = X["lag3"].notna().astype(np.float32)
    elif "r_up_purchase_count" in X:                                                   # the reference repo's feature set
        X["bought_ge2"] = (X["r_up_purchase_count"] >= 2).astype(np.float32); X["bought_ge3"] = (X["r_up_purchase_count"] >= 3).astype(np.float32)
    return X


def load(fold, split, cols):
    """Rows of one split (user_id, product_id, label + cols) from features/fold{k}.parquet."""
    feats = [c for c in cols if c not in ("label", "user_id", "product_id")]
    d = pd.read_parquet(FEATURES_DIR / f"fold{fold}.parquet", columns=["user_id", "product_id", "label"] + feats,
                        filters=[("split", "==", split)])
    return d.reset_index(drop=True)


def save_row(row):
    """Append the row to results.csv, replacing an older row of the same model / fold (file lock: several model scripts may run in parallel)."""
    import fcntl
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / "results.csv"
    with open(RESULTS / ".results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = pd.DataFrame([row])
        if f.exists():
            old = pd.read_csv(f)
            same = (old.model == row["model"]) & (old.fold == row["fold"])
            new = pd.concat([old[~same], new], ignore_index=True)
        new.to_csv(f, index=False)


def report(row):
    print(f"fold {row['fold']} {row['model']:>16s} AUPRC {row['auprc']:.4f}  AUROC {row['auroc']:.4f}  P {row['precision']:.4f}  "
          f"R {row['recall']:.4f}  F1 {row['f1']:.4f}  F1-order {row['f1_order']:.4f}  R@10 {row['recall_at_10']:.4f}  "
          f"NDCG@10 {row['ndcg_at_10']:.4f}  (val AUPRC {row['val_auprc']:.4f})", flush=True)


def main(name, fit_predict):
    ap = argparse.ArgumentParser(description=f"train / evaluate {name}")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pos-weight", type=float, default=1.0, help="class weight of the positive class (1 = none; ~9.2 = fully balanced)")
    ap.add_argument("--neg-frac", type=float, default=1.0, help="keep only this fraction of the NEGATIVE training rows (random under-sampling); val/test untouched")
    args = ap.parse_args()
    cols = FEATURES
    if args.pos_weight != 1.0:
        name = f"{name}_pw{args.pos_weight:g}"
    if args.neg_frac < 1.0:
        name = f"{name}_neg{args.neg_frac:g}"
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        t0 = time.time()
        train, val, test = (load(fold, s, cols) for s in ("train", "val", "test"))
        if args.neg_frac < 1.0:   # random under-sampling of negatives (training rows only)
            keep = (train.label.values == 1) | (np.random.RandomState(fold).rand(len(train)) < args.neg_frac)
            train = train[keep].reset_index(drop=True)
        kw = {"groups": (train.user_id.values, val.user_id.values)} if "groups" in inspect.signature(fit_predict).parameters else {}
        if "pos_weight" in inspect.signature(fit_predict).parameters:
            kw["pos_weight"] = args.pos_weight
        elif args.pos_weight != 1.0:
            raise SystemExit("this model does not support --pos-weight")
        s_val, s_test, info = fit_predict(train[cols], train.label.values, val[cols], val.label.values, test[cols], fold, **kw)
        row = dict(model=name, n_features=len(cols), fold=fold, **evaluate(val, test, s_val, s_test),
                   **info, seconds=round(time.time() - t0))
        for part, sc in (("val", s_val), ("test", s_test)):
            np.save(RESULTS / "preds" / f"{name}_fold{fold}_{part}.npy", np.asarray(sc, dtype=np.float32))
        save_row(row); report(row)
