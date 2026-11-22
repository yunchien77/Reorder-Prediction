"""Robustness analysis + bootstrap confidence intervals for the final models (rep_score rule, XGBoost 54, 4-model mean ensemble).
  segments : user reorder-rate bucket (terciles of user_reorder_ratio), user activity bucket (terciles of prior order count),
             product popularity tier (head = top 1 % of products by purchases, torso = next 19 %, tail = the rest)
  metrics  : per-segment AUPRC, positive rate, and (user segments) F1 (order) / Recall@10 / NDCG@10 averaged over ALL users of the segment
  bootstrap: user-level bootstrap (200 resamples of the test users) of AUPRC on every fold, plus paired differences
  cd src && python -m analysis.robustness   ->  results/robustness_segments.csv, results/bootstrap.csv"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import common

FOLDS = [1, 2, 3, 4, 5]
B = 200
R = pd.read_csv(common.RESULTS / "results.csv")
MODELS = {"rep_score（規則）": "rule:rep_score", "XGBoost 54": "xgb", "4 模型平均 ensemble": "ens54_mean"}


def score(model, fold, df):
    if model == "rule:rep_score":
        return df["rep_score"].values.astype(np.float64)
    return np.load(common.RESULTS / "preds" / f"{model}_fold{fold}_test.npy").astype(np.float64)


def threshold(model, fold):
    x = R[(R.model == model) & (R.fold == fold)]
    return float(x.threshold.iloc[0])


def tiers(df):
    cnt = np.expm1(df.product_popularity.values)
    prod = pd.DataFrame({"p": df.product_id.values, "c": cnt}).drop_duplicates("p").sort_values("c", ascending=False)
    n = len(prod); head = set(prod.p.values[: int(0.01 * n)]); torso = set(prod.p.values[int(0.01 * n): int(0.20 * n)])
    return np.where(df.product_id.isin(head), "head（前 1% 商品）", np.where(df.product_id.isin(torso), "torso（前 1%–20%）", "tail（其餘 80%）"))


rows = []
boot_out = []
for fold in FOLDS:
    boot_scores = {}
    df = common.load(fold, "test", ["rep_score", "user_reorder_ratio", "user_total_orders", "product_popularity"])
    df["tier"] = tiers(df)
    u = df.groupby("user_id").agg(rr=("user_reorder_ratio", "first"), n=("user_total_orders", "first"))
    u["rr_b"] = pd.qcut(u.rr, 3, labels=["低（回購比例最低 1/3）", "中", "高（回購比例最高 1/3）"], duplicates="drop")
    u["act_b"] = pd.qcut(u.n, 3, labels=["少（訂單數最少 1/3）", "中", "多（訂單數最多 1/3）"], duplicates="drop")
    df = df.join(u[["rr_b", "act_b"]], on="user_id")
    for mname, m in MODELS.items():
        s = score(m, fold, df)
        thr = threshold(m, fold)
        boot_scores[mname] = s
        for seg_kind, col in [("使用者回購比例", "rr_b"), ("使用者活躍度", "act_b"), ("商品熱門度", "tier")]:
            for g, idx in df.groupby(col, observed=True).indices.items():
                sub = df.iloc[idx]; ss = s[idx]; y = sub.label.values
                d = dict(fold=fold, model=mname, kind=seg_kind, segment=g, users=int(sub.user_id.nunique()), pairs=len(sub), pos_rate=float(y.mean()),
                         auprc=float(average_precision_score(y, ss)))
                if seg_kind != "商品熱門度":
                    um = common.user_metrics(sub, ss, thr)
                    f = um["users_eval"] / d["users"]          # all-user average (users without a positive count as 0)
                    d.update(f1_order=um["f1_order"] * f, recall_at_10=um["recall_at_10"] * f, ndcg_at_10=um["ndcg_at_10"] * f)
                else:
                    pred = ss >= thr
                    d.update(precision=float((pred & (y == 1)).sum() / max(pred.sum(), 1)), recall=float((pred & (y == 1)).sum() / max(y.sum(), 1)))
                rows.append(d)
    print("segments done", fold, flush=True)

    # bootstrap on this fold's test users (resampled with replacement; implemented as integer sample weights per user)
    y = df.label.values
    uid, inv = np.unique(df.user_id.values, return_inverse=True)
    rng = np.random.RandomState(fold)
    names = list(boot_scores)
    vals = {n: [] for n in names}
    for b in range(B):
        w = np.bincount(rng.randint(0, len(uid), len(uid)), minlength=len(uid))[inv]
        keep = w > 0
        for n in names:
            vals[n].append(average_precision_score(y[keep], boot_scores[n][keep], sample_weight=w[keep]))
    pt = {n: average_precision_score(y, boot_scores[n]) for n in names}
    for n in names:
        v = np.array(vals[n]); boot_out.append(dict(fold=fold, item=n, point=pt[n], lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    for a, bname in [("XGBoost 54", "rep_score（規則）"), ("4 模型平均 ensemble", "XGBoost 54")]:
        d = np.array(vals[a]) - np.array(vals[bname])
        boot_out.append(dict(fold=fold, item=f"差：{a} − {bname}", point=pt[a] - pt[bname], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    print("bootstrap done", fold, flush=True)

pd.DataFrame(rows).to_csv(common.RESULTS / "robustness_segments.csv", index=False)
pd.DataFrame(boot_out).to_csv(common.RESULTS / "bootstrap.csv", index=False)
print(pd.DataFrame(boot_out).to_string())
