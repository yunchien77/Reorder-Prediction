"""Combine saved models (their val / test scores) into one and evaluate like any other model.
  --mode mean : average of the probabilities
  --mode stack: logistic regression on the members' logits, fitted on VAL.  The val scores used for the F1 threshold are
                out-of-fold (5-fold, grouped by user), so the threshold is not chosen on scores the stacker was fitted on.
Members are model names with saved predictions (results/preds/), e.g. xgb lgbm catboost mlp.
Saves the ensemble's val / test scores as results/preds/{name}_..., so decision.py can use them.
   cd src && python -m ensemble.blend --name ens54_mean --mode mean --members xgb lgbm catboost mlp --folds 1 2 3 4 5"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import common


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--mode", choices=["mean", "stack"], default="mean")
    ap.add_argument("--members", nargs="+", required=True); ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    for fold in a.folds:
        val, test = common.load(fold, "val", ["label"]), common.load(fold, "test", ["label"])
        load = lambda m, part: np.load(common.RESULTS / "preds" / f"{m}_fold{fold}_{part}.npy").astype(np.float64)
        V = np.column_stack([load(m, "val") for m in a.members]); T = np.column_stack([load(m, "test") for m in a.members])
        info = {}
        if a.mode == "mean":
            sv, st = V.mean(1), T.mean(1)
        else:
            LV, LT = logit(V), logit(T)
            sv = np.zeros(len(LV))
            for tr, te in GroupKFold(5).split(LV, val.label.values, val.user_id.values):
                sv[te] = LogisticRegression(C=1.0).fit(LV[tr], val.label.values[tr]).predict_proba(LV[te])[:, 1]
            meta = LogisticRegression(C=1.0).fit(LV, val.label.values)
            st = meta.predict_proba(LT)[:, 1]
            info = {f"w_{m}": round(float(w), 3) for m, w in zip(a.members, meta.coef_[0])}
        row = dict(model=a.name, n_features=len(common.FEATURES), members=",".join(a.members), fold=fold,
                   **common.evaluate(val, test, sv, st), **info)
        for part, sc in (("val", sv), ("test", st)):
            np.save(common.RESULTS / "preds" / f"{a.name}_fold{fold}_{part}.npy", sc.astype(np.float32))
        common.save_row(row); common.report(row)
