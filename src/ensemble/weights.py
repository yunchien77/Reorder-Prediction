"""Different ways to weight the members of one ensemble (members = model names with saved val/test scores).
  equal_prob   : mean of probabilities            equal_logit : mean of logits           rank_avg : mean of percentile ranks
  nonneg_logit : logistic-regression stack with weights >= 0 (fitted on val)
  auprc_opt    : non-negative weights summing to 1 chosen to maximise AUPRC on val (Nelder-Mead on softmax weights)
Weights are fitted on val; the val scores used for the F1 threshold are 5-fold out-of-fold (grouped by user).
   cd src && python -m ensemble.weights --folds 1 2 3 4 5 --members xgb lgbm catboost mlp"""
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
import common
from ensemble.blend import logit


def fit_nonneg_logit(L, y):
    k = L.shape[1]
    def f(p):
        z = L @ p[:k] + p[k]
        return np.mean(np.logaddexp(0, z) - y * z)
    r = minimize(f, np.r_[np.ones(k) / k, 0.0], method="L-BFGS-B", bounds=[(0, None)] * k + [(None, None)])
    return r.x


def fit_auprc(P, y):
    k = P.shape[1]
    def f(a):
        w = np.exp(a) / np.exp(a).sum()
        return -average_precision_score(y, P @ w)
    r = minimize(f, np.zeros(k), method="Nelder-Mead", options=dict(xatol=1e-2, fatol=1e-7, maxiter=150, initial_simplex=np.vstack([np.zeros(k)] + [0.7 * np.eye(k)[i] for i in range(k)])))
    w = np.exp(r.x) / np.exp(r.x).sum()
    return w


def pct_rank(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def make(method, Vp, Tp, y, groups):
    """returns (oof val scores, test scores, weights)"""
    Lv, Lt = logit(Vp), logit(Tp)
    if method == "equal_prob":
        return Vp.mean(1), Tp.mean(1), np.ones(Vp.shape[1]) / Vp.shape[1]
    if method == "equal_logit":
        return Lv.mean(1), Lt.mean(1), np.ones(Vp.shape[1]) / Vp.shape[1]
    if method == "rank_avg":
        rv = np.column_stack([pct_rank(c) for c in Vp.T]).mean(1); rt = np.column_stack([pct_rank(c) for c in Tp.T]).mean(1)
        return pct_rank(rv), pct_rank(rt), np.ones(Vp.shape[1]) / Vp.shape[1]
    sv = np.zeros(len(y))
    if method == "nonneg_logit":
        fit = lambda idx: fit_nonneg_logit(Lv[idx], y[idx]); score = lambda p, L: L @ p[:-1] + p[-1]
        for tr, te in GroupKFold(5).split(Lv, y, groups):
            sv[te] = score(fit(tr), Lv[te])
        p = fit(np.arange(len(y)))
        return 1 / (1 + np.exp(-sv)), 1 / (1 + np.exp(-score(p, Lt))), p[:-1]
    if method == "auprc_opt":
        for tr, te in GroupKFold(5).split(Vp, y, groups):
            sv[te] = Vp[te] @ fit_auprc(Vp[tr], y[tr])
        w = fit_auprc(Vp, y)
        return sv, Tp @ w, w
    raise ValueError(method)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True); ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--prefix", default="ensw"); ap.add_argument("--methods", nargs="+", default=["equal_prob", "equal_logit", "rank_avg", "nonneg_logit", "auprc_opt"])
    a = ap.parse_args()
    for fold in a.folds:
        val, test = common.load(fold, "val", ["label"]), common.load(fold, "test", ["label"])
        load = lambda m, part: np.load(common.RESULTS / "preds" / f"{m}_fold{fold}_{part}.npy").astype(np.float64)
        Vp = np.column_stack([load(m, "val") for m in a.members]); Tp = np.column_stack([load(m, "test") for m in a.members])
        for method in a.methods:
            sv, st, w = make(method, Vp, Tp, val.label.values, val.user_id.values)
            info = {f"w_{m}": round(float(x), 3) for m, x in zip(a.members, w)}
            row = dict(model=f"{a.prefix}_{method}", n_features=len(common.FEATURES), members=",".join(a.members), fold=fold, **common.evaluate(val, test, sv, st), **info)
            common.save_row(row); common.report(row)
