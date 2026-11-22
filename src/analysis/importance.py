"""Feature importance of the 54-feature XGBoost on every fold (--folds): gain share and permutation importance on VAL (AUPRC drop when one column is shuffled).
   cd src && python -m analysis.importance     ->  results/feature_importance.csv  (long format, one row per fold x feature)"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgboost_lib
from sklearn.metrics import average_precision_score

import common
from feature.groups import GROUPS, POOL



def run(FOLD):
    tr, va = (common.load(FOLD, s, POOL) for s in ("train", "val"))
    m = xgboost_lib.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                  tree_method="hist", device="cuda", eval_metric="aucpr", early_stopping_rounds=50, random_state=FOLD)
    m.fit(tr[POOL], tr.label.values, eval_set=[(va[POOL], va.label.values)], verbose=False)
    v0 = average_precision_score(va.label.values, m.predict_proba(va[POOL])[:, 1])
    gain = pd.Series(m.get_booster().get_score(importance_type="total_gain")).reindex(POOL).fillna(0); gain /= gain.sum()
    rng = np.random.RandomState(0); perm = {}
    for c in POOL:
        drops = []
        for _ in range(2):
            x = va[POOL].copy(); x[c] = rng.permutation(x[c].values)
            drops.append(v0 - average_precision_score(va.label.values, m.predict_proba(x)[:, 1]))
        perm[c] = float(np.mean(drops))
    imp = pd.DataFrame({"gain_share": gain, "perm_drop_val_auprc": pd.Series(perm)})
    imp["group"] = {c: g for g, cs in GROUPS.items() for c in cs}
    imp["fold"] = FOLD
    print(f"fold {FOLD} done", flush=True)
    return imp.rename_axis("feature").reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    out = pd.concat([run(k) for k in ap.parse_args().folds]).round(5)
    out.to_csv(common.RESULTS / "feature_importance.csv", index=False)
    print(out.groupby("feature")[["gain_share", "perm_drop_val_auprc"]].mean().sort_values("gain_share", ascending=False).head(25).round(4).to_string())
