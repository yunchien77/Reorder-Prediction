"""Feature importance of the 54-feature XGBoost on fold 1: gain share and permutation importance on VAL (AUPRC drop when one column is shuffled).
   cd src && python -m analysis.importance     ->  results/feature_importance_pool.csv"""
import numpy as np
import pandas as pd
import xgboost as xgboost_lib
from sklearn.metrics import average_precision_score

import common
from feature.groups import GROUPS, POOL

FOLD = 1


if __name__ == "__main__":
    tr, va, te = (common.load(FOLD, s, POOL) for s in ("train", "val", "test"))
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
    imp = imp.sort_values("gain_share", ascending=False).round(5)
    imp.to_csv(common.RESULTS / "feature_importance_pool.csv")
    print(imp.head(25).round(4).to_string())
