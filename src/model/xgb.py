"""XGBoost: raw features (NaN handled natively), GPU, early stopping on val AUCPR.   about 35 s per fold.
   cd src && python -m model.xgb --folds 1 2 3 4 5"""
import xgboost as xgboost_lib
import common


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, pos_weight=1.0):
    m = xgboost_lib.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", device="cuda", eval_metric="aucpr",
                                  early_stopping_rounds=50, random_state=seed, scale_pos_weight=pos_weight)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], dict(best_iter=int(m.best_iteration))


if __name__ == "__main__":
    common.main("xgb", fit_predict)
