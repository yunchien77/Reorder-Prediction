"""LightGBM: raw features (NaN handled natively), early stopping on val logloss.   CPU.
   cd src && python -m model.lgbm --folds 1 2 3 4 5"""
import lightgbm as lgb
import common


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, pos_weight=1.0):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=16, random_state=seed, verbose=-1, scale_pos_weight=pos_weight)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], dict(best_iter=int(m.best_iteration_))


if __name__ == "__main__":
    common.main("lgbm", fit_predict)
