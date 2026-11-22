"""CatBoost (ordered boosting, symmetric trees; GPU; NaN handled natively; early stopping on val logloss).
   cd src && python -m model.cat --folds 1 2 3 4 5"""
from catboost import CatBoostClassifier
import common


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, pos_weight=1.0):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=8, task_type="GPU", random_seed=seed, verbose=0,
                           early_stopping_rounds=50, eval_metric="Logloss", border_count=254, scale_pos_weight=pos_weight, allow_writing_files=False)
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], dict(best_iter=int(m.get_best_iteration()))


if __name__ == "__main__":
    common.main("catboost", fit_predict)
