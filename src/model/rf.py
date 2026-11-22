"""Random Forest: NaN -> -1, + bought_ge2 / bought_ge3 flags; trained on a 1M-row random sample of train (5.9M rows is too big).
   cd src && python -m model.rf --folds 1 2 3 4 5"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import common


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, pos_weight=1.0):
    Xtr, Xva, Xte = common.add_flags(Xtr), common.add_flags(Xva), common.add_flags(Xte)
    Xtr, Xva, Xte = (x.fillna(-1).values.astype(np.float32) for x in (Xtr, Xva, Xte))
    idx = np.random.RandomState(seed).choice(len(Xtr), min(1_000_000, len(Xtr)), replace=False)
    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt", n_jobs=16, random_state=seed,
                               class_weight={0: 1.0, 1: pos_weight})
    m.fit(Xtr[idx], ytr[idx])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], dict(n_inputs=Xtr.shape[1], train_rows=len(idx))


if __name__ == "__main__":
    common.main("rf", fit_predict)
