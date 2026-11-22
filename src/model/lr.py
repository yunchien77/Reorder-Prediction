"""Logistic Regression: NaN -> train median, + bought_ge2 / bought_ge3 flags (54 features + 2 flags), standardised.
   cd src && python -m model.lr --folds 1 2 3 4 5"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import common


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, pos_weight=1.0):
    Xtr, Xva, Xte = common.add_flags(Xtr), common.add_flags(Xva), common.add_flags(Xte)
    med = Xtr.median()
    Xtr, Xva, Xte = (x.fillna(med).values.astype(np.float32) for x in (Xtr, Xva, Xte))
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=300, C=1.0, class_weight={0: 1.0, 1: pos_weight}).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xva))[:, 1], m.predict_proba(sc.transform(Xte))[:, 1], dict(n_iter=int(m.n_iter_[0]), n_inputs=Xtr.shape[1])


if __name__ == "__main__":
    common.main("lr", fit_predict)
