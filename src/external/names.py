"""Shared helpers of the external-data builders: Instacart products (+aisle / department) and name matching.

Matching only uses the product NAME (TF-IDF cosine, word 1-2 grams) against a corpus of product names; nothing about users, orders or labels
is used, so the resulting product-level features are static and cannot leak the target.  Matches below the similarity threshold are
dropped later (features = NaN)."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import common

RAW = common.ROOT / "dataset" / "raw"
EXTERNAL = common.ROOT / "dataset" / "external"
OUT = common.FEATURES_DIR / "external"
NONFOOD = {"household", "personal care", "pets", "other"}       # departments that are not food (never matched)


def products():
    """All Instacart products with aisle / department, and is_food (department not in NONFOOD)."""
    p = pd.read_csv(RAW / "products.csv").merge(pd.read_csv(RAW / "aisles.csv")).merge(pd.read_csv(RAW / "departments.csv"))
    p["is_food"] = ~p.department.isin(NONFOOD)
    return p[["product_id", "product_name", "aisle", "department", "is_food"]]


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())


def top1(queries, corpus, max_df=0.01):
    """Best TF-IDF match of every query name in the corpus -> (index into corpus (-1 = none), cosine similarity)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=max_df, dtype=np.float32)
    XT = vec.fit_transform(pd.Series(corpus).map(_norm)).T.tocsr()
    Q = vec.transform(pd.Series(queries).map(_norm))
    idx = np.full(len(queries), -1, np.int64); score = np.zeros(len(queries), np.float32)
    for i in range(0, len(queries), 200):
        S = (Q[i:i + 200] @ XT).tocsr()
        for j in range(S.shape[0]):
            lo, hi = S.indptr[j], S.indptr[j + 1]
            if hi > lo:
                k = S.data[lo:hi].argmax(); idx[i + j] = S.indices[lo + k]; score[i + j] = S.data[lo + k]
    return idx, score
