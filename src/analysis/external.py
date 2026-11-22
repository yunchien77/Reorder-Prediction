"""Does external data help?  XGBoost on the 54 base features  +  product-level features of one or several external sources (5-fold).
The 54 base features are NOT changed; external columns are joined by product_id (static product attributes, no target information).
Products without a match (or with name similarity < --thr for FDC / OFF) get NaN = missing, which XGBoost handles natively.
  sources : fk (FoodKeeper shelf life via aisle map) | fdc (USDA FoodData Central Branded, name match) | off (Open Food Facts US, name match)
  variants: base, fk, fdc, off, fk+fdc, fk+off, fdc+off, fk+fdc+off   (base = the 54 features only, same code path / seeds)
  cd src && python -m external.build && python -m analysis.external [--thr 0.9 0.8] [--folds 1 2 3 4 5] [--variants ...]   ->  results/external.csv"""
import argparse
import itertools

import numpy as np
import pandas as pd

import common
from model.xgb import fit_predict

SRC = ["fk", "fdc", "off"]
ALL = ["base"] + ["+".join(c) for r in (1, 2, 3) for c in itertools.combinations(SRC, r)]
ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--thr", type=float, nargs="+", default=[0.9], help="minimum name similarity for FDC / OFF matches (lower = more matches, more noise)")
ap.add_argument("--variants", nargs="+", default=ALL, choices=ALL)
a = ap.parse_args()

TAB = {s: pd.read_parquet(common.FEATURES_DIR / "external" / f"{s}.parquet").set_index("product_id") for s in SRC}


def table(src, thr):
    t = TAB[src].copy()
    if src != "fk":
        bad = t[f"{src}_score"] < thr
        t.loc[bad, :] = np.nan                       # below the threshold: treat as not matched (all columns missing, score too)
    return t.astype(np.float32)


def X_for(df, srcs, thr):
    parts = [df[common.FEATURES].reset_index(drop=True)] + [table(s, thr).reindex(df.product_id.values).reset_index(drop=True) for s in srcs]
    return pd.concat(parts, axis=1)


out = common.RESULTS / "external.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {(r["variant"], r["thr"], r["fold"]) for r in rows}
configs = []
for v in a.variants:
    srcs = [] if v == "base" else v.split("+")
    for thr in (a.thr if any(s != "fk" for s in srcs) else [np.nan]):
        configs.append((v, srcs, thr))
for fold in a.folds:
    tr, va, te = (common.load(fold, s, common.FEATURES + ["product_id"]) for s in ("train", "val", "test"))
    for v, srcs, thr in configs:
        if any(r[0] == v and (r[1] == thr or (np.isnan(r[1]) and np.isnan(thr))) and r[2] == fold for r in done):
            continue
        Xtr, Xva, Xte = (X_for(d, srcs, thr) for d in (tr, va, te))
        sv, st, info = fit_predict(Xtr, tr.label.values, Xva, va.label.values, Xte, fold)
        m = common.evaluate(va, te, sv, st)
        rows.append(dict(variant=v, thr=thr, n=Xtr.shape[1], fold=fold, best_iter=info["best_iter"], **m))
        done.add((v, thr, fold))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"fold {fold} {v:12s} thr={thr} n={Xtr.shape[1]} AUPRC {m['auprc']:.4f} F1 {m['f1']:.4f} F1-order {m['f1_order']:.4f}", flush=True)
