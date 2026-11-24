"""Which segments do the extra features help?  XGBoost (same seeds as analysis.external / analysis.embed) trained three ways per fold:
  base = 54 features | ext = 54 + FoodKeeper + FDC + OFF (similarity >= 0.9) | emb = 54 + item2vec / text-embedding similarities + 16-dim product vectors (94)
Test scores are saved to results/preds/segext_{kind}_fold{k}.npy (training is skipped when they exist); AUPRC is then computed inside each
product / user segment, and the differences ext - base, emb - base are written to results/segments_ext.csv (one row per fold x segment).
  cd src && python -m analysis.segments_ext [--folds 1 2 3 4 5]   (needs external.build and embed.build first)"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

import common
from embed.build import OUT as EMB, SIMS
from external.names import products
from model.xgb import fit_predict

PRED = common.RESULTS / "preds"
a = argparse.ArgumentParser(); a.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5]); a = a.parse_args()
TAB = {s: pd.read_parquet(common.FEATURES_DIR / "external" / f"{s}.parquet").set_index("product_id") for s in ("fk", "fdc", "off")}


def ext(src, thr=0.9):
    t = TAB[src].copy()
    if src != "fk":
        t.loc[t[f"{src}_score"] < thr, :] = np.nan
    return t.astype(np.float32)


def train(fold):
    D = {s: common.load(fold, s, common.FEATURES + ["product_id"]) for s in ("train", "val", "test")}
    full = pd.read_parquet(EMB / f"fold{fold}.parquet"); E = {s: full[full.split == s].reset_index(drop=True) for s in D}
    emb = {"i2v": np.load(EMB / f"i2v_fold{fold}.npy"), "txt": np.load(EMB / "txt_pca16.npy")}

    def X(s, kind):
        parts = [D[s][common.FEATURES].reset_index(drop=True)]
        if kind == "ext":
            parts += [ext(x).reindex(D[s].product_id.values).reset_index(drop=True) for x in ("fk", "fdc", "off")]
        if kind == "emb":
            parts.append(E[s][[f"{t}_{c}" for t in ("i2v", "txt") for c in SIMS]])
            for t in ("i2v", "txt"):
                M = emb[t][D[s].product_id.values]; parts.append(pd.DataFrame(M, columns=[f"{t}_e{i}" for i in range(M.shape[1])]))
        return pd.concat(parts, axis=1)

    for kind in ("base", "ext", "emb"):
        f = PRED / f"segext_{kind}_fold{fold}.npy"
        if not f.exists():
            _, st, _ = fit_predict(X("train", kind), D["train"].label.values, X("val", kind), D["val"].label.values, X("test", kind), fold)
            np.save(f, st.astype(np.float32))


P = products().set_index("product_id")
match = pd.DataFrame({"fk": TAB["fk"].fk_shelf_days.notna(), "fdc": TAB["fdc"].fdc_score >= .9, "off": TAB["off"].off_score >= .9}).reindex(P.index).fillna(False)
match["any3"] = match.fk | match.fdc | match.off
DEPTS = ["produce", "dairy eggs", "snacks", "beverages", "frozen", "pantry", "bakery", "canned goods", "deli", "dry goods pasta", "household", "meat seafood",
         "personal care", "breakfast", "babies", "alcohol", "pets"]
rows = []
for fold in a.folds:
    train(fold)
    d = common.load(fold, "test", ["up_times_purchased", "product_popularity", "user_total_orders", "product_id"])
    s = {k: np.load(PRED / f"segext_{k}_fold{fold}.npy") for k in ("base", "ext", "emb")}
    y = d.label.values; n = d.up_times_purchased.values
    ptile = pd.Series(d.product_popularity.values).rank(pct=True).values
    seg = {"買過次數|1 次": n == 1, "買過次數|2 次": n == 2, "買過次數|3–4 次": (n >= 3) & (n <= 4), "買過次數|5 次以上": n >= 5,
           "商品熱門度|前 5%": ptile > .95, "商品熱門度|中間 5–50%": (ptile > .5) & (ptile <= .95), "商品熱門度|後 50%": ptile <= .5}
    q = pd.qcut(d.user_total_orders, 5, labels=False, duplicates="drop").values
    for i in range(5):
        seg[f"使用者訂單數|Q{i + 1}（{'少' if i == 0 else '多' if i == 4 else '中'}）"] = q == i
    dep = P.department.reindex(d.product_id.values).values
    for k in DEPTS:
        seg[f"部門|{k}"] = dep == k
    mm = match.reindex(d.product_id.values)
    for k, nm in (("fk", "FoodKeeper"), ("fdc", "FDC"), ("off", "OFF"), ("any3", "任一外部資料")):
        seg[f"配對到|{nm}：有"] = mm[k].values; seg[f"配對到|{nm}：沒有"] = ~mm[k].values
    for name, m in seg.items():
        if m.sum() < 2000 or y[m].sum() < 50:
            continue
        b = ap(y[m], s["base"][m])
        rows.append(dict(fold=fold, seg=name, share=m.mean(), pos=y[m].mean(), base=b, ext=ap(y[m], s["ext"][m]) - b, emb=ap(y[m], s["emb"][m]) - b))
    print("fold", fold, flush=True)
pd.DataFrame(rows).to_csv(common.RESULTS / "segments_ext.csv", index=False)
