"""Do product-embedding features help?  XGBoost on the 54 base features + embedding-similarity features (5-fold).  The 54 are NOT changed.
  components: i2v_sim / txt_sim : cosine of the product to the user's last order / last 3 orders / history (4 features each)
              i2v_emb / txt_emb : the product's own embedding (16 dims)
  i2v = item2vec-style vectors from train users' baskets (per fold, no leakage; not external data);  txt = pretrained MiniLM on product names.
  cd src && python -m embed.build && python -m analysis.embed [--folds 1 2 3 4 5] [--variants ...]   ->  results/embed.csv"""
import argparse

import numpy as np
import pandas as pd

import common
from embed.build import OUT, SIMS
from model.xgb import fit_predict

VARIANTS = {"base": [], "i2v_sim": ["i2v_sim"], "txt_sim": ["txt_sim"], "i2v_sim+txt_sim": ["i2v_sim", "txt_sim"],
            "i2v_sim+i2v_emb": ["i2v_sim", "i2v_emb"], "txt_sim+txt_emb": ["txt_sim", "txt_emb"],
            "all": ["i2v_sim", "txt_sim", "i2v_emb", "txt_emb"]}
ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
a = ap.parse_args()


def X_for(df, E, emb, comps):
    parts = [df[common.FEATURES].reset_index(drop=True)]
    for c in comps:
        tag, kind = c.split("_")
        if kind == "sim":
            parts.append(E[[f"{tag}_{s}" for s in SIMS]].reset_index(drop=True))
        else:
            M = emb[tag][df.product_id.values]
            parts.append(pd.DataFrame(M, columns=[f"{tag}_e{i}" for i in range(M.shape[1])]))
    return pd.concat(parts, axis=1)


out = common.RESULTS / "embed.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {(r["variant"], r["fold"]) for r in rows}
for fold in a.folds:
    D = {s: common.load(fold, s, common.FEATURES + ["product_id"]) for s in ("train", "val", "test")}
    full = pd.read_parquet(OUT / f"fold{fold}.parquet")
    Es = {}
    for s in D:
        Es[s] = full[full.split == s].reset_index(drop=True)
        assert (Es[s].user_id.values == D[s].user_id.values).all() and (Es[s].product_id.values == D[s].product_id.values).all()
    emb = {"i2v": np.load(OUT / f"i2v_fold{fold}.npy"), "txt": np.load(OUT / "txt_pca16.npy")}
    for v in a.variants:
        if (v, fold) in done:
            continue
        Xtr, Xva, Xte = (X_for(D[s], Es[s], emb, VARIANTS[v]) for s in ("train", "val", "test"))
        sv, st, info = fit_predict(Xtr, D["train"].label.values, Xva, D["val"].label.values, Xte, fold)
        m = common.evaluate(D["val"], D["test"], sv, st)
        rows.append(dict(variant=v, n=Xtr.shape[1], fold=fold, best_iter=info["best_iter"], **m))
        done.add((v, fold))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"fold {fold} {v:16s} n={Xtr.shape[1]} AUPRC {m['auprc']:.4f} F1 {m['f1']:.4f} F1-order {m['f1_order']:.4f}", flush=True)
