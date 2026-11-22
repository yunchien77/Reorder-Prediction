"""Feature-group ablation of the 54 features with any model (xgb, lgbm, cat, lr, rf, mlp), all metrics.
  drop : the 54-feature pool minus one group (12 groups)        -> necessary if the score falls
  add  : core (G1 + G2) plus one more group (10 groups)         -> helpful if the score rises over the core
  cd src && python -m analysis.ablation --model xgb [--folds 1 2 3 4 5]  ->  results/ablation_{model}.csv"""
import argparse
import importlib

import pandas as pd

import common
from feature.groups import GROUPS, POOL

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="xgb", choices=["xgb", "lgbm", "cat", "lr", "rf", "mlp"])
ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
a = ap.parse_args()
fit_predict = importlib.import_module(f"model.{a.model}").fit_predict

names = list(GROUPS)
core = sum([GROUPS[g] for g in names[:2]], [])
configs = [("pool", "全部 54 個", POOL), ("core", "核心 G1+G2", core)]
configs += [("drop", g, [c for c in POOL if c not in GROUPS[g]]) for g in names]
configs += [("add", g, core + GROUPS[g]) for g in names[2:]]
out = common.RESULTS / f"ablation_{a.model}.csv"
out.parent.mkdir(exist_ok=True)
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {(r["kind"], r["group"], r["fold"]) for r in rows}
for fold in a.folds:
    tr, va, te = (common.load(fold, s, POOL) for s in ("train", "val", "test"))
    for kind, g, cols in configs:
        if (kind, g, fold) in done:
            continue
        sv, st, _ = fit_predict(tr[cols], tr.label.values, va[cols], va.label.values, te[cols], fold)
        m = common.evaluate(va, te, sv, st)
        rows.append(dict(model=a.model, kind=kind, group=g, n=len(cols), fold=fold, **m))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(a.model, fold, kind, g, len(cols), round(m["auprc"], 4), flush=True)
