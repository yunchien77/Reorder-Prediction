"""Builds the consolidated feature tables used for all experiments:  features/fold{k}.parquet  (one file per CV fold).

Columns : user_id, product_id, label, split (train / val / test) + 54 features:
            31 base features (INTERACTION + PRODUCT + USER),
            7  extra prior-only features (cart position 3, aisle / department recency 2, product-level cadence 2),
            16 "current time" features (day of week / hour of the order to be predicted, and their interaction with product / aisle / department / the user's habit).
          NOT included: any "days since the previous order" column (known only after the order is placed).
Rules   : product / aisle / department statistics (5 + 3 + 6 lift columns) come from the TRAIN users of that fold only; everything else uses only the user's own prior orders
          (plus the time slot of the order to be predicted for the 16 current-time columns).
Run     : cd src && python -m feature.build [--folds 1 2 3 4 5]       (~30 min, ~12 GB RAM; needs data.split first)
"""
import argparse
import numpy as np
import pandas as pd

import common
from feature import base as bd
from feature import time_feat as bt

FEATURES = common.FEATURES_DIR
FINAL_COLS = common.FEATURES            # the 54 features (order used everywhere)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=bd.DEFAULT_FOLDS)
    a = ap.parse_args()
    splits = {k: pd.read_csv(bd.SPLITS / f"fold{k}.csv", usecols=["user_id", "split"]).set_index("user_id").split for k in a.folds}
    prior, rows, products, tgt = bd.load_inputs(splits[a.folds[0]].index, extended=True)
    f = bd.build_base_features(prior, rows, products, extended=True)      # no target-order columns at all
    lab = pd.read_csv(bd.RAW / "order_products__train.csv", usecols=["order_id", "product_id"]).merge(
        tgt.reset_index()[["order_id", "user_id"]], on="order_id")[bd.KEYS]
    lab["label"] = np.int8(1)
    f = f.merge(lab, on=bd.KEYS, how="left"); f["label"] = f.label.fillna(0).astype(np.int8)
    o, r, t = bt.prepare(prior, rows, products, tgt)
    tbase = bt.user_part(o, r, t)
    am = products.set_index("product_id")
    tbase["aisle_id"] = tbase.product_id.map(am.aisle_id); tbase["department_id"] = tbase.product_id.map(am.department_id)
    del rows, o
    print(f"base features built: {len(f):,} pairs", flush=True)
    FEATURES.mkdir(exist_ok=True)
    for k in a.folds:
        train_u = splits[k].index[splits[k] == "train"]
        stats = bd.add_product_stats(f, train_u, extended=True)
        tm = bt.lifts(tbase.copy(), r, t, train_u).sort_values(bd.KEYS).reset_index(drop=True)[bd.KEYS + bt.TIME_COLS]
        assert (tm.user_id.values == f.user_id.values).all() and (tm.product_id.values == f.product_id.values).all(), "time features cover different pairs"
        d = pd.concat([f[bd.KEYS + ["label"]], f.drop(columns=bd.KEYS + ["label"], errors="ignore"), stats, tm.drop(columns=bd.KEYS)], axis=1)
        d["split"] = pd.Categorical(f.user_id.map(splits[k]), categories=["train", "val", "test"])
        d = d[bd.KEYS + ["label", "split"] + FINAL_COLS]
        for c in FINAL_COLS:
            d[c] = d[c].astype(np.float32)
        d.to_parquet(FEATURES / f"fold{k}.parquet", index=False)
        print(f"fold {k}: {len(d):,} rows, {len(FINAL_COLS)} features -> {FEATURES / f'fold{k}.parquet'}", flush=True)
        del d, stats, tm


if __name__ == "__main__":
    main()
