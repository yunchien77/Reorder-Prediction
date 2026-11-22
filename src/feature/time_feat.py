"""Time-of-day / day-of-week features that are tied to the PRODUCT (e.g. "this product is bought more in the morning").
They use the time of the order to be predicted (target order's day of week and hour) -> only valid when predicting at order time,
like the other target-order features.

Slots : hour bin 0..3 = [0-9], [10-13], [14-17], [18-23] ;  day of week 0..6 (as in orders.csv)
Columns (16):
  target_dow, target_hour_bin                       the slot of the order to be predicted
  user_dow_share, user_hour_share                   share of the user's prior orders in that slot
  up_dow_purchase_count, up_hour_purchase_count     times the user bought THIS product in that slot
  up_dow_share, up_hour_share                       same as a share, smoothed toward the user's overall slot share (k = 2)
  up_dow_diff, up_hour_diff                         up_*_share - user_*_share  (>0: the user buys this product especially in this slot)
  prod_/aisle_/dept_ + dow_lift / hour_lift         how over-represented the slot is among purchases of the product / aisle / department,
                                                    P(slot | entity) / P(slot), smoothed (K = 20), from TRAIN users' prior orders only
Used by build.py (functions only).
"""
import numpy as np
import pandas as pd
from feature import base as bd

K_USER, K_ENT = 2.0, 20.0
TIME_COLS = ["target_dow", "target_hour_bin", "user_dow_share", "user_hour_share", "up_dow_purchase_count", "up_hour_purchase_count",
             "up_dow_share", "up_hour_share", "up_dow_diff", "up_hour_diff",
             "prod_dow_lift", "aisle_dow_lift", "dept_dow_lift", "prod_hour_lift", "aisle_hour_lift", "dept_hour_lift"]


def hbin(h):
    return np.digitize(np.asarray(h), [10, 14, 18]).astype(np.int8)        # 0: <10, 1: 10-13, 2: 14-17, 3: >=18


def prepare(prior, rows, products, tgt):
    """Purchase rows with the slot of the order they happened in + one row per (user, product) with the target slot."""
    o = prior[["user_id", "order_number", "order_dow", "order_hour_of_day"]].copy()
    o["hb"] = hbin(o.order_hour_of_day); o["dow"] = o.order_dow.astype(np.int8)
    r = rows[["user_id", "product_id", "order_number"]].merge(o[["user_id", "order_number", "hb", "dow"]], on=["user_id", "order_number"])
    r = r.merge(products[["product_id", "aisle_id", "department_id"]], on="product_id")
    t = pd.DataFrame({"tdow": tgt.order_dow.astype(np.int8), "thb": hbin(tgt.order_hour_of_day)})
    return o, r, t


def user_part(o, r, t):
    f = r.groupby(bd.KEYS).size().rename("n_up").reset_index()
    f = f.merge(t, left_on="user_id", right_index=True)
    rt = r.merge(t, left_on="user_id", right_index=True)
    for name, col, tcol in (("dow", "dow", "tdow"), ("hour", "hb", "thb")):
        c = rt[rt[col] == rt[tcol]].groupby(bd.KEYS).size().rename(f"up_{name}_purchase_count").reset_index()
        f = f.merge(c, on=bd.KEYS, how="left"); f[f"up_{name}_purchase_count"] = f[f"up_{name}_purchase_count"].fillna(0)
        ot = o.merge(t, left_on="user_id", right_index=True)
        share = (ot[col] == ot[tcol]).groupby(ot.user_id).mean().rename(f"user_{name}_share")
        f = f.merge(share, left_on="user_id", right_index=True)
        f[f"up_{name}_share"] = (f[f"up_{name}_purchase_count"] + K_USER * f[f"user_{name}_share"]) / (f.n_up + K_USER)
        f[f"up_{name}_diff"] = f[f"up_{name}_share"] - f[f"user_{name}_share"]
    f["target_dow"] = f.user_id.map(t.tdow); f["target_hour_bin"] = f.user_id.map(t.thb)
    return f


def lifts(f, r, t, train_users):
    """P(slot | entity) / P(slot) from the train users' prior purchases, looked up at each user's target slot."""
    rr = r[r.user_id.isin(train_users)]
    for name, col, tcol in (("dow", "dow", "tdow"), ("hour", "hb", "thb")):
        p = rr[col].value_counts(normalize=True)
        slot = f.user_id.map(t[tcol]).values
        pg = p.reindex(slot).values
        for ent, key in (("prod", "product_id"), ("aisle", "aisle_id"), ("dept", "department_id")):
            cnt = rr.groupby([key, col]).size()
            n = rr.groupby(key).size()
            c = cnt.reindex(pd.MultiIndex.from_arrays([f[key].values, slot])).fillna(0).values
            nn = f[key].map(n).fillna(0).values
            f[f"{ent}_{name}_lift"] = ((c + K_ENT * pg) / (nn + K_ENT)) / pg       # unseen entity -> 1.0
    return f


def build(prior, rows, products, tgt, train_users):
    o, r, t = prepare(prior, rows, products, tgt)
    f = user_part(o, r, t)
    am = products.set_index("product_id")
    f["aisle_id"] = f.product_id.map(am.aisle_id); f["department_id"] = f.product_id.map(am.department_id)
    f = lifts(f, r, t, train_users)
    f = f.sort_values(bd.KEYS).reset_index(drop=True)
    return f[bd.KEYS + TIME_COLS]
