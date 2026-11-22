"""Feature builders (functions) used by build.py: load the raw tables, build the split-independent features and the train-users-only product statistics.
   Reads features/split/fold{k}.csv (made by data/split.py).
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
SPLITS = ROOT / "features" / "split"
DEFAULT_FOLDS = [1, 2, 3, 4, 5]
BETAS = {"b05": 0.5, "b08": 0.8, "b09": 0.9}   # decay: sum of beta ** (orders ago - 1); b08 is the TREx value
SMOOTH = 20                                      # shrink product/aisle/department reorder rates toward the global rate
KEYS = ["user_id", "product_id"]
HELPERS = ["reord_sum", "aisle_id", "department_id"]
STAT_COLS = ["product_reorder_rate", "product_popularity", "aisle_reorder_rate", "department_reorder_rate", "rep_score"]
# extra prior-only statistics of the product over train users
EXT_STAT_COLS = ["product_avg_cart_pos", "product_mean_gap", "gap_ratio_to_product"]


def load_inputs(user_ids, extended=False):
    """Orders, prior purchase rows (with days since the user's first order) and target orders for `user_ids`."""
    orders = pd.read_csv(RAW / "orders.csv", dtype={"eval_set": "category"})
    orders = orders[orders.user_id.isin(user_ids)]
    prior = orders[orders.eval_set == "prior"].sort_values(["user_id", "order_number"]).copy()
    # days_since_prior_order is capped at 30 in Instacart, so cumulative days under-estimate long gaps
    prior["days_elapsed"] = prior.groupby("user_id")["days_since_prior_order"].cumsum().fillna(0)
    tgt = orders[orders.eval_set == "train"].set_index("user_id")
    op = pd.read_csv(RAW / "order_products__prior.csv", usecols=["order_id", "product_id", "reordered"] + (["add_to_cart_order"] if extended else []))
    rows = op.merge(prior[["order_id", "user_id", "order_number", "days_elapsed"]], on="order_id")
    if extended:   # cart position relative to the size of the order (0..1)
        rows["cart_rel"] = rows.add_to_cart_order / rows.groupby("order_id").add_to_cart_order.transform("max")
    products = pd.read_csv(RAW / "products.csv", usecols=["product_id", "aisle_id", "department_id"])
    return prior, rows, products, tgt


def build_base_features(prior, rows, products, extended=False):
    """One row per (user, product bought in prior orders) with the 26 split-independent features (+ helper columns)."""
    user_info = prior.groupby("user_id").agg(n_prior=("order_number", "size"), last_day=("days_elapsed", "max")).reset_index()
    ru = rows[["user_id", "product_id", "reordered"]].merge(products, on="product_id")

    # ---- user-level habits (prior orders only) ----
    uf = user_info.merge(ru.groupby("user_id").size().rename("n_rows"), on="user_id")
    uf["user_avg_basket_size"] = uf.n_rows / uf.n_prior
    uf = uf.merge(ru.groupby("user_id").reordered.mean().rename("user_reorder_ratio"), on="user_id")
    uf = uf.merge(prior.groupby("user_id").days_since_prior_order.mean().rename("user_mean_gap_days"), on="user_id")
    ua = ru.groupby(["user_id", "aisle_id"]).size().rename("ua").reset_index()
    ud = ru.groupby(["user_id", "department_id"]).size().rename("ud").reset_index()
    del ru

    # ---- (user, product) interaction features ----
    r = rows[["user_id", "product_id", "order_number", "days_elapsed", "reordered"] + (["add_to_cart_order", "cart_rel"] if extended else [])].merge(user_info, on="user_id")
    r["age"] = r.n_prior - r.order_number                        # 0 = the user's last prior order
    for tag, b in BETAS.items():
        r[f"w_{tag}"] = b ** r.age
    r["last3"] = (r.order_number > r.n_prior - 3).astype(np.int8)
    r["last5"] = (r.order_number > r.n_prior - 5).astype(np.int8)
    r = r.sort_values(KEYS + ["order_number"], ascending=[True, True, False])
    g = r.groupby(KEYS)
    r["k"] = g.cumcount()                                        # 0 = most recent purchase of this product
    r["ago"] = r.age + 1                                         # "k orders ago", 1 = the previous order
    r["hit"] = (r.ago == r.k + 1).astype(np.int8)
    r["streak_ok"] = g.hit.cummin()
    r["gap"] = -g.order_number.diff()                            # orders between two consecutive purchases
    agg = dict(up_times_purchased=("order_number", "size"), last_on=("order_number", "max"), first_on=("order_number", "min"),
               last_de=("days_elapsed", "max"), n_prior=("n_prior", "first"), last_day=("last_day", "first"),
               in_last_3=("last3", "sum"), in_last_5=("last5", "sum"), up_streak=("streak_ok", "sum"),
               gap_mean=("gap", "mean"), gap_std=("gap", "std"), reord_sum=("reordered", "sum"))
    for tag in BETAS:
        agg[f"decay_freq_{tag}"] = (f"w_{tag}", "sum")
    if extended:
        agg.update(up_avg_cart_pos=("add_to_cart_order", "mean"), up_avg_cart_rel=("cart_rel", "mean"))
    f = r.groupby(KEYS).agg(**agg).reset_index()
    f["up_order_rate"] = f.up_times_purchased / f.n_prior
    f["up_orders_since_last"] = f.n_prior - f.last_on            # 0 = bought in the last prior order
    f["up_days_since_last_prior"] = f.last_day - f.last_de
    for j in (1, 2, 3):
        f = f.merge(r[r.k == j - 1][KEYS + ["ago"]].rename(columns={"ago": f"lag{j}"}), on=KEYS, how="left")
    del r
    f["since_over_gap"] = (f.up_orders_since_last + 1) / f.gap_mean   # >1: past the usual repurchase interval
    f["up_rate_since_first"] = f.up_times_purchased / (f.n_prior - f.first_on + 1)
    f["up_rank_in_user"] = f.groupby("user_id").up_times_purchased.rank(pct=True)
    f["user_distinct_products"] = f.groupby("user_id").up_times_purchased.transform("size")
    f = f.rename(columns={"n_prior": "user_total_orders"})

    # ---- user habits, aisle/department share ----
    f = f.merge(uf[["user_id", "n_rows", "user_avg_basket_size", "user_reorder_ratio", "user_mean_gap_days"]], on="user_id")
    f = f.merge(products, on="product_id")
    f = f.merge(ua, on=["user_id", "aisle_id"], how="left").merge(ud, on=["user_id", "department_id"], how="left")
    f["user_aisle_share"] = f.ua / f.n_rows
    f["user_dept_share"] = f.ud / f.n_rows

    if extended:      # how long ago did the user last buy ANY product of the same aisle / department (prior orders only)
        for key, name in (("aisle_id", "aisle_orders_since_last"), ("department_id", "dept_orders_since_last")):
            last = rows[["user_id", "product_id", "order_number"]].merge(products[["product_id", key]], on="product_id") \
                .groupby(["user_id", key]).order_number.max().rename("_last").reset_index()
            f = f.merge(last, on=["user_id", key], how="left")
            f[name] = f.user_total_orders - f.pop("_last")
    return f.drop(columns=["last_on", "last_de", "last_day", "n_rows", "ua", "ud"]).sort_values(KEYS).reset_index(drop=True)


def add_product_stats(f, train_users, extended=False):
    """The 5 statistics that depend on the split: reorder rate / popularity of product, aisle, department from TRAIN users only.
    Needs the helper columns reord_sum, aisle_id, department_id of `f`.  Returns a frame aligned with f.index."""
    t = f.loc[f.user_id.isin(train_users), ["product_id", "aisle_id", "department_id", "up_times_purchased", "reord_sum"]]
    glob = float(t.reord_sum.sum() / t.up_times_purchased.sum())       # global reorder rate of the train users' prior rows

    def stat(key):
        g = t.groupby(key)[["reord_sum", "up_times_purchased"]].sum()
        return (g.reord_sum + SMOOTH * glob) / (g.up_times_purchased + SMOOTH), np.log1p(g.up_times_purchased)
    prod_rate, prod_pop = stat("product_id")
    aisle_rate, _ = stat("aisle_id")
    dept_rate, _ = stat("department_id")
    out = pd.DataFrame(index=f.index)
    out["product_reorder_rate"] = f.product_id.map(prod_rate).fillna(glob)        # unseen in train users -> global rate
    out["product_popularity"] = f.product_id.map(prod_pop).fillna(0)
    out["aisle_reorder_rate"] = f.aisle_id.map(aisle_rate).fillna(glob)
    out["department_reorder_rate"] = f.department_id.map(dept_rate).fillna(glob)
    out["rep_score"] = f.decay_freq_b08 * out.product_reorder_rate                # simplified TREx-Rep style score
    if extended:   # prior-only statistics of the product over TRAIN users (no label involved)
        tr = f.loc[f.user_id.isin(train_users), ["product_id", "up_avg_cart_pos", "gap_mean"]]
        out["product_avg_cart_pos"] = f.product_id.map(tr.groupby("product_id").up_avg_cart_pos.mean())
        pgap = f.product_id.map(tr.groupby("product_id").gap_mean.mean())
        out["product_mean_gap"] = pgap
        out["gap_ratio_to_product"] = f.gap_mean / pgap                                # >1: this user waits longer than most users
    return out


def build_features(prior, rows, products, train_users, extended=False):
    """All base features + the train-users-only statistics."""
    f = build_base_features(prior, rows, products, extended)
    return pd.concat([f, add_product_stats(f, train_users, extended)], axis=1).drop(columns=HELPERS)
