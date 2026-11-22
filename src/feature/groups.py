"""The 12 feature groups (G1-G12) that partition the 54 final features; used by importance / ablation / report."""
import common

GROUPS = {
    "G1 買得多不多、近不近（頻率／衰減／連續）": ["up_times_purchased", "up_order_rate", "decay_freq_b05", "decay_freq_b08", "decay_freq_b09", "in_last_3", "in_last_5",
                                          "up_streak", "up_rank_in_user", "up_rate_since_first", "first_on"],
    "G2 距上次買多久（訂單數／天數／最近幾次間隔）": ["up_orders_since_last", "up_days_since_last_prior", "lag1", "lag2", "lag3"],
    "G3 購買週期 cadence": ["gap_mean", "gap_std", "since_over_gap"],
    "G4 商品統計（回購率／熱門度／走道部門回購率／rep_score）": common.PRODUCT,
    "G5 使用者整體習慣": ["user_total_orders", "user_distinct_products", "user_avg_basket_size", "user_reorder_ratio", "user_mean_gap_days"],
    "G6 使用者的走道／部門偏好": ["user_aisle_share", "user_dept_share"],
    "G7 購物車位置": common.EXT_CART,
    "G8 走道／部門近因": common.EXT_AISLE,
    "G9 商品層級週期": common.EXT_PGAP,
    "G10 當前時段（星期幾、幾點）": ["target_dow", "target_hour_bin"],
    "G11 時段 × 使用者／此商品的習慣": ["user_dow_share", "user_hour_share", "up_dow_purchase_count", "up_hour_purchase_count", "up_dow_share", "up_hour_share", "up_dow_diff", "up_hour_diff"],
    "G12 時段 × 商品／走道／部門": ["prod_dow_lift", "aisle_dow_lift", "dept_dow_lift", "prod_hour_lift", "aisle_hour_lift", "dept_hour_lift"],
}
POOL = common.FEATURES
assert sorted(sum(GROUPS.values(), [])) == sorted(POOL), "groups must partition the pool"
