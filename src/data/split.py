"""User-level 5-fold cross-validation split (identical to baseline_eval/eval_cv.py with --cv_folds 5 --cv_seed 42).

Population : the 131,209 users whose last order is labelled (orders.csv eval_set == 'train').
Outer fold : StratifiedKFold(5, shuffle, random_state=42) on users, stratified by 5 quantile bins of the number of prior orders.
             Fold k is the TEST set of run k (26,242 users, 20 %); every user is tested exactly once.
Train/val  : the other 4 folds are cut into train 85 % / val 15 % (train_test_split, random_state = 42 + k, same stratification),
             i.e. about 68 % / 12 % / 20 % of all users.
Outputs    : features/split/fold{k}.csv   (user_id, split, n_prior_orders)   k = 1..5
Run        : cd src && python -m data.split          (seconds)
Next       : python -m feature.build  builds features/fold{k}.parquet.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
SPLITS = ROOT / "features" / "split"
N_FOLDS, N_BINS, CV_SEED, VAL_SIZE = 5, 5, 42, 0.15


def eligible_users():
    """Same order / stratification as eval_cv.build_cv_folds (the order matters for reproducing the folds)."""
    orders = pd.read_csv(RAW / "orders.csv", usecols=["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"], dtype={"eval_set": "category"})
    eligible = orders.loc[orders.eval_set == "train", "user_id"].unique()
    n_prior = orders[orders.eval_set == "prior"].groupby("user_id").size().reindex(eligible).fillna(0)
    stratum = pd.qcut(n_prior, N_BINS, labels=False, duplicates="drop")
    return pd.DataFrame({"user_id": eligible, "stratum": stratum.to_numpy(), "n_prior_orders": n_prior.to_numpy().astype(int)})


def make_folds(users):
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (trainval_idx, test_idx) in enumerate(outer.split(users.user_id, users.stratum), start=1):
        trainval = users.iloc[trainval_idx]
        train_idx, val_idx = train_test_split(trainval.index.to_numpy(), test_size=VAL_SIZE, random_state=CV_SEED + fold, stratify=trainval.stratum)
        split = pd.Series("test", index=users.index)
        split[train_idx] = "train"; split[val_idx] = "val"
        yield fold, split


def main():
    argparse.ArgumentParser().parse_args()
    SPLITS.mkdir(parents=True, exist_ok=True)
    users = eligible_users()
    print(f"{len(users):,} users; prior orders per user: median {users.n_prior_orders.median():.0f}; {N_FOLDS}-fold CV, {N_BINS} strata")
    tested = pd.Series(0, index=users.user_id)
    print("\nfold  split   users   share   mean prior orders")
    for fold, split in make_folds(users):
        out = users[["user_id"]].assign(split=split.values, n_prior_orders=users.n_prior_orders.values)
        out.to_csv(SPLITS / f"fold{fold}.csv", index=False)
        tested[out.user_id[out.split == "test"]] += 1
        for s in ("train", "val", "test"):
            u = out[out.split == s]
            print(f"{fold:4d}  {s:5s} {len(u):7,}  {len(u) / len(out):6.1%}  {u.n_prior_orders.mean():8.2f}")
    assert (tested == 1).all(), "every user must be a test user exactly once"
    print("\nOK: every user is in the test set of exactly one fold")


if __name__ == "__main__":
    main()
