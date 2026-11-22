"""No-training baselines: rank by one column (up_order_rate, rep_score, up_times_purchased).
   cd src && python -m model.rules --folds 1 2 3 4 5"""
import argparse
import numpy as np
import common

RULES = ["up_order_rate", "rep_score", "up_times_purchased"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    for fold in ap.parse_args().folds:
        val, test = common.load(fold, "val", RULES), common.load(fold, "test", RULES)
        for r in RULES:
            row = dict(model=f"rule:{r}", n_features=1, fold=fold,
                       **common.evaluate(val, test, val[r].values.astype(np.float64), test[r].values.astype(np.float64)))
            common.save_row(row); common.report(row)
