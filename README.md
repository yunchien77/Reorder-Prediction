# Instacart 回購預測

## 0. 前置安裝

```bash
conda create -n imba python=3.11 -y
conda activate imba
pip install -r requirements.txt
```

資料從 Kaggle 的 [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) 下載，解壓縮到 `dataset/raw/`
（需要 `orders.csv`、`order_products__prior.csv`、`order_products__train.csv`、`products.csv`）。

以下指令都在 `src/` 底下執行：

```bash
cd src
```

## 1. 切分資料集

5-fold（seed 42），輸出 `features/split/fold{k}.csv`。

```bash
python -m data.split
```

## 2. 建特徵

54 個特徵、12 類；商品統計每折只用 train 使用者計算。輸出 `features/fold{k}.parquet`。

```bash
python -m feature.build
```

## 3. 模型

結果寫入 `results/results.csv`，預測存在 `results/preds/`。

```bash
python -m model.rules                          # 規則基準（rep_score 等）
python -m model.lr                             # 同理：rf、lgbm、xgb、cat、mlp
```

## 4. Ensemble 與決策規則

需先跑完 xgb、lgbm、cat、mlp。

```bash
python -m ensemble.blend --name ens54_mean --mode mean --members xgb lgbm catboost mlp
python -m ensemble.weights --members xgb lgbm catboost mlp   # 各種權重
python -m ensemble.decision --model xgb                      # 共用門檻 vs 每人期望 F1（也可換 --model ens54_mean）
```

## 5. 分析

`robustness` 需先跑完規則、xgb 與 `ens54_mean`。

```bash
python -m analysis.ablation --model xgb   # 特徵類別消融（可換 lgbm、cat、lr、rf、mlp）
python -m analysis.importance             # 特徵重要度（XGBoost，5 fold）
python -m analysis.robustness             # 分群穩健性與 bootstrap（5 fold）
```

## 6. External Data

資料放在 `dataset/external/`：`foodkeeper/foodkeeper.json`（[FoodKeeper](https://catalog.data.gov/dataset/fsis-foodkeeper-data)）、`fdc/branded/`（[FoodData Central](https://fdc.nal.usda.gov/download-datasets) 的 Branded Foods CSV，解壓出 `food.csv`、`branded_food.csv`、`food_nutrient.csv`）、`off/en.openfoodfacts.org.products.csv.gz`（[Open Food Facts](https://world.openfoodfacts.org/data)）。

外部特徵不動原本 54 個，另存 `features/external/`；沒配對到就是缺失值。

```bash
python -m external.build                       # 建商品層級外部特徵（fk / fdc / off）
python -m analysis.external --thr 0.9 0.8      # 原本 54 個 + 外部特徵（XGBoost，5 折），結果 results/external.csv
```
