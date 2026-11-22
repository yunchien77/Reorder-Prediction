"""USDA FoodData Central (Branded Foods, US) -> product-level features (name match; NaN where no match).
Input  : dataset/external/fdc/branded/{food,branded_food,food_nutrient}.csv   (unzip the Branded CSV zip; see README)
Output : features/external/fdc.parquet   product_id, fdc_score, fdc_kcal, fdc_protein, fdc_fat, fdc_satfat, fdc_carb, fdc_sugar, fdc_fiber,
         fdc_sodium (per 100 g / 100 ml), fdc_n_ingredients, fdc_serving_g, fdc_cat (branded food category, integer code)"""
import numpy as np
import pandas as pd

from external import names

D = names.EXTERNAL / "fdc" / "branded"
NUT = {1008: "fdc_kcal", 1003: "fdc_protein", 1004: "fdc_fat", 1258: "fdc_satfat", 1005: "fdc_carb", 2000: "fdc_sugar", 1079: "fdc_fiber", 1093: "fdc_sodium"}


def build():
    food = pd.read_csv(D / "food.csv", usecols=["fdc_id", "data_type", "description"])
    food = food[food.data_type == "branded_food"]
    b = pd.read_csv(D / "branded_food.csv", usecols=["fdc_id", "ingredients", "branded_food_category", "serving_size", "serving_size_unit", "market_country"], low_memory=False)
    c = food.merge(b, on="fdc_id")
    c = c[c.market_country.fillna("United States") == "United States"].reset_index(drop=True)
    p = names.products(); q = p[p.is_food].reset_index(drop=True)
    idx, score = names.top1(q.product_name.values, c.description.fillna("").values)
    hit = idx >= 0
    out = pd.DataFrame({"product_id": q.product_id.values, "fdc_score": score})
    m = c.iloc[np.where(hit, idx, 0)].reset_index(drop=True)
    out["fdc_id"] = np.where(hit, m.fdc_id.values, -1)
    out["fdc_n_ingredients"] = np.where(hit, m.ingredients.fillna("").str.count(",").add(1).where(m.ingredients.notna()).values, np.nan)
    out["fdc_serving_g"] = np.where(hit & m.serving_size_unit.isin(["g", "ml"]).values, m.serving_size.values, np.nan)
    cats = pd.Series(c.branded_food_category.fillna("(none)")); codes = {k: i for i, k in enumerate(sorted(cats.unique()))}
    out["fdc_cat"] = np.where(hit, m.branded_food_category.fillna("(none)").map(codes).values, np.nan)
    # nutrients of the matched products (food_nutrient.csv is 1.5 GB: read in chunks, keep only the needed rows)
    want = set(out.fdc_id[out.fdc_id > 0]); parts = []
    for ch in pd.read_csv(D / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"], chunksize=2_000_000):
        parts.append(ch[ch.fdc_id.isin(want) & ch.nutrient_id.isin(NUT)])
    n = pd.concat(parts).drop_duplicates(["fdc_id", "nutrient_id"]).pivot(index="fdc_id", columns="nutrient_id", values="amount").rename(columns=NUT)
    out = out.merge(n, left_on="fdc_id", right_index=True, how="left").drop(columns="fdc_id")
    out = out.merge(p[["product_id"]], on="product_id", how="right")            # non-food products: all NaN
    out["fdc_score"] = out.fdc_score.fillna(0)
    names.OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(names.OUT / "fdc.parquet", index=False)
    print("fdc:", len(out), "products;", int((out.fdc_score >= 0.9).sum()), "with score >= 0.9;", int((out.fdc_score >= 0.8).sum()), "with score >= 0.8")
    return out
