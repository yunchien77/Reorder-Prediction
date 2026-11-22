"""Open Food Facts (products sold in the United States) -> product-level features (name match; NaN where no match).
Input  : dataset/external/off/en.openfoodfacts.org.products.csv.gz   (tab separated, read in chunks, never unpacked)
Output : features/external/off.parquet   product_id, off_score, off_kcal, off_fat, off_satfat, off_carb, off_sugar, off_fiber, off_protein, off_salt,
         off_nutriscore (a..e -> 1..5), off_nova (1..4), off_additives_n, off_pnns1 (food group, integer code)"""
import numpy as np
import pandas as pd

from external import names

F = names.EXTERNAL / "off" / "en.openfoodfacts.org.products.csv.gz"
COLS = {"energy-kcal_100g": "off_kcal", "fat_100g": "off_fat", "saturated-fat_100g": "off_satfat", "carbohydrates_100g": "off_carb", "sugars_100g": "off_sugar",
        "fiber_100g": "off_fiber", "proteins_100g": "off_protein", "salt_100g": "off_salt", "nova_group": "off_nova", "additives_n": "off_additives_n"}


def build():
    use = ["product_name", "countries_tags", "nutriscore_grade", "pnns_groups_1"] + list(COLS)
    parts = []
    for ch in pd.read_csv(F, sep="\t", usecols=use, chunksize=300_000, low_memory=False, on_bad_lines="skip"):
        parts.append(ch[ch.countries_tags.fillna("").str.contains("en:united-states") & ch.product_name.notna()])
    o = pd.concat(parts).drop(columns="countries_tags").reset_index(drop=True)
    p = names.products(); q = p[p.is_food].reset_index(drop=True)
    idx, score = names.top1(q.product_name.values, o.product_name.values)
    hit = idx >= 0
    m = o.iloc[np.where(hit, idx, 0)].reset_index(drop=True)
    out = pd.DataFrame({"product_id": q.product_id.values, "off_score": score})
    for src, dst in COLS.items():
        out[dst] = np.where(hit, pd.to_numeric(m[src], errors="coerce").values, np.nan)
    out["off_nutriscore"] = np.where(hit, m.nutriscore_grade.map({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}).values, np.nan)
    codes = {k: i for i, k in enumerate(sorted(o.pnns_groups_1.dropna().unique()))}
    out["off_pnns1"] = np.where(hit, m.pnns_groups_1.map(codes).values, np.nan)
    out = out.merge(p[["product_id"]], on="product_id", how="right")
    out["off_score"] = out.off_score.fillna(0)
    names.OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(names.OUT / "off.parquet", index=False)
    print("off:", len(o), "US products in the corpus;", int((out.off_score >= 0.9).sum()), "products with score >= 0.9;", int((out.off_score >= 0.8).sum()), "with score >= 0.8")
    return out
