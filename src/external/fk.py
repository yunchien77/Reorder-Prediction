"""USDA FoodKeeper (shelf life) -> product-level features, through a hand-made map  Instacart aisle -> FoodKeeper items.
FoodKeeper lists generic foods (not brands), so products are linked by AISLE, not by name; aisles that are not food or are not about one kind of food
(international / cuisine aisles, 'missing', 'other', ...) stay NaN.  Every product of an aisle gets the median over the mapped FoodKeeper items.
Input  : dataset/external/foodkeeper/foodkeeper.json
Output : features/external/fk.parquet   product_id, fk_pantry_days, fk_fridge_days, fk_freeze_days, fk_shelf_days (shortest of pantry / fridge, else freezer)
Map entries: 'text' = substring of the lower-cased item label "name (subtitle)";  '=text' = exact label;  'Category|text' = only in that FoodKeeper category."""
import json

import numpy as np
import pandas as pd

from external import names

DAYS = {"Days": 1, "Weeks": 7, "Months": 30, "Years": 365, "Indefinitely": 3650}
FZ, BB = "Food Purchased Frozen", "Baby Food"
GRAINS = ["rice (white", "rice (brown", "quinoa (uncooked", "beans (dried", "lentils (dried", "oats (whole grain"]
AISLE_MAP = {
    "fresh fruits": ["apples", "bananas", "berries (", "blueberries", "citrus fruit", "grapes", "melons", "peaches, nectarines", "pineapple", "strawberries", "avocados"],
    "fresh vegetables": ["broccoli and", "carrots, parsnips", "celery", "cucumbers", "lettuce (", "peppers", "onions (yellow", "tomatoes", "potatoes", "cabbage", "mushrooms", "squash (summer"],
    "packaged vegetables fruits": ["bagged greens", "baby carrots", "cherry tomatoes", "fruit, cut"], "packaged produce": ["bagged greens", "fruit, cut"],
    "fresh herbs": ["herbs", "cilantro", "parsley (fresh", "basil (fresh", "=mint"], "frozen produce": [f"{FZ}|=vegetables", f"{FZ}|fruits (such as berries"],
    "prepared soups salads": ["soup, stews", "chicken salad", "potato salad", "pasta salad", "coleslaw"], "specialty cheeses": ["cheese (soft", "cheese (hard"],
    "packaged cheese": ["cheese (hard", "cheese (shredded", "cheese (processed slices", "string cheese", "cream cheese"], "other creams cheeses": ["cream cheese", "cottage cheese", "ricotta", "sour cream"],
    "energy granola bars": ["cereal or granola bars", "=granola"], "instant foods": ["main dishes or meals", "potatoes (instant", "soup mixes"],
    "marinades meat preparation": ["marinades"], "packaged meat": ["pork (ground", "beef (ground", "=bacon", "sausage (raw", "=hot dogs", "luncheon meat or poultry (pre-packaged"],
    "bakery desserts": ["commercial cakes", "cheesecake", "cookies (soft", "pastries, danish", "fruit pies", "doughnuts"], "pasta sauce": ["spaghetti sauce (in jars"],
    "fresh pasta": ["=fresh pasta"], "prepared meals": ["main dishes or meals", "commercial brand vacuum-packed dinners", "casseroles"],
    "tofu meat alternatives": ["=tofu", "=tempeh", "soy meat substitutes", "=miso"], "frozen vegan vegetarian": ["soy meat substitutes", "=tempeh", "soy crumbles"],
    "packaged seafood": ["lean fish (cod", "fatty fish", "cooked fish", "surimi", "shrimp, crayfish"], "seafood counter": ["lean fish (cod", "fatty fish", "scallops", "crab meat (fresh", "shrimp, crayfish"],
    "baking ingredients": ["=baking powder", "=baking soda", "flour (white", "sugar (granulated", "=cornstarch", "cocoa and cocoa", "yeast"],
    "bulk dried fruits vegetables": ["fruits, dried", "mushrooms (dried"], "oils vinegars": ["oils (olive", "=vinegar", "=canola oil", "=coconut oil", "=sesame oil", "balsamic"],
    "popcorn jerky": ["popcorn", "jerky (commercially"], "coffee": ["coffee (whole beans", "coffee (instant", "coffee (commercial ground", "coffee (home ground"],
    "red wines": ["=red wine"], "white wines": ["=white wine"], "specialty wines champagnes": ["=red wine", "=white wine"], "spirits": ["hard liquors"],
    "honeys syrups nectars": ["=honey", "=syrup", "=molasses"], "refrigerated": ["juice, boxes", "vegetable juice (commercial, sold refrigerated", "orange juice (commercially packaged carton"],
    "frozen meat seafood": [f"{FZ}|shrimp", f"{FZ}|sausages", f"{FZ}|fish (breaded", f"{FZ}|lobster"],
    "poultry counter": ["chicken (whole)", "chicken parts (breast halves, boneless", "turkey (whole)", "ground turkey or chicken"],
    "packaged poultry": ["chicken (pre-packaged", "turkey  (pre-packaged", "chicken nuggets"], "butter": ["=butter", "=margarine"],
    "ice cream ice": ["=ice cream", "ice pops", "sherbet"], "frozen dessert": ["=ice cream", "sherbet", "ice pops"], "frozen meals": ["frozen entrees", "pizza (frozen"], "frozen pizza": ["pizza (frozen"],
    "buns rolls": ["commercial bread products"], "bread": ["commercial bread products", "whole wheat bread (commercially"], "candy chocolate": ["chocolate (unsweetened", "=marshmallows", "gummy"],
    "breakfast bars pastries": ["toaster pastries", "cereal or granola bars", "muffin (commercially packaged"], "fruit vegetable snacks": ["gummy (fruit) snacks", "fruits, dried"],
    "preserved dips spreads": ["jams, jellies", "peanut butter (commercial", "=tahini", "=chutney", "=guacamole", "hummus (commercial"],
    "spreads": ["peanut butter (commercial", "jams, jellies", "=almond butter", "chocolate hazlenut"], "frozen breakfast": ["pancakes, waffles"],
    "cream": ["cream (heavy", "cream (half and half", "cream (whipping"], "frozen breads doughs": ["dough (commercial", "bagel  (commercially frozen", "cookie dough"],
    "canned meals beans": ["canned goods (low acid"], "canned jarred vegetables": ["canned goods (low acid", "canned goods (high acid", "tomato paste", "tomato sauce (spaghetti"],
    "cookies cakes": ["cookies (crispy", "cookies (soft", "commercial cakes"], "grains rice dried goods": GRAINS, "bulk grains rice dried goods": GRAINS,
    "energy sports drinks": ["nutrition supplement drinks (bottled", "diet powder mixes"], "fresh dips tapenades": ["=guacamole", "hummus", "salsa (fresh", "=tapenade"],
    "soup broth bouillon": ["beef broth", "chicken broth", "vegetable stock", "soup mixes"], "refrigerated pudding desserts": ["=pudding"],
    "condiments": ["ketchup", "=mustard", "mayonnaise", "=hot sauce", "=relish", "soy sauce"], "soft drinks": ["soda ("], "crackers": ["=crackers", "graham cracker"],
    "milk": ["milk (plain", "milk (lactose", "milk  (shelf-stable", "milk  (ultra"], "eggs": ["eggs (in shell"], "soy lactosefree": ["=soy milk", "=almond milk", "milk (lactose", "=rice milk"],
    "salad dressing toppings": ["salad dressings (commercial", "=croutons", "bacon bits"], "cocoa drink mixes": ["cocoa and cocoa", "diet powder mixes"], "baby food formula": [f"{BB}|"],
    "breakfast bakery": ["bagel (fresh", "muffin (commercially packaged", "pastries, danish", "cinnamon rolls"], "tea": ["tea (bags", "tea (loose", "tea (instant"],
    "canned meat seafood": ["tuna (canned", "seafood (canned", "canned chicken", "meat products (canned"],
    "lunch meat": ["luncheon meat or poultry (pre-packaged", "turkey  (pre-packaged", "ham (pre-packaged", "salami", "chicken (pre-packaged"],
    "baking supplies decor": ["frosting", "sugar (confectioners", "pure vanilla", "gelatin ("], "juice nectars": ["juice, boxes", "orange juice (commercially packaged", "fruit juice in cartons", "nectar", "=lemon juice"],
    "canned fruit applesauce": ["applesauce (commercial", "fruit cocktail"], "ice cream toppings": ["chocolate syrup", "whipped topping"],
    "spices seasonings": ["spice/spices (ground", "herbs (dried", "seasoning blends", "=chili powder", "salt (table", "black pepper (ground", "garlic powder"],
    "doughs gelatins bake mixes": ["cake, brownie, bread mixes", "biscuit or pancake mix", "tube cans", "gelatin (", "pudding mixes"],
    "hot dogs bacon sausage": ["=hot dogs", "=bacon", "sausage (raw", "sausage (fully cooked"], "chips pretzels": ["potato chips", "=pretzels"],
    "pickled goods olives": ["=pickles", "olives (black", "=relish", "capers", "beets (pickled"], "frozen juice": ["orange juice (commercially frozen", "juice concentrates"],
    "water seltzer sparkling water": ["water (commercially bottled"], "nuts seeds dried fruit": ["nuts (jars", "almonds  (no shell", "walnuts", "cashews", "peanuts (no shell", "fruits, dried"],
    "trail mix snack mix": ["nuts (jars", "fruits, dried", "roasted nuts", "=granola"], "yogurt": ["=yogurt", "kefir"], "cereal": ["cereal (ready-to-eat"],
    "meat counter": ["beef (ground", "beef (steaks", "pork (ground", "pork (loin chops, bone-in", "lamb (chops"], "tortillas flat bread": ["tortillas (flour", "tortillas (corn"],
    "frozen appetizers sides": ["frozen potato products", "frozen pretzels", "fish (breaded"], "hot cereal pancake mixes": ["cereal (cook before eating", "biscuit or pancake mix"],
    "dry pasta": ["pasta (dry"],
}


def _days(row, prefix):
    for col in (f"{prefix}_Max", f"DOP_{prefix}_Max", f"{prefix}_Min", f"DOP_{prefix}_Min"):
        v = row.get(col)
        if pd.notna(v):
            mult = DAYS.get(row.get(col.replace("_Max", "_Metric").replace("_Min", "_Metric")))
            return float(v) * mult if mult else np.nan
    return np.nan


def build():
    sheets = {s["name"]: pd.DataFrame([{k: v for r in row for k, v in r.items()} for row in s["data"]]) for s in json.load(open(names.EXTERNAL / "foodkeeper" / "foodkeeper.json"))["sheets"]}
    cat = dict(zip(sheets["Category"].ID, sheets["Category"].Category_Name))
    it = sheets["Product"].copy()
    it["category"] = it.Category_ID.map(cat)
    it["label"] = (it.Name.astype(str) + np.where(it.Name_subtitle.notna(), " (" + it.Name_subtitle.astype(str) + ")", "")).str.lower()
    for k, pre in [("fk_pantry_days", "Pantry"), ("fk_fridge_days", "Refrigerate"), ("fk_freeze_days", "Freeze")]:
        it[k] = it.apply(_days, axis=1, prefix=pre)
    rows = {}
    for aisle, entries in AISLE_MAP.items():
        sel = np.zeros(len(it), bool)
        for e in entries:
            c, _, t = e.rpartition("|")
            m = (it.label == t[1:]) if t.startswith("=") else it.label.str.contains(t, regex=False)
            sel |= m & ((it.category == c) if c else True)
        assert sel.any(), f"no FoodKeeper item for aisle '{aisle}'"
        rows[aisle] = it.loc[sel, ["fk_pantry_days", "fk_fridge_days", "fk_freeze_days"]].median()
    a = pd.DataFrame(rows).T
    a["fk_shelf_days"] = a[["fk_pantry_days", "fk_fridge_days"]].min(axis=1).fillna(a.fk_freeze_days)
    p = names.products()
    out = p[["product_id", "aisle"]].merge(a, left_on="aisle", right_index=True, how="left").drop(columns="aisle")
    names.OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(names.OUT / "fk.parquet", index=False)
    print("fk:", len(AISLE_MAP), "aisles mapped;", int(out.fk_shelf_days.notna().sum()), "of", len(out), "products have a value")
    return out
