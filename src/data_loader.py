"""Load the three Recipe Box JSON files into one DataFrame."""

import json
import re
from pathlib import Path
import pandas as pd

from src.normalize import normalize_list, count_steps

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "recipes_raw"
SOURCES = ["ar", "epi", "fn"]  # recipes_raw_nosource_<source>.json


def load_recipes():
    rows = []
    for source in SOURCES:
        with open(RAW_DIR / f"recipes_raw_nosource_{source}.json", encoding="utf-8") as f:
            data = json.load(f)
        for key, rec in data.items():
            title = (rec.get("title") or "").strip()
            ingredients = rec.get("ingredients") or []
            clean = [i.replace("ADVERTISEMENT", "").strip() for i in ingredients]
            clean = [i for i in clean if i]
            instructions = (rec.get("instructions") or "").strip()
            has_text = any(c.isalpha() for c in instructions)  # ";" is not an instruction
            is_test = re.match(r"(?i)qa(test|v?\s*\d|\s|$)", title)  # 5 website test pages ("QATest2")
            if title and clean and has_text and not is_test:  # skip incomplete / test records
                rows.append({
                    "recipe_id": f"{source}:{key}",
                    "title": title,
                    "ingredients_raw": ingredients,   # original, untouched
                    "ingredients": clean,             # without "ADVERTISEMENT"
                    "instructions": instructions,
                })
    df = pd.DataFrame(rows)
    df["ingredients_norm"] = df["ingredients"].apply(normalize_list)  # canonical names
    df["n_steps"] = df["instructions"].apply(count_steps)                # sentence count
    return df


if __name__ == "__main__":
    df = load_recipes()
    print("Recipes:", len(df), "| unique IDs:", df["recipe_id"].is_unique)
    print(df.sample(3, random_state=0)[["recipe_id", "title"]])
