"""Task 2: rank EXISTING recipes by the ingredients the user has (never invents recipes).

For each recipe:  used    = your ingredients the recipe uses
                  unused  = your ingredients it doesn't use
                  missing = what the recipe needs that you don't have (pantry staples aside)
Ranking: sort keys in the user's preference order, default
         most of your ingredients used -> fewest missing -> fewest steps -> title A-Z
"""

import re
from fractions import Fraction
from functools import lru_cache

from src.normalize import check_group, matches, name_status, normalize_ingredient, singular
from src.rules import MEASURES, PANTRY, PRODUCT_FORMS, UNITS

# Words after a number that are not a unit: "2 whole eggs" is still a count of eggs
NOT_A_UNIT = {"of", "a", "an", "about", "plus", "more", "additional", "each", "whole", "x"}

SORT_KEYS = {"most_used": ("n_used", False), "fewest_missing": ("n_missing", True),
             "fewest_steps": ("n_steps", True)}
DEFAULT_ORDER = ["most_used", "fewest_missing", "fewest_steps"]
LABELS = {"most_used": "uses more of your ingredients", "fewest_missing": "needs fewer extra ingredients",
          "fewest_steps": "has fewer steps"}


def have_name(text):
    """'4 eggs' -> 'egg': the canonical name of an item the user has."""
    return (normalize_ingredient(text) or (text.lower(),))[0]


@lru_cache(maxsize=None)  # called for every (item, recipe ingredient) pair: compute once
def covers(have, name):
    """Does an ingredient you have cover this recipe ingredient? Fresh tomato does NOT give you
    tomato sauce, and chicken does not give you chicken broth."""
    if not matches(have, name, strict=True):
        return False
    after = name.split(have, 1)[1].split() if have in name else []
    return not PRODUCT_FORMS & set(after)


def parse_quantity(text):
    """'2 1/2 cups rice' -> (Fraction(5, 2), 'cup'); '3 eggs' -> (3, None); no number -> None.
    Unit '?' = a size we can't compare: '1 (14.5 ounce) can tomatoes', '1 package pasta'."""
    m = re.match(r"\s*(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*(\(|[a-zA-Z]+)?", text)
    if not m:
        return None
    amount = sum(Fraction(p) for p in m.group(1).split())
    word = (m.group(2) or "").lower()
    if word == "(":
        return amount, "?"
    unit = MEASURES.get(word) or MEASURES.get(singular(word))
    if unit is None and singular(word) in UNITS and singular(word) not in NOT_A_UNIT:
        unit = "?"
    return amount, unit


def quantity_note(have_text, recipe_lines, have):
    """Compare quantities only when both are given in the same unit; otherwise say so."""
    mine = parse_quantity(have_text)
    if mine is None:
        return "quantity not given"
    line = next((l for l in recipe_lines if any(covers(have, n) for n in normalize_ingredient(l))), None)
    need = parse_quantity(line) if line else None
    if need is None or need[1] != mine[1] or mine[1] == "?":
        return f"quantity not verified (recipe: '{line}')"
    unit = f" {mine[1]}" if mine[1] else ""
    verdict = "enough" if mine[0] >= need[0] else "NOT enough"
    return f"{verdict}: have {float(mine[0]):g}{unit}, need {float(need[0]):g}{unit}"


def violations(row, exclude_ingredients, exclude_groups, max_steps):
    """Which hard constraints a recipe breaks (only used for clearly-marked alternatives)."""
    v = [f"{row['n_steps']} steps > {max_steps}"] if max_steps and row["n_steps"] > max_steps else []
    v += [f"contains {i}" for i in exclude_ingredients if any(matches(i, n) for n in row["ingredients_norm"])]
    for g in exclude_groups:
        status, evidence = check_group(row["ingredients_norm"], g)
        if status != "ok":
            v.append(f"{'contains' if status == 'contains' else 'may contain'} {g} ({', '.join(evidence)})")
        elif name_status(row["title"].lower(), g) == "contains":  # same check as search.py
            v.append(f"may contain {g} (title mentions it, ingredient list doesn't)")
    return v


def match(searcher, have, exclude_ingredients=(), exclude_groups=(), max_steps=None, prefer=(), n=5):
    """have: items as the user wrote them, e.g. ['4 eggs', 'onions', '2 cups rice']."""
    items = [(text, have_name(text)) for text in have]
    names = [name for _, name in items]
    order = list(dict.fromkeys([*prefer, *DEFAULT_ORDER]))  # user's priorities first
    filters = dict(exclude_ingredients=exclude_ingredients, exclude_groups=exclude_groups, max_steps=max_steps)

    def score(df):
        """Add used / unused / missing / pantry columns, keep recipes using >= 1 of your ingredients."""
        used = df["ingredients_norm"].apply(lambda ns: [h for h in names if any(covers(h, x) for x in ns)])
        df = df.assign(used=used, n_used=used.str.len())
        df = df[df["n_used"] > 0]
        missing = df["ingredients_norm"].apply(
            lambda ns: [x for x in ns if x not in PANTRY and not any(covers(h, x) for h in names)])
        df = df.assign(missing=missing, n_missing=missing.str.len())
        columns = [SORT_KEYS[k][0] for k in order] + ["title"]
        return df.sort_values(columns, ascending=[SORT_KEYS[k][1] for k in order] + [True], kind="stable")

    eligible, _ = searcher.apply_filters(**filters)
    ranked = score(eligible)
    alternatives = []
    if ranked.empty:  # never relax silently: show the closest recipes, each marked with what it breaks
        closest = score(searcher.df).head(n)
        alternatives = [dict(recipe_id=r["recipe_id"], title=r["title"], used=r["used"],
                             missing=r["missing"], violates=violations(r, *filters.values()))
                        for _, r in closest.iterrows()]

    results, rows = [], list(ranked.head(n).iterrows())
    for i, (_, r) in enumerate(rows):
        reason = ""
        if i + 1 < len(rows):  # why this one is above the next: the first sort key that differs
            nxt = rows[i + 1][1]
            key = next((k for k in order if r[SORT_KEYS[k][0]] != nxt[SORT_KEYS[k][0]]), None)
            reason = (f"above #{i + 2} because it {LABELS[key]} ({r[SORT_KEYS[key][0]]} vs "
                      f"{nxt[SORT_KEYS[key][0]]})" if key else f"tied with #{i + 2}; ordered by title")
        results.append(dict(
            recipe_id=r["recipe_id"], title=r["title"], n_steps=int(r["n_steps"]),
            used=r["used"], unused=[h for h in names if h not in r["used"]], missing=r["missing"],
            pantry=[x for x in r["ingredients_norm"] if x in PANTRY], reason=reason,
            quantities={name: quantity_note(text, r["ingredients"], name)
                        for text, name in items if name in r["used"] and parse_quantity(text)}))

    return {"have": names, "filters": filters, "ranking": " -> ".join(LABELS[k] for k in order) + " -> title",
            "eligible_count": len(ranked), "results": results, "alternatives": alternatives}


if __name__ == "__main__":
    from src.data_loader import load_recipes
    from src.search import RecipeSearch

    s = RecipeSearch(load_recipes())

    def show(title, **kw):
        r = match(s, **kw)
        print(f"\n=== {title}\n{r['eligible_count']} eligible | ranking: {r['ranking']}")
        for i, x in enumerate(r["results"], 1):
            print(f"{i}. {x['title']} [{x['recipe_id']}] {x['n_steps']} steps")
            print(f"   used {x['used']} | unused {x['unused']} | missing {x['missing']} | pantry {x['pantry']}")
            if x["quantities"]:
                print(f"   quantities: {x['quantities']}")
            print(f"   {x['reason']}")
        for x in r["alternatives"]:
            print(f"   ALTERNATIVE (violates {x['violates']}): {x['title']} used {x['used']}")

    show("PDF example", have=["eggs", "onions", "tomatoes", "rice"], exclude_ingredients=["peanut"], max_steps=6)
    show("Follow-up: exclude eggs too", have=["onions", "tomatoes", "rice"],
         exclude_ingredients=["peanut", "egg"], max_steps=6)
    show("With quantities", have=["4 eggs", "1 cup rice", "2 tomatoes"], max_steps=6, n=3)
    show("No match", have=["saffron", "lobster"], exclude_groups=["dairy"], max_steps=1, n=3)
