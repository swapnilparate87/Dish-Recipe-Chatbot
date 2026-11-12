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

from src.normalize import group_flags, has_any, matches, normalize_ingredient, singular
from src.rules import MEASURES, PANTRY, PRODUCT_FORMS, UNITS

# Words after a number that are not a unit: "2 whole eggs" is still a count of eggs
NOT_A_UNIT = {"of", "a", "an", "about", "plus", "more", "additional", "each", "whole", "x"}
# preference -> (column, ascending, label)
SORT_KEYS = {"most_used": ("n_used", False, "uses more of your ingredients"),
             "fewest_missing": ("n_missing", True, "needs fewer extra ingredients"),
             "fewest_steps": ("n_steps", True, "has fewer steps")}


def have_name(text):
    """'4 eggs' -> 'egg': the canonical name of an item the user has."""
    return (normalize_ingredient(text) or (text.lower(),))[0]


@lru_cache(maxsize=None)  # called for every (item, recipe ingredient) pair: compute once
def covers(have, name):
    """Does an ingredient you have cover this recipe ingredient? Fresh tomato does NOT give you
    tomato sauce, and chicken does not give you chicken broth."""
    if not matches(have, name, strict=True):
        return False
    return not PRODUCT_FORMS & set(name.split(have, 1)[1].split() if have in name else [])


def parse_quantity(text):
    """'2 1/2 cups rice' -> (Fraction(5, 2), 'cup'); '3 eggs' -> (3, None); no number -> None.
    Unit '?' = a size we can't compare: '1 (14.5 ounce) can tomatoes', '1 package pasta'."""
    m = re.match(r"\s*(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*(\(|[a-zA-Z]+)?", text)
    if not m:
        return None
    amount, word = sum(Fraction(p) for p in m.group(1).split()), (m.group(2) or "").lower()
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
    return (f"{'enough' if mine[0] >= need[0] else 'NOT enough'}: "
            f"have {float(mine[0]):g}{unit}, need {float(need[0]):g}{unit}")


def violations(row, exclude_ingredients, exclude_groups, max_steps):
    """Which hard constraints a recipe breaks (only used for clearly-marked alternatives)."""
    v = [f"{row['n_steps']} steps > {max_steps}"] if max_steps and row["n_steps"] > max_steps else []
    v += [f"contains {i}" for i in exclude_ingredients if has_any(row["ingredients_norm"], [i])]
    return v + [f"{'contains' if s == 'contains' else 'may contain'} {g} ({', '.join(ev)})"
                for g, s, ev in group_flags(row["ingredients_norm"], exclude_groups, row["title"])]


def match(searcher, have, exclude_ingredients=(), exclude_groups=(), max_steps=None, prefer=(), n=5):
    """have: items as the user wrote them, e.g. ['4 eggs', 'onions', '2 cups rice']."""
    items = [(text, have_name(text)) for text in have]
    names = [name for _, name in items]
    order = [SORT_KEYS[k] for k in dict.fromkeys([*prefer, *SORT_KEYS])]  # user's priorities first
    filters = dict(exclude_ingredients=exclude_ingredients, exclude_groups=exclude_groups, max_steps=max_steps)

    def score(df):
        """Add used / missing columns, keep recipes using >= 1 of your ingredients, sort."""
        used = df["ingredients_norm"].apply(lambda ns: [h for h in names if any(covers(h, x) for x in ns)])
        df = df.assign(used=used, n_used=used.str.len())
        df = df[df["n_used"] > 0]
        missing = df["ingredients_norm"].apply(
            lambda ns: [x for x in ns if x not in PANTRY and not any(covers(h, x) for h in names)])
        df = df.assign(missing=missing, n_missing=missing.str.len())
        return df.sort_values([c for c, _, _ in order] + ["title"], ascending=[a for _, a, _ in order] + [True],
                              kind="stable")

    ranked = score(searcher.apply_filters(**filters)[0])
    alternatives = []
    if ranked.empty:  # never relax silently: show the closest recipes, each marked with what it breaks
        alternatives = [dict(recipe_id=r["recipe_id"], title=r["title"], used=r["used"], missing=r["missing"],
                             violates=violations(r, *filters.values()))
                        for _, r in score(searcher.df).head(n).iterrows()]

    rows, results = [r for _, r in ranked.head(n).iterrows()], []
    for i, r in enumerate(rows):
        reason = ""
        if i + 1 < len(rows):  # why this one is above the next: the first sort key that differs
            nxt = rows[i + 1]
            col, _, label = next(((c, a, l) for c, a, l in order if r[c] != nxt[c]), (None, None, None))
            reason = (f"above #{i + 2} because it {label} ({r[col]} vs {nxt[col]})" if col
                      else f"tied with #{i + 2}; ordered by title")
        results.append(dict(
            recipe_id=r["recipe_id"], title=r["title"], n_steps=int(r["n_steps"]),
            used=r["used"], unused=[h for h in names if h not in r["used"]], missing=r["missing"],
            pantry=[x for x in r["ingredients_norm"] if x in PANTRY], reason=reason,
            quantities={name: quantity_note(text, r["ingredients"], name)
                        for text, name in items if name in r["used"] and parse_quantity(text)}))
    return {"have": names, "filters": filters, "ranking": " -> ".join(l for _, _, l in order) + " -> title",
            "eligible_count": len(ranked), "results": results, "alternatives": alternatives}


if __name__ == "__main__":
    from src.data_loader import load_recipes
    from src.search import RecipeSearch

    s = RecipeSearch(load_recipes())
    for title, kw in [("PDF example", dict(have=["eggs", "onions", "tomatoes", "rice"],
                                           exclude_ingredients=["peanut"], max_steps=6)),
                      ("With quantities", dict(have=["4 eggs", "1 cup rice", "2 tomatoes"], max_steps=6, n=3)),
                      ("No match", dict(have=["saffron", "lobster"], exclude_groups=["dairy"], max_steps=1, n=3))]:
        r = match(s, **kw)
        print(f"\n=== {title}: {r['eligible_count']} eligible | ranking: {r['ranking']}")
        for i, x in enumerate(r["results"], 1):
            print(f"{i}. {x['title']} | used {x['used']} | missing {x['missing']} | {x['quantities']} | {x['reason']}")
        for x in r["alternatives"]:
            print(f"   ALTERNATIVE (violates {x['violates']}): {x['title']} used {x['used']}")
