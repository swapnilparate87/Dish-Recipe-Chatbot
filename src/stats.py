"""Dataset statistics, computed in pandas on the COMPLETE filtered set (never by the LLM).
Every function returns the answer plus how it was computed: filters, recipes analyzed, method."""

from collections import Counter

from src.matcher import covers, have_name
from src.normalize import matches


def _base(searcher, filters):
    """Shared first step: filter all recipes, describe what was analyzed.
    query / have narrow the set to the current search ("how many of THESE"): a dish name keeps
    recipes whose title matches it, a have-list keeps recipes using at least one of the items."""
    query, have = filters.pop("query", None), filters.pop("have", None)
    eligible, uncertain = searcher.apply_filters(**filters)
    if query:
        eligible, uncertain = searcher.rank(eligible, query), searcher.rank(uncertain, query)
    if have:
        names = [have_name(h) for h in have]
        eligible = eligible[eligible["ingredients_norm"].apply(
            lambda ns: any(covers(h, x) for h in names for x in ns))]
    filters.update({k: v for k, v in (("query", query), ("have", have)) if v})  # for describe()
    info = {"filters": filters, "recipes_analyzed": len(eligible),
            "uncertain_excluded": len(uncertain), "dataset_size": len(searcher.df)}
    return eligible, info


def count_recipes(searcher, **filters):
    """'How many recipes contain garlic and onion? What % of the dataset?'"""
    eligible, info = _base(searcher, filters)
    all_names = [n for names in eligible["ingredients_norm"] for n in names]
    variants = {ing: [n for n, _ in Counter(n for n in all_names if matches(ing, n, True)).most_common(5)]
                for ing in filters.get("include", ())}  # evidence: which names counted as "garlic"
    return {**info, "count": len(eligible), "matched_names": variants,
            "percent": round(100 * len(eligible) / len(searcher.df), 2),
            "calculation": "matching recipes / all recipes in the dataset x 100"}


def common_ingredients(searcher, top=10, **filters):
    """'What are the most common ingredients (in dairy-free recipes)?'"""
    eligible, info = _base(searcher, filters)
    counts = Counter(n for names in eligible["ingredients_norm"] for n in names)
    n = len(eligible) or 1
    return {**info, "top": [(name, c, round(100 * c / n, 1)) for name, c in counts.most_common(top)],
            "calculation": "number of recipes containing each normalized ingredient "
                           "(counted once per recipe); % of recipes analyzed"}


def ingredient_count_stats(searcher, **filters):
    """'What is the average / median number of ingredients?'"""
    eligible, info = _base(searcher, filters)
    sizes = eligible["ingredients"].str.len()
    if sizes.empty:  # no recipe matches: there is no average to report
        return {**info, "mean": None, "median": None, "min": None, "max": None,
                "calculation": "no recipes matched the filters"}
    return {**info, "mean": round(float(sizes.mean()), 2), "median": float(sizes.median()),
            "min": int(sizes.min()), "max": int(sizes.max()),
            "calculation": "ingredient lines per recipe (original list, ADVERTISEMENT removed)"}


def rank_by_steps(searcher, n=5, fewest=True, **filters):
    """'Among recipes containing tomato, which five have the fewest instruction steps?'"""
    eligible, info = _base(searcher, filters)
    ranked = eligible.sort_values(["n_steps", "title"], ascending=[fewest, True], kind="stable")
    top = ranked.head(n)
    cutoff = top["n_steps"].iloc[-1] if len(top) else None
    return {**info,
            "results": [(r.recipe_id, r.title, int(r.n_steps)) for r in top.itertuples()],
            "ties_at_cutoff": int((eligible["n_steps"] == cutoff).sum()) if cutoff is not None else 0,
            "calculation": "steps = sentences in the instructions (see README); "
                           f"sorted by steps ({'fewest' if fewest else 'most'} first), ties by title A-Z"}


if __name__ == "__main__":
    from src.data_loader import load_recipes
    from src.search import RecipeSearch

    s = RecipeSearch(load_recipes())
    print(count_recipes(s, include=["garlic", "onion"]), "\n")
    print(common_ingredients(s, top=5, exclude_groups=["dairy"]), "\n")
    print(ingredient_count_stats(s), "\n")
    print(rank_by_steps(s, n=5, include=["tomato"]))
