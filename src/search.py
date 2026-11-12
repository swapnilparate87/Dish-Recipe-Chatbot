"""Recipe search in two stages.

1. FILTER all recipes with hard rules (no LLM, no top-k), so stats can use the full set.
2. RANK what is left: dish name -> exact title, then BM25 on titles;
                      ingredients only -> every recipe with them, simplest first.
"""

import re

import pandas as pd
from rank_bm25 import BM25Okapi

from src.normalize import group_flags, has_any, matches, singular


def tokens(text):
    return [singular(w) for w in re.findall(r"[a-z]+", text.lower())]


def restriction_status(names, title, groups):
    """('contains' | 'uncertain' | 'ok', flags) for one recipe. Uncertain = MIGHT break a restriction."""
    flags = group_flags(names, groups, title)
    if any(status == "contains" for _, status, _ in flags):
        return "contains", []
    return ("uncertain" if flags else "ok"), [f"{g}? {', '.join(ev)}" for g, _, ev in flags]


class RecipeSearch:
    def __init__(self, df):
        self.df = df
        self.bm25 = BM25Okapi(df["title"].apply(tokens).tolist())  # built once

    def apply_filters(self, include=(), exclude_ingredients=(), exclude_groups=(), max_steps=None):
        """Stage 1 -> (eligible, uncertain). Uncertain recipes (pesto for dairy) are never counted as safe."""
        df, names = self.df, self.df["ingredients_norm"]
        keep = pd.Series(True, index=df.index)
        for ing in include:              # strict: "chicken broth" is not chicken
            keep &= names.apply(lambda ns: has_any(ns, [ing], strict=True))
        for ing in exclude_ingredients:  # broad: "no chicken" also removes chicken broth
            keep &= ~names.apply(lambda ns: has_any(ns, [ing]))
        if max_steps is not None:
            keep &= df["n_steps"] <= max_steps
        df = df[keep]
        checks = [restriction_status(ns, t, exclude_groups) for ns, t in zip(df["ingredients_norm"], df["title"])]
        df = df.assign(status=[c[0] for c in checks], flags=[c[1] for c in checks])
        return df[df["status"] == "ok"], df[df["status"] == "uncertain"]

    def rank(self, df, query):
        """Stage 2. Dish name: exact title first, then BM25. Otherwise: simplest first.
        Stable sorts: ties keep dataset order, so results are reproducible."""
        if not query:
            df = df.assign(n_ing=df["ingredients"].str.len())
            return df.sort_values(["n_ing", "n_steps", "title"], kind="stable")
        score = self.bm25.get_scores(tokens(query))[df.index]  # index = row position
        score = score + 1000 * (df["title"].str.lower() == query.lower().strip())
        return df.assign(score=score)[score > 0].sort_values("score", ascending=False, kind="stable")

    def search(self, query=None, top_n=5, **filters):
        eligible, uncertain = (self.rank(d, query) for d in self.apply_filters(**filters))
        include = filters.get("include", ())

        def evidence(row):  # what we show per recipe: ID, title and WHY it matched
            return {"recipe_id": row["recipe_id"], "title": row["title"], "n_steps": row["n_steps"],
                    "matched": {i: [n for n in row["ingredients_norm"] if matches(i, n, True)] for i in include},
                    "flags": row["flags"], "ingredients": row["ingredients"]}

        return {"filters": {"query": query, **filters}, "total_matches": len(eligible),
                "ranking": "exact title, then BM25 title relevance" if query
                           else "fewest ingredients, then fewest steps",
                "results": [evidence(r) for _, r in eligible.head(top_n).iterrows()],
                "uncertain_count": len(uncertain),
                "uncertain": [evidence(r) for _, r in uncertain.head(3).iterrows()]}

    def get(self, recipe_id):
        """Full original recipe by ID."""
        row = self.df[self.df["recipe_id"] == recipe_id]
        return None if row.empty else row.iloc[0].to_dict()


if __name__ == "__main__":
    from src.data_loader import load_recipes

    s = RecipeSearch(load_recipes())
    for kwargs in [dict(query="penne pasta"), dict(query="penne pasta", exclude_groups=["dairy"]),
                   dict(include=["chicken", "rice"]), dict(query="penne pasta", include=["chocolate"], max_steps=2)]:
        r = s.search(**kwargs)
        print(f"\n{kwargs}: {r['total_matches']} matches (+{r['uncertain_count']} uncertain), by {r['ranking']}")
        for x in r["results"]:
            print(f"  {x['recipe_id']} | {x['title']} | {x['n_steps']} steps | {x['matched']}")
