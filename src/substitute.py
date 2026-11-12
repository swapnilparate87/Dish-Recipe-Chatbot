"""Task 3: evidence-based ingredient substitution.

1. Find the target lines in the recipe ("2 eggs, beaten").
2. Find SIMILAR recipes: title relevance (BM25) + shared ingredients (Jaccard). Split them into
   versions WITH and WITHOUT the target ingredient.
3. For each candidate substitute (from SUBSTITUTES in rules.py = general cooking knowledge):
   direct evidence   = similar recipes WITHOUT the target that use the candidate (cited by ID)
   indirect evidence = other recipes with a similar ingredient profile, without the target, using it
   conflicting       = the candidate is as common in versions that KEEP the target
4. Check restrictions, pick the best supported candidate, let the LLM rewrite the recipe,
   then CHECK the rewrite (no new ingredients, target gone, restrictions kept). If the check
   fails, fall back to a mechanical edit and say so.
"""

import re
from collections import defaultdict
from functools import lru_cache

from src.llm import rewrite_recipe
from src.normalize import group_flags, has_any, has_phrase, matches, normalize_ingredient, normalize_list
from src.rules import EXTRAS, GROUPS, NOT_A, PANTRY, SUBSTITUTES, UNITS, VERB_USES
from src.search import tokens

STRENGTH_TEXT = {  # in ranking order
    "strong": "direct evidence: several similar recipes without it use this",
    "moderate": "direct evidence, but only 1-2 similar recipes",
    "indirect": "indirect: used in other recipes with a similar ingredient profile",
    "weak": "weak: only 1-4 loosely related recipes",
    "conflicting": "conflicting: about as common in versions that KEEP the ingredient",
    "none": "no evidence in the dataset (general cooking knowledge only)",
}
STRENGTH_ORDER = list(STRENGTH_TEXT)


def similar_recipes(searcher, recipe, pool_size=1500, min_jaccard=0.2):
    """Same/related dish: top titles by BM25, kept only if they share enough ingredients."""
    base = set(recipe["ingredients_norm"]) - PANTRY
    scores = searcher.bm25.get_scores(tokens(recipe["title"]))
    top = scores.argsort()[::-1][:pool_size]
    pool = searcher.df.iloc[top[scores[top] > 0]]  # score 0 = no title word in common: not the same dish
    jaccard = pool["ingredients_norm"].apply(lambda ns: len(base & set(ns)) / (len(base | set(ns) - PANTRY) or 1))
    return pool[(jaccard >= min_jaccard) & (pool["recipe_id"] != recipe["recipe_id"])]


@lru_cache(maxsize=4)  # one entry per searcher (= per dataset), so a new dataset gets a new index
def word_index(searcher):
    """word -> every dataset ingredient name containing it ('oil' -> {'olive oil', ...}). Built once."""
    index = defaultdict(set)
    for names in searcher.df["ingredients_norm"]:
        for n in names:
            for w in n.split():
                index[w].add(n)
    return index


def in_dataset(searcher, name):
    """Is `name` (or a name containing it) used in any recipe? Only names sharing its last word are checked."""
    return any(matches(name, n) for n in word_index(searcher).get(name.split()[-1], ()))


def substitutes_key(t):
    """Which SUBSTITUTES entry covers ingredient t: 'heavy cream' -> 'cream'. A different product
    doesn't count: 'rice flour' is itself a flour OPTION (and gluten-free), so it gets no 'flour' entry."""
    if t in SUBSTITUTES:
        return t
    exceptions = lambda k: [x for g in GROUPS.values() for x in g.get("exceptions", {}).get(k, [])]
    return next((k for k in SUBSTITUTES if matches(k, t) and not has_any([t], SUBSTITUTES[k]["options"])
                 and not any(has_phrase(x, t) for x in exceptions(k))), None)


def word_pattern(t, others=(), modifiers=()):
    """t as a whole word in text (plural too). The named group 'keep' catches look-alikes that are NOT t:
    a different product ('peanut butter', 'cream cheese'), a verb ('cream the sugar') or one of the
    substitute options (`others`: 'rice flour' is not the flour being replaced).
    `modifiers` = words in front of t that belong to it ('heavy' cream): matched (and replaced) too."""
    keep = [re.escape(x) for x in [*NOT_A.get(t, []), *others]] + ([VERB_USES[t]] if t in VERB_USES else [])
    alts = [rf"(?P<keep>\b(?:{'|'.join(keep)})\w*)"] if keep else []
    front = rf"(?:\b(?:{'|'.join(map(re.escape, modifiers))})\s+)*" if modifiers else ""
    return re.compile("|".join(alts + [rf"{front}\b{re.escape(t)}(?:e?s)?\b"]), re.I)


def mentions(t, text, others=()):
    """Does the text really mention ingredient t? ('spread peanut butter' doesn't mention butter)"""
    return any(not m.groupdict().get("keep") for m in word_pattern(t, others).finditer(text))


def replace_word(t, text, new, others=(), modifiers=()):
    """Swap ingredient t (with its modifiers) for `new` in text, leaving look-alikes untouched."""
    return word_pattern(t, others, modifiers).sub(lambda m: m.group() if m.groupdict().get("keep") else new, text)


def is_target(name, replace, options):
    """Is this ingredient name one we are replacing? 'rice flour' is NOT a target of 'flour' when
    rice flour is one of the substitute options (it's the replacement, not the problem)."""
    return has_any([name], replace) and not has_any([name], options)


def modifier_words(target_lines, replace):
    """Words of the target lines that may stand in front of the target in instructions: 'heavy',
    'whipping' from '1 cup heavy whipping cream'. Units, numbers and small words are left out."""
    small = {"the", "and", "or", "of", "to", "in", "a", "an", "for", "with"}
    words = {w for l in target_lines for w in re.findall(r"[a-z]+", re.sub(r"\(.*?\)", " ", l.lower()))}
    return sorted(words - UNITS - small - {w for t in replace for w in t.split()}, key=len, reverse=True)


def example_line(row, candidate):
    """The candidate's ingredient line in an evidence recipe: shows a real amount ('1/2 cup applesauce')."""
    return next((l for l in row["ingredients"] if has_any(normalize_ingredient(l), [candidate])), None)


def candidate_hits(df, candidates):
    """For every recipe, the set of candidates it uses: ONE pass over the rows for all candidates."""
    return [{c for c in candidates if has_any(ns, [c])} for ns in df["ingredients_norm"]]


def gather_evidence(searcher, recipe, replace, candidates, cached=None):
    """cached = the evidence from the previous answer for the SAME recipe and target ("use yogurt
    instead"): reused when it covers every candidate, instead of scanning the dataset again."""
    if cached and set(candidates) <= {c["substitute"] for c in cached["candidates"]}:
        return {**cached, "candidates": [dict(c) for c in cached["candidates"] if c["substitute"] in candidates]}
    similar = similar_recipes(searcher, recipe)
    keeps = similar["ingredients_norm"].apply(lambda ns: has_any(ns, replace))
    with_t, without_t = similar[keeps], similar[~keeps]
    base = set(recipe["ingredients_norm"]) - PANTRY
    others = searcher.df[~searcher.df.index.isin(similar.index)]  # indirect: anywhere else, similar
    others = others[others["ingredients_norm"].apply(lambda ns: len(base & set(ns)) >= 3 and not has_any(ns, replace))]
    frames = [(d, candidate_hits(d, candidates)) for d in (without_t, others, with_t)]

    rows = []
    for c in candidates:
        direct, indirect, kept = (d[[c in h for h in hits]] for d, hits in frames)
        rate_without, rate_with = len(direct) / max(len(without_t), 1), len(kept) / max(len(with_t), 1)
        if direct.empty:
            strength = "indirect" if len(indirect) >= 5 else "weak" if len(indirect) else "none"
        elif rate_without < 1.5 * rate_with or rate_without - rate_with < 0.03:
            strength = "conflicting"  # not clearly MORE common without the target (banana in banana bread)
        else:
            strength = "strong" if len(direct) >= 3 else "moderate"
        cited = direct if not direct.empty else indirect
        rows.append(dict(
            substitute=c, strength=strength, direct=len(direct), indirect=len(indirect),
            pct_without=round(100 * rate_without), pct_with=round(100 * rate_with),
            ids=[(r["recipe_id"], r["title"]) for _, r in cited.head(3).iterrows()],
            example=next((line for _, r in cited.head(5).iterrows() if (line := example_line(r, c))), None)))
    rows.sort(key=lambda r: (STRENGTH_ORDER.index(r["strength"]), -r["direct"], -r["indirect"]))
    return {"similar": len(similar), "similar_with": len(with_t), "similar_without": len(without_t),
            "candidates": rows}


def restriction_problems(candidate, replace, exclude_groups, exclude_ingredients, listed=()):
    """Why a candidate can't be used ([] = allowed). 'May contain' counts as a problem.
    Extras its conversion needs (milk -> + butter) must pass the restrictions too.
    `listed` = options from SUBSTITUTES: 'rice flour' is a listed option for flour, so not 'itself flour'."""
    extras = EXTRAS.get(candidate, [])
    names = [candidate, *extras]
    return ([f"is itself {t}" for t in replace
             if (matches(t, candidate) and candidate not in listed) or matches(candidate, t)]
            + [f"its conversion needs {x}, which you're replacing" for x in extras for t in replace if matches(t, x)]
            + [f"excluded ({e})" for e in exclude_ingredients for n in names if matches(e, n)]
            + [f"{'contains' if s == 'contains' else 'may contain'} {g}"
               for g, s, _ in group_flags(names, exclude_groups)])


def check_rewrite(new, recipe, targets, replace, substitute, exclude_groups, options=()):
    """Problems with the LLM's rewrite ([] = accepted). Grounding guard: no invented ingredients and
    none of the other ingredients dropped. An added name must be the substitute (or a shorter form:
    'oil' for 'vegetable oil') or an extra its conversion names. 'sesame oil' is NOT accepted for 'vegetable oil'."""
    names = normalize_list(new.ingredients)
    is_sub = lambda n: matches(substitute, n) or matches(n, substitute)
    is_extra = lambda n: any(matches(x, n) for x in EXTRAS.get(substitute, []))
    allowed = set(recipe["ingredients_norm"]) | PANTRY
    added = [n for n in names if n not in allowed]  # only what the rewrite ADDED is judged
    kept = normalize_list([l for l in recipe["ingredients"] if l not in targets])  # must all stay
    already = has_any(recipe["ingredients_norm"], [substitute])  # banana bread: banana is already there
    new_sub_line = any(l not in recipe["ingredients"] and has_any(normalize_ingredient(l), [substitute])
                       for l in new.ingredients)
    return ([f"still contains {t}" for t in replace if any(is_target(n, [t], options) for n in names)]
            + [f"dropped '{n}'" for n in kept if n not in names and n not in PANTRY]
            + ([f"no added amount of {substitute} (it was already in the recipe)"]
               if already and not new_sub_line else [])
            + [f"added '{n}'" for n in added if not (is_sub(n) or is_extra(n))]
            + [f"adds {g}" for g, s, _ in group_flags(added, exclude_groups) if s == "contains"]
            + ([] if any(is_sub(n) for n in names) else [f"{substitute} missing from the ingredient list"])
            + [f"instructions still mention {t}" for t in replace if mentions(t, new.instructions, options)])


def mechanical_edit(recipe, targets, replace, substitute, amount, options=()):
    """Fallback when the LLM rewrite is rejected: swap lines and words, nothing else.
    'stir in the heavy cream' -> 'stir in the coconut milk' (the modifier goes too)."""
    ingredients = [f"{substitute} ({amount}; replaces: {l})" if l in targets else l for l in recipe["ingredients"]]
    instructions, modifiers = recipe["instructions"], modifier_words(targets, replace)
    for t in replace:
        instructions = replace_word(t, instructions, substitute, options, modifiers)
    return ingredients, instructions


def clean_use(use, options):
    """The user's 'use X' as a canonical name: 'Bananas' -> 'banana'; 'Greek yogurt' -> the 'yogurt'
    option (keeps its tuned amount). Not an option at all -> the cleaned name."""
    name = (normalize_ingredient(use) or (use.lower().strip(),))[0]
    return name if name in options else next((o for o in options if matches(o, name)), name)


def substitute(searcher, recipe, replace, exclude_groups=(), exclude_ingredients=(), use=None, cached=None):
    key = next((k for t in replace if (k := substitutes_key(t))), None)
    role, options = (SUBSTITUTES[key]["role"], dict(SUBSTITUTES[key]["options"])) if key else ("unknown", {})
    listed = tuple(options)  # options from cooking knowledge (before the user's own 'use X')
    use = clean_use(use, options) if use else None
    if use and use not in options:
        options[use] = "amount: start with the same amount (rule of thumb)"
    result = {"recipe": recipe, "replace": replace, "targets": [
        l for l in recipe["ingredients"] if any(is_target(n, replace, options) for n in normalize_ingredient(l))]}
    if not result["targets"]:
        result.pop("targets")
        return {**result, "error": f"This recipe has no {', '.join(replace)} in its ingredient list, "
                                   "so there is nothing to replace."}
    if not options:
        return {**result, "error": f"I have no candidate substitutes for {', '.join(replace)}. "
                                   "Name one (\"use X instead\") and I'll check the dataset for evidence."}

    ev = gather_evidence(searcher, recipe, replace, list(options), cached)
    for c in ev["candidates"]:
        c["amount"] = options[c["substitute"]]
        c["problems"] = restriction_problems(c["substitute"], replace, exclude_groups, exclude_ingredients, listed)
        if not in_dataset(searcher, c["substitute"]):
            c["problems"].append("not found in the dataset")  # PDF: use only dataset ingredients
        if has_any(recipe["ingredients_norm"], [c["substitute"]]):
            c["already_in_recipe"] = True  # evidence can't show it replaces the target
    result.update(role=role, evidence=ev, restrictions=list(exclude_groups) + [f"no {e}" for e in exclude_ingredients])
    # already in the recipe -> tried last: evidence can't show it REPLACES the target (banana in banana bread)
    allowed = sorted((c for c in ev["candidates"] if not c["problems"]),
                     key=lambda c: bool(c.get("already_in_recipe")))
    chosen = next((c for c in allowed if c["substitute"] == use), None) if use else next(iter(allowed), None)
    if chosen is None:
        return {**result, "error": f"No candidate for {', '.join(replace)} passes your restrictions"
                                   + (f" ({use} doesn't)" if use else "") + ", so I can't support a reliable adaptation."}

    sub, amount, problems = chosen["substitute"], chosen["amount"], None
    for _ in range(2):  # one retry, telling the model what was wrong
        try:
            new = rewrite_recipe(recipe, result["targets"], sub, amount, role, feedback=problems)
            problems = check_rewrite(new, recipe, result["targets"], replace, sub, exclude_groups, tuple(options))
        except Exception as e:
            problems = [f"language model unavailable ({type(e).__name__})"]
            break
        if not problems:
            break
    if problems:
        ingredients, instructions = mechanical_edit(recipe, result["targets"], replace, sub, amount, tuple(options))
        effects, note = [], (f"LLM rewrite rejected ({'; '.join(problems)}), so lines and words were swapped "
                             "mechanically. Review the instructions.")
    else:
        ingredients, instructions, effects, note = new.ingredients, new.instructions, new.effects, None
    still = {g: e for g, _, e in group_flags(normalize_list(ingredients), exclude_groups)}  # e.g. Parmesan
    return {**result, "chosen": chosen, "weak_evidence": chosen["strength"] in ("weak", "conflicting", "none"),
            "still_contains": still, "ingredients": ingredients, "instructions": instructions,
            "effects": effects, "note": note}


if __name__ == "__main__":
    from src.data_loader import load_recipes
    from src.search import RecipeSearch

    s = RecipeSearch(load_recipes())
    r = substitute(s, s.df[s.df["title"] == "Banana Bread"].iloc[0].to_dict(), ["egg"])
    print(r["recipe"]["title"], r["targets"], "| role:", r["role"])
    for c in r["evidence"]["candidates"]:
        print(f"  {c['substitute']:14} {c['strength']:11} direct {c['direct']:3} ({c['pct_without']}% vs "
              f"{c['pct_with']}%) indirect {c['indirect']:4} {c['problems'] or ''} e.g. {c['example']}")
    print("CHOSEN:", r["chosen"]["substitute"], "| note:", r["note"])
    print("\n".join(r["ingredients"]), "\n", r["instructions"][:400], "\n", r["effects"])
