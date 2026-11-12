"""Conversation state: remembers the active filters between turns, runs search / matching /
stats / substitution, and writes the answer from templates (no LLM, so every fact comes from data).
Each answer also sets self.view = the structured result, which the web UI draws as cards/charts."""

from src import stats
from src.llm import parse
from src.matcher import have_name, match
from src.normalize import group_flags, matches, normalize_list
from src.search import tokens
from src.substitute import STRENGTH_TEXT, substitute, substitutes_key

LIST_KEYS = ("include", "exclude_ingredients", "exclude_groups")
NORMALIZATION_NOTE = ("Ingredient names are normalized (quantities, units and preparation words removed, "
                      "plurals made singular). An ingredient matches names that contain it as a whole word "
                      "(garlic -> garlic powder), except different products (butter is not peanut butter, "
                      "chicken is not chicken broth).")
HELP = ("I can search recipes (\"find a penne pasta recipe\"), apply restrictions (\"no dairy\"), "
        "answer statistics (\"how many recipes contain garlic?\") and show a recipe (\"show me the first one\").")
STATS = {"common_ingredients": stats.common_ingredients, "ingredient_count": stats.ingredient_count_stats,
         "rank_steps": stats.rank_by_steps, "count": stats.count_recipes}


def new_filters():
    return {"query": None, "include": [], "exclude_ingredients": [], "exclude_groups": [], "max_steps": None}


def describe(f):
    """Filters as readable text: 'dish "penne pasta"; no dairy; at most 5 steps'."""
    parts = ([f'dish "{f["query"]}"'] if f.get("query") else []) \
        + ([f"uses any of: {', '.join(f['have'])}"] if f.get("have") else []) \
        + [f"with {i}" for i in f.get("include", [])] + [f"no {g}" for g in f.get("exclude_groups", [])] \
        + [f"without {i}" for i in f.get("exclude_ingredients", [])] \
        + ([f"at most {f['max_steps']} steps"] if f.get("max_steps") is not None else [])
    return "; ".join(parts) or "none"


def numbered(items, line):
    """'1. ...', '2. ...' lines built by `line(item)`."""
    return [f"{i}. {line(x)}" for i, x in enumerate(items, 1)]


class Session:
    _attrs = None  # attribute names a current Session has (computed once per code version)

    @classmethod
    def is_current(cls, obj):
        """False if obj was made by OLDER code (the app was edited while running): it would lack new
        attributes and crash. The class is rebuilt when the code changes, so _attrs is recomputed then."""
        if cls._attrs is None:
            cls._attrs = set(vars(cls(None)))  # cheap: __init__ only sets attributes
        return isinstance(obj, cls) and cls._attrs <= set(vars(obj))

    def __init__(self, searcher, debug=False):
        self.searcher, self.debug = searcher, debug
        self.filters = new_filters()
        self.last_results = []      # recipe IDs shown last, for "show me the second one"
        self.mode = "search"        # "search" (Task 1) or "match" (Task 2): what follow-ups rerun
        self.have, self.prefer = [], []  # Task 2: ingredients the user has, ranking priorities
        self.view = None            # structured result of the last answer, for the web UI
        self.forget_recipe()

    def forget_recipe(self):
        """A new result list (or "start over") means the old recipe is no longer the one discussed."""
        self.current_recipe = None  # recipe being discussed (shown or adapted): target of "remove eggs"
        self.adapted = None         # its adapted version, so a 2nd substitution builds on the 1st
        self.last_sub = None        # last substitution (recipe, starting version, what), for "use yogurt instead"

    def handle(self, message):
        self.last_intent = None
        try:
            context = describe(self.filters)
            if self.current_recipe:  # lets "this recipe" refer to the right one
                context += f"; current recipe: {self.searcher.get(self.current_recipe)['title']}"
            intent = parse(message, context)
        except Exception as e:
            return f"Sorry, the language model could not be reached or understood this ({type(e).__name__})."
        self.last_intent = intent  # shown in the web UI as "What I understood"
        if self.debug:
            print(f"[debug] {intent}")
        return self.respond(intent)

    def respond(self, intent):
        """Intent -> answer. Separate from handle() so it can be tested without the LLM."""
        self.view = None
        if intent.task == "unsupported":
            return (f"The dataset has no {intent.missing_attribute or 'such'} information: it only contains "
                    "titles, ingredient lists and instructions. I won't guess a value.")
        if intent.task == "other":
            return HELP
        if intent.task == "show_recipe":
            return self.show_recipe(intent.recipe_number or 1)
        if intent.task in ("substitute", "stats"):
            return (self.run_substitute if intent.task == "substitute" else self.run_stats)(intent)
        if intent.task == "reset":
            self.filters, self.mode, self.have, self.prefer = new_filters(), "search", [], []
            self.forget_recipe()
        note = ""
        if intent.task == "match" and intent.have:  # a new "I have ..." request
            self.mode, self.have, self.prefer, self.filters["max_steps"] = "match", intent.have, intent.prefer, None
            names = [have_name(h) for h in intent.have]  # whole words: "eggplant" is not "egg"
            dropped = [e for e in self.filters["exclude_ingredients"] if any(matches(e, n) for n in names)]
            if dropped:  # the newest statement wins, and the user is told
                self.filters["exclude_ingredients"] = [e for e in self.filters["exclude_ingredients"] if e not in dropped]
                note = f"Note: removed your earlier restriction 'without {', '.join(dropped)}' because you said you have it.\n"
        else:
            self.prefer = intent.prefer or self.prefer  # "I'd rather have fewer steps" keeps applying
            if intent.query or intent.include:
                self.mode = "search"  # a new dish or required ingredient = back to normal search
        note += self.update_filters(intent)
        return note + (self.run_match if self.mode == "match" else self.run_search)(intent.n or 5)

    def update_filters(self, intent):
        """Merge the changes into the active filters. Restrictions stay until the user removes them.
        If the new message contradicts an older one ("with X" after "no X"), the NEWEST wins and the
        returned note says so; within one message, "no X" wins."""
        f = self.filters
        if intent.query and intent.query != f["query"]:
            f["query"], f["include"], f["max_steps"] = intent.query, [], None  # new dish: keep restrictions
        dropped = [x for x in f["exclude_ingredients"] if x in intent.include and x not in intent.exclude_ingredients]
        for key in LIST_KEYS:
            f[key] = list(dict.fromkeys(f[key] + list(getattr(intent, key))))  # add, no duplicates
        f["exclude_ingredients"] = [x for x in f["exclude_ingredients"] if x not in dropped]
        f["include"] = [i for i in f["include"] if i not in f["exclude_ingredients"]]
        if intent.max_steps is not None:
            f["max_steps"] = intent.max_steps
        for item in intent.remove:
            for key in LIST_KEYS:
                f[key] = [x for x in f[key] if x != item]
            if item in ("max_steps", "steps"):
                f["max_steps"] = None
        return (f"Note: removed your earlier restriction 'without {', '.join(dropped)}' "
                "because you now asked for it.\n") if dropped else ""

    def show(self, kind, data, ids=None, **extra):
        """Remember what was shown: the view for the UI, result IDs for "show me the 2nd one"."""
        self.view = {"kind": kind, "data": data, **extra}
        if ids is not None:
            self.last_results = ids
            self.forget_recipe()

    def run_search(self, top_n):
        f = self.filters
        r = self.searcher.search(query=f["query"], top_n=top_n, **{k: v for k, v in f.items() if k != "query"})
        self.show("search", r, [x["recipe_id"] for x in r["results"]], filters=describe(f))
        lines = [f"Active filters: {describe(f)}",
                 "No recipe in the dataset matches all of these filters. You can drop a filter "
                 "(e.g. \"any number of steps is fine\")." if r["total_matches"] == 0
                 else f"{r['total_matches']} matching recipes, ranked by {r['ranking']}:"]
        for i, x in enumerate(r["results"], 1):
            lines.append(f"{i}. {x['title']}  [{x['recipe_id']}]  {x['n_steps']} steps")
            lines += [f"   {ing}: matched {', '.join(names)}" for ing, names in x["matched"].items()]
            if f["exclude_groups"] or f["exclude_ingredients"]:
                lines.append(f"   evidence, full ingredient list: {'; '.join(x['ingredients'])}")
        if r["uncertain_count"]:
            examples = "; ".join(f"{x['title']} ({', '.join(x['flags'])})" for x in r["uncertain"])
            lines.append(f"Not included: {r['uncertain_count']} recipes that MAY break your restrictions, e.g. {examples}")
        return "\n".join(lines)

    def run_match(self, n):
        f = self.filters
        # "exclude eggs too": you can't use them either. Not saved, so "eggs are fine now" brings them back
        have = [h for h in self.have if not any(matches(e, have_name(h)) for e in f["exclude_ingredients"])]
        r = match(self.searcher, have, f["exclude_ingredients"], f["exclude_groups"], f["max_steps"], self.prefer, n)
        hard = describe({k: v for k, v in f.items() if k not in ("query", "include")})
        self.show("match", r, [x["recipe_id"] for x in r["results"]], filters=hard)
        lines = [f"You have: {', '.join(r['have'])} | hard constraints: {hard}",
                 f"Ranking: {r['ranking']}. Pantry staples (salt, black pepper, water) are assumed."]
        if not r["results"]:
            lines.append("No recipe meets all hard constraints and uses any of your ingredients.")
            if r["alternatives"]:
                lines.append("Closest recipes, each BREAKING a constraint (not recommended as matches):")
                lines += [f"- {x['title']} [{x['recipe_id']}] uses {', '.join(x['used'])}; "
                          f"VIOLATES: {'; '.join(x['violates'])}" for x in r["alternatives"]]
            return "\n".join(lines)
        lines.append(f"{r['eligible_count']} eligible recipes use at least one of your ingredients. Top {len(r['results'])}:")
        for i, x in enumerate(r["results"], 1):
            lines += [f"{i}. {x['title']}  [{x['recipe_id']}]  {x['n_steps']} steps",
                      f"   uses {len(x['used'])}/{len(r['have'])}: {', '.join(x['used'])}"
                      + (f" | not used: {', '.join(x['unused'])}" if x["unused"] else ""),
                      f"   you'd also need ({len(x['missing'])}): {', '.join(x['missing']) or 'nothing'}"]
            lines += [f"   {name}: {note}" for name, note in x["quantities"].items()]
            lines += [f"   rank: {x['reason']}"] if x["reason"] else []
        if not any(x["quantities"] for x in r["results"]):
            lines.append("Matching is by ingredient name; quantity sufficiency was not verified.")
        return "\n".join(lines)

    def run_stats(self, intent):
        if intent.use_active_filters:  # "how many of these": the same set the last answer was about
            filters = {k: v for k, v in self.filters.items() if v}
            if self.mode == "match":  # matching ignores dish/include and uses what you have
                filters = {k: v for k, v in filters.items() if k not in ("query", "include")} | {"have": self.have}
        else:
            filters = {k: getattr(intent, k) for k in (*LIST_KEYS, "max_steps") if getattr(intent, k)}
        stat, n = intent.stat or "count", intent.n or 5
        extra = {"top": n} if stat == "common_ingredients" else {"n": n, "fewest": intent.fewest} if stat == "rank_steps" else {}
        r = STATS[stat](self.searcher, **extra, **filters)
        if stat == "common_ingredients":
            body = "\n".join(numbered(r["top"], lambda t: f"{t[0]}: {t[1]} recipes ({t[2]}%)"))
        elif stat == "ingredient_count":
            body = (f"Mean {r['mean']} ingredients, median {r['median']} (min {r['min']}, max {r['max']})."
                    if r["mean"] is not None else "No recipe matches these filters, so there is no average.")
        elif stat == "rank_steps":
            self.last_results = [rid for rid, _, _ in r["results"]]
            self.forget_recipe()
            body = "\n".join(numbered(r["results"], lambda t: f"{t[1]}  [{t[0]}]  {t[2]} steps"))
            body += f"\n({r['ties_at_cutoff']} recipes have the same step count as the last one shown.)"
        else:
            body = (f"{r['count']} recipes out of {r['dataset_size']} = {r['percent']}% of the dataset.\n"
                    + "\n".join(f"'{ing}' counted as: {', '.join(v)}, ..." for ing, v in r["matched_names"].items())
                    + f"\n{NORMALIZATION_NOTE}")
        self.show("stats", r, stat=stat, filters=describe(filters))
        return (f"{body}\nFilters: {describe(filters)} | recipes analyzed: {r['recipes_analyzed']} "
                f"(+{r['uncertain_excluded']} uncertain excluded) | calculation: {r['calculation']}")

    def show_recipe(self, number):
        if not 1 <= number <= len(self.last_results):
            return f"There is no result number {number}. Run a search first."
        r = self.searcher.get(self.last_results[number - 1])
        self.show("recipe", r)
        self.forget_recipe()  # shows the ORIGINAL recipe, so earlier adaptations are dropped
        self.current_recipe = r["recipe_id"]
        ingredients = "\n".join(f"- {i}" for i in r["ingredients"])
        return f"{r['title']}  [{r['recipe_id']}]\n\nIngredients:\n{ingredients}\n\nInstructions:\n{r['instructions']}"

    # ---------- Task 3: substitution ----------
    def pick_recipe(self, intent):
        """Which recipe to adapt: a numbered result, a named dish, or the one being discussed."""
        if intent.recipe_number:  # range already checked in run_substitute
            return self.last_results[intent.recipe_number - 1]
        if intent.query and intent.query != self.filters["query"]:  # (a copied dish filter is ignored)
            current = self.searcher.get(self.current_recipe) if self.current_recipe else None
            if current and set(tokens(intent.query)) <= set(tokens(current["title"])):
                return self.current_recipe  # "this banana bread recipe" = the one on screen
            hits = self.searcher.search(query=intent.query, top_n=1)["results"]
            return hits[0]["recipe_id"] if hits else None
        return self.current_recipe or (self.last_results[0] if self.last_results else None)

    def run_substitute(self, intent):
        n = intent.recipe_number
        if n and not 1 <= n <= len(self.last_results):
            return f"There is no result number {n} (I showed {len(self.last_results)}). Which recipe should I adapt?"
        rid = self.pick_recipe(intent)
        if rid is None:
            return "Which recipe should I adapt? Search for one first or name it (\"remove eggs from banana bread\")."
        make_free = not intent.replace and intent.exclude_groups and not intent.use  # "make it dairy-free"
        last = self.last_sub if self.last_sub and self.last_sub["recipe_id"] == rid else None  # same recipe only
        follow_up = not intent.replace and not make_free and last  # "use yogurt instead"
        if follow_up:  # redo the LAST swap with another substitute: start from the version before it
            recipe = last["base"]
        elif self.adapted and self.adapted["recipe_id"] == rid:  # a new swap builds on earlier ones
            recipe = self.adapted
        else:
            recipe = self.searcher.get(rid)
        replace = intent.replace or (last["replace"] if follow_up else [])
        if make_free:
            hits = [n for _, _, ev in group_flags(recipe["ingredients_norm"], intent.exclude_groups) for n in ev]
            if not hits:
                return (f"{recipe['title']} has no {' or '.join(intent.exclude_groups)} ingredients by name, "
                        "so there is nothing to replace (product labels may still matter).")
            # start with an ingredient we have candidates for (cream before parmesan)
            # same rule as substitute.py: 'almond flour' has no entry of its own, 'heavy cream' -> cream
            replace = [next((n for n in hits if substitutes_key(n)), hits[0])]
        if not replace:
            return "Which ingredient should I replace or remove?"
        # restrictions of earlier swaps stay: a recipe made dairy-free must not get yogurt for the eggs
        groups = list(dict.fromkeys([*intent.exclude_groups, *self.filters["exclude_groups"],
                                     *recipe.get("groups", []), *(last["groups"] if follow_up else [])]))
        cached = last.get("evidence") if follow_up else None  # same recipe + target: reuse the evidence
        r = substitute(self.searcher, recipe, replace, groups, self.filters["exclude_ingredients"], intent.use, cached)
        self.current_recipe = rid
        self.last_sub = {"recipe_id": rid, "base": recipe, "replace": replace, "groups": groups,
                         "evidence": r.get("evidence")}
        if "chosen" in r:  # success: remember the adapted version for the next swap
            done = f"{', '.join(replace)} -> {r['chosen']['substitute']}"
            self.adapted = {**recipe, "ingredients": r["ingredients"], "instructions": r["instructions"],
                            "ingredients_norm": normalize_list(r["ingredients"]), "groups": groups,
                            "changes": [*recipe.get("changes", []), done]}
        self.show("substitute", r)
        return format_substitution(r)


def format_substitution(r):
    """Plain-text answer (CLI). The web UI draws the same dict as cards."""
    rec = r["recipe"]
    lines = [f"Adapting: {rec['title']} [{rec['recipe_id']}]"
             + (f" (your adapted version: {'; '.join(rec['changes'])})" if rec.get("changes") else "")]
    if r.get("targets"):
        lines.append(f"Replacing: {'; '.join(r['targets'])} (role: {r.get('role', '?')})")
    if r.get("evidence"):
        ev, x = r["evidence"], ", ".join(r["replace"])
        lines.append(f"Local evidence: {ev['similar']} similar recipes ({ev['similar_with']} with {x}, "
                     f"{ev['similar_without']} without). Candidates (amounts are cooking rules of thumb, not data):")
        for c in ev["candidates"]:
            ids = ", ".join(i for i, _ in c["ids"])
            lines.append(f"- {c['substitute']} ({c['amount']}): {STRENGTH_TEXT[c['strength']]}. "
                         f"{c['direct']} similar {x}-free recipes use it ({c['pct_without']}% vs {c['pct_with']}% "
                         f"of versions with {x}); indirect {c['indirect']}"
                         + (f"; e.g. '{c['example']}'" if c["example"] else "") + (f" [{ids}]" if ids else "")
                         + (" | already in this recipe" if c.get("already_in_recipe") else "")
                         + (f" | NOT usable: {', '.join(c['problems'])}" if c["problems"] else ""))
    if r.get("error"):
        return "\n".join(lines + [r["error"]])
    c = r["chosen"]
    lines.append(f"Chosen: {c['substitute']} ({c['amount']}), evidence: {c['strength']}.")
    if r["weak_evidence"]:
        lines.append("WARNING: the dataset evidence for this substitute is weak or conflicting; treat it as a guess.")
    lines += ["\nUpdated ingredients:\n" + "\n".join(f"- {i}" for i in r["ingredients"]),
              f"\nRevised instructions:\n{r['instructions']}"]
    if r["effects"]:
        lines.append("\nPredicted effects (not verified by the dataset):\n" + "\n".join(f"- {e}" for e in r["effects"]))
    lines += [f"\nWARNING: the recipe still contains (or may contain) {g}: {', '.join(names)}. "
              f"It is NOT {g}-free yet; ask me to replace that too." for g, names in r.get("still_contains", {}).items()]
    if r["note"]:
        lines.append(f"\nNote: {r['note']}")
    lines.append("\nSources: local dataset only (no internet search). This adaptation is untested, and recipe text "
                 "can't confirm allergen safety: check product labels.")
    return "\n".join(lines)
