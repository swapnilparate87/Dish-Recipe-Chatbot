"""Conversation state: remembers the active filters between turns, runs search/stats,
and writes the answer from templates (no LLM, so every fact comes from the data)."""

from src import stats
from src.matcher import have_name, match
from src.llm import parse
from src.normalize import matches

LIST_KEYS = ("include", "exclude_ingredients", "exclude_groups")
NORMALIZATION_NOTE = ("Ingredient names are normalized (quantities, units and preparation words removed, "
                      "plurals made singular). An ingredient matches names that contain it as a whole word "
                      "(garlic -> garlic powder), except different products (butter is not peanut butter, "
                      "chicken is not chicken broth).")


def new_filters():
    return {"query": None, "include": [], "exclude_ingredients": [], "exclude_groups": [], "max_steps": None}


def describe(f):
    """Filters as readable text: 'dish "penne pasta"; no dairy; at most 5 steps'."""
    parts = []
    if f.get("query"):
        parts.append(f'dish "{f["query"]}"')
    if f.get("have"):
        parts.append(f"uses any of: {', '.join(f['have'])}")
    parts += [f"with {i}" for i in f.get("include", [])]
    parts += [f"no {g}" for g in f.get("exclude_groups", [])]
    parts += [f"without {i}" for i in f.get("exclude_ingredients", [])]
    if f.get("max_steps") is not None:
        parts.append(f"at most {f['max_steps']} steps")
    return "; ".join(parts) or "none"


class Session:
    def __init__(self, searcher, debug=False):
        self.searcher, self.debug = searcher, debug
        self.filters = new_filters()
        self.last_results = []  # recipe IDs shown last, for "show me the second one"
        self.mode = "search"    # "search" (Task 1) or "match" (Task 2): what follow-ups rerun
        self.have, self.prefer = [], []  # Task 2: ingredients the user has, ranking priorities
        self.view = None  # structured result of the last answer, for the web UI (cards, charts)

    def handle(self, message):
        self.last_intent = None
        try:
            intent = parse(message, describe(self.filters))
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
            return ("I can search recipes (\"find a penne pasta recipe\"), apply restrictions (\"no dairy\"), "
                    "answer statistics (\"how many recipes contain garlic?\") and show a recipe "
                    "(\"show me the first one\").")
        if intent.task == "show_recipe":
            return self.show_recipe(intent.recipe_number or 1)
        if intent.task == "stats":
            return self.run_stats(intent)
        if intent.task == "reset":
            self.filters, self.mode, self.have, self.prefer = new_filters(), "search", [], []
        note = ""
        if intent.task == "match" and intent.have:  # a new "I have ..." request
            self.mode, self.have, self.filters["max_steps"] = "match", intent.have, None
            self.prefer = intent.prefer  # new request: default ranking unless the user states one
            names = [have_name(h) for h in intent.have]  # whole words: "eggplant" is not "egg"
            dropped = [e for e in self.filters["exclude_ingredients"] if any(matches(e, n) for n in names)]
            if dropped:  # the newest statement wins, and the user is told
                self.filters["exclude_ingredients"] = [e for e in self.filters["exclude_ingredients"]
                                                       if e not in dropped]
                note = f"Note: removed your earlier restriction 'without {', '.join(dropped)}' because you said you have it.\n"
        else:
            self.prefer = intent.prefer or self.prefer  # "I'd rather have fewer steps" keeps applying
            if intent.query or intent.include:
                self.mode = "search"  # a new dish or required ingredient = back to normal search
        note += self.update_filters(intent)
        return note + (self.run_match(intent.n or 5) if self.mode == "match" else self.run_search(intent.n or 5))

    def update_filters(self, intent):
        """Merge the changes into the active filters. Restrictions stay until the user removes them.
        If the new message contradicts an older one ("with X" after "no X"), the NEWEST wins and the
        returned note says so; within one message, "no X" wins."""
        f = self.filters
        if intent.query and intent.query != f["query"]:
            f["query"], f["include"], f["max_steps"] = intent.query, [], None  # new dish: keep restrictions
        new_include, new_exclude = set(intent.include), set(intent.exclude_ingredients)
        dropped = [x for x in f["exclude_ingredients"] if x in new_include and x not in new_exclude]
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

    def run_search(self, top_n):
        f = self.filters
        r = self.searcher.search(query=f["query"], top_n=top_n,
                                 **{k: v for k, v in f.items() if k != "query"})
        self.last_results = [x["recipe_id"] for x in r["results"]]
        self.view = {"kind": "search", "data": r, "filters": describe(f)}
        lines = [f"Active filters: {describe(f)}"]
        if r["total_matches"] == 0:
            lines.append("No recipe in the dataset matches all of these filters. "
                         "You can drop a filter (e.g. \"any number of steps is fine\").")
        else:
            lines.append(f"{r['total_matches']} matching recipes, ranked by {r['ranking']}:")
        for i, x in enumerate(r["results"], 1):
            lines.append(f"{i}. {x['title']}  [{x['recipe_id']}]  {x['n_steps']} steps")
            for ing, names in x["matched"].items():
                lines.append(f"   {ing}: matched {', '.join(names)}")
            if f["exclude_groups"] or f["exclude_ingredients"]:
                lines.append(f"   evidence, full ingredient list: {'; '.join(x['ingredients'])}")
        if r["uncertain_count"]:
            examples = "; ".join(f"{x['title']} ({', '.join(x['flags'])})" for x in r["uncertain"])
            lines.append(f"Not included: {r['uncertain_count']} recipes that MAY break your restrictions, "
                         f"e.g. {examples}")
        return "\n".join(lines)

    def run_match(self, n):
        f = self.filters
        # "exclude eggs too": you can't use them either. Not saved, so "eggs are fine now" brings them back
        have = [h for h in self.have
                if not any(matches(e, have_name(h)) for e in f["exclude_ingredients"])]
        r = match(self.searcher, have,f["exclude_ingredients"], f["exclude_groups"],
                  f["max_steps"], self.prefer, n)
        self.last_results = [x["recipe_id"] for x in r["results"]]
        hard = describe({k: v for k, v in f.items() if k not in ("query", "include")})
        self.view = {"kind": "match", "data": r, "filters": hard}
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
            lines.append(f"{i}. {x['title']}  [{x['recipe_id']}]  {x['n_steps']} steps")
            lines.append(f"   uses {len(x['used'])}/{len(r['have'])}: {', '.join(x['used'])}"
                         + (f" | not used: {', '.join(x['unused'])}" if x["unused"] else ""))
            lines.append(f"   you'd also need ({len(x['missing'])}): {', '.join(x['missing']) or 'nothing'}")
            for name, note in x["quantities"].items():
                lines.append(f"   {name}: {note}")
            if x["reason"]:
                lines.append(f"   rank: {x['reason']}")
        if not any(x["quantities"] for x in r["results"]):
            lines.append("Matching is by ingredient name; quantity sufficiency was not verified.")
        return "\n".join(lines)

    def run_stats(self, intent):
        if intent.use_active_filters:  # "how many of these": the same set the last answer was about
            filters = {k: v for k, v in self.filters.items() if v}
            if self.mode == "match":  # match ignores dish/include and uses what you have
                filters = {k: v for k, v in filters.items() if k not in ("query", "include")}
                filters["have"] = self.have
        else:
            filters = {k: getattr(intent, k) for k in (*LIST_KEYS, "max_steps") if getattr(intent, k)}
        s, n = self.searcher, intent.n or 5
        if intent.stat == "common_ingredients":
            r = stats.common_ingredients(s, top=n, **filters)
            body = "\n".join(f"{i}. {name}: {c} recipes ({p}%)" for i, (name, c, p) in enumerate(r["top"], 1))
        elif intent.stat == "ingredient_count":
            r = stats.ingredient_count_stats(s, **filters)
            body = (f"Mean {r['mean']} ingredients, median {r['median']} (min {r['min']}, max {r['max']})."
                    if r["mean"] is not None else "No recipe matches these filters, so there is no average.")
        elif intent.stat == "rank_steps":
            r = stats.rank_by_steps(s, n=n, fewest=intent.fewest, **filters)
            self.last_results = [rid for rid, _, _ in r["results"]]
            body = "\n".join(f"{i}. {t}  [{rid}]  {k} steps" for i, (rid, t, k) in enumerate(r["results"], 1))
            body += f"\n({r['ties_at_cutoff']} recipes have the same step count as the last one shown.)"
        else:
            r = stats.count_recipes(s, **filters)
            body = (f"{r['count']} recipes out of {r['dataset_size']} = {r['percent']}% of the dataset.\n"
                    + "\n".join(f"'{ing}' counted as: {', '.join(v)}, ..." for ing, v in r["matched_names"].items())
                    + f"\n{NORMALIZATION_NOTE}")
        self.view = {"kind": "stats", "stat": intent.stat or "count", "data": r, "filters": describe(filters)}
        return (f"{body}\nFilters: {describe(filters)} | recipes analyzed: {r['recipes_analyzed']} "
                f"(+{r['uncertain_excluded']} uncertain excluded) | calculation: {r['calculation']}")

    def show_recipe(self, number):
        if not 1 <= number <= len(self.last_results):
            return f"There is no result number {number}. Run a search first."
        r = self.searcher.get(self.last_results[number - 1])
        self.view = {"kind": "recipe", "data": r}
        ingredients = "\n".join(f"- {i}" for i in r["ingredients"])
        return f"{r['title']}  [{r['recipe_id']}]\n\nIngredients:\n{ingredients}\n\nInstructions:\n{r['instructions']}"
