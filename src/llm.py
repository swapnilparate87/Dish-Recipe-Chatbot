"""The LLM's only job: read the user's latest message and return structured filter CHANGES.
All recipes, numbers and facts in answers come from Python (search.py / stats.py), not the LLM."""

import os
import re
import time
from collections import deque
from functools import lru_cache
from typing import Literal

import requests
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel

load_dotenv()
FAST_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"      # ~1-3 s per message
FALLBACK_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"  # bigger and slower; used only if the fast one fails


class Intent(BaseModel):
    """What the latest message asks for (the prompt explains each field to the model).
    Fixed choices (Literal) stop the model inventing values; unknown values fail validation."""
    task: Literal["search", "match", "stats", "show_recipe", "substitute", "unsupported", "reset",
                  "other"] = "search"
    query: str | None = None                # dish name, searched in titles
    include: list[str] = []                 # ingredients that must be present
    exclude_ingredients: list[str] = []
    exclude_groups: list[Literal["dairy", "nuts", "gluten"]] = []
    max_steps: int | None = None
    remove: list[str] = []                  # earlier filters the user drops ("dairy is fine now")
    stat: Literal["count", "common_ingredients", "ingredient_count", "rank_steps"] | None = None
    n: int | None = None                    # how many results
    fewest: bool = True
    use_active_filters: bool = False        # stats about the current search ("of these")
    recipe_number: int | None = None        # "the second one" -> 2
    missing_attribute: str | None = None    # task unsupported: "calories"
    have: list[str] = []                    # task match: items the user has, as written ("2 cups rice")
    prefer: list[Literal["most_used", "fewest_missing", "fewest_steps"]] = []
    replace: list[str] = []                 # task substitute: ingredient(s) to take out
    use: str | None = None                  # task substitute: "use yogurt instead"


SYSTEM = """You read messages sent to a recipe chatbot and output ONLY the changes the LATEST message makes.
The dataset has only: recipe title, ingredient list, instructions. Nothing else.

Rules:
- task=search: find recipes. Dish names go in query; required ingredients in include.
- Restrictions ("no dairy", "I can't have X", "allergic to X", "without X") go in exclude_*, NEVER include.
  dairy / nuts / gluten (or "lactose", "tree nuts", "celiac", "dairy-free", "gluten-free") -> exclude_groups. One named food -> exclude_ingredients.
- Ingredient names: singular, plain ("tomatoes" -> "tomato", "eggs" -> "egg").
- "under N steps" -> max_steps N-1; "at most N steps" -> N.
- User takes back a filter ("dairy is fine now", "any number of steps") -> remove.
- task=match: the user lists ingredients they HAVE ("I have X, Y", "what can I make with X"). Put each item in
  have exactly as written incl. quantity ("2 cups rice", "eggs"); NOT in include. Ranking wishes in prefer, most
  important first: "as many of these as possible" -> most_used, "fewest additional/extra ingredients" ->
  fewest_missing, "quick/fewest steps" -> fewest_steps.
- task=stats: counting, percentages, averages, medians, most common ingredients, rankings.
  stat: count | common_ingredients | ingredient_count (average/median ingredients) | rank_steps.
  Ingredients in the question go in include, query stays null. "of these"/"among them" -> use_active_filters=true.
- task=show_recipe: user wants the full ingredients/instructions of a listed result ("the second one" -> recipe_number=2).
- task=substitute: adapt a recipe by removing/replacing an ingredient ("remove eggs", "replace the cream",
  "make it without butter"). replace = the ingredient(s) to take out (singular). The recipe is the one being
  discussed, a listed result (recipe_number) or a dish name (query, e.g. "this banana bread recipe").
  "use X instead" -> use. Restrictions for the substitute ("without dairy") -> exclude_groups.
  "this recipe" / "this pasta recipe" = the current recipe: leave query EMPTY.
- task=unsupported: asks for data the dataset doesn't have (calories, nutrition, cooking time, servings, cost, ratings).
- task=reset: "start over", "new search". task=other: greetings, help, anything else.

- EVERY ingredient named in the latest message must appear in include, have or exclude_ingredients.
- Active filters are context only: never copy them into your output.

Reply with ONLY one JSON object. Allowed keys: task, query, include, exclude_ingredients, exclude_groups,
max_steps, remove, stat, n, fewest, use_active_filters, recipe_number, missing_attribute, have, prefer,
replace, use.
Omit empty keys.

Examples:
"Find a penne pasta recipe" -> {"task": "search", "query": "penne pasta"}
"Find a penne recipe with heavy cream" -> {"task": "search", "query": "penne", "include": ["heavy cream"]}
"I cannot have dairy" -> {"task": "search", "exclude_groups": ["dairy"]}
"Exclude eggs too" -> {"task": "search", "exclude_ingredients": ["egg"]}
"Dairy is fine now" -> {"task": "search", "remove": ["dairy"]}
"Average number of ingredients in gluten-free recipes?" -> {"task": "stats", "stat": "ingredient_count", "exclude_groups": ["gluten"]}
"I have 4 eggs, onions and rice. Five recipes using as many as possible, no peanuts, at most 6 steps, fewest extra ingredients" -> {"task": "match", "have": ["4 eggs", "onions", "rice"], "exclude_ingredients": ["peanut"], "max_steps": 6, "n": 5, "prefer": ["most_used", "fewest_missing"]}
"How many recipes contain both garlic and onion?" -> {"task": "stats", "stat": "count", "include": ["garlic", "onion"]}
"Among recipes containing tomato, which five have the fewest steps?" -> {"task": "stats", "stat": "rank_steps", "include": ["tomato"], "n": 5, "fewest": true}
"How many of these are there?" -> {"task": "stats", "stat": "count", "use_active_filters": true}
"Show me the second one" -> {"task": "show_recipe", "recipe_number": 2}
"How many calories are in it?" -> {"task": "unsupported", "missing_attribute": "calories"}
"Take this banana bread recipe and remove eggs" -> {"task": "substitute", "query": "banana bread", "replace": ["egg"]}
"Replace cream in this pasta recipe without dairy" -> {"task": "substitute", "replace": ["cream"], "exclude_groups": ["dairy"]}
"Use yogurt instead" -> {"task": "substitute", "use": "yogurt"}
"""


@lru_cache(maxsize=None)
def _model(name, max_tokens=400):
    # Thinking OFF: by default the model writes ~375 hidden "reasoning" tokens before a one-line
    # JSON answer, which made every turn ~10x slower. Parsing a message doesn't need it.
    return ChatNVIDIA(model=name, temperature=0, max_tokens=max_tokens).bind(
        chat_template_kwargs={"enable_thinking": False})


def _models():
    """Fast model first (or LLM_MODEL from .env), the big model as a fallback."""
    return list(dict.fromkeys([os.getenv("LLM_MODEL") or FAST_MODEL, FALLBACK_MODEL]))


def is_temporary(error):
    """Worth retrying? Network drops, timeouts and HTTP 429 / 5xx ('[503] Service Unavailable').
    Not: bad JSON from the model (temperature 0 gives the same answer again), a wrong API key (401)."""
    if isinstance(error, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)):
        return True
    return re.match(r"\[(429|5\d\d)\]", str(error)) is not None


def parse(message, active_filters):
    """Message + current filters (as text) -> Intent. The model writes JSON as text; Pydantic
    validates it (unknown values like task='recipe' are rejected)."""
    prompt = [("system", SYSTEM), ("user", f"Active filters: {active_filters}\nLatest message: {message}")]
    return _complete(prompt, Intent)


CALLS = deque(maxlen=10_000)  # last LLM calls: model, latency, time to first token, tokens (read by eval/)


def _call(name, max_tokens, prompt):
    """Stream the answer so we can measure time to first token (TTFT); return the full text."""
    start, ttft, msg = time.perf_counter(), None, None
    for chunk in _model(name, max_tokens).stream(prompt):
        if ttft is None and chunk.content:
            ttft = time.perf_counter() - start
        msg = chunk if msg is None else msg + chunk  # adding chunks also adds up token usage
    usage = msg.usage_metadata or {}
    CALLS.append({"model": name, "latency": time.perf_counter() - start, "ttft": ttft,
                  "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)})
    return msg.content


def _complete(prompt, schema, max_tokens=400, retries=2):
    """Call the model chain and validate its JSON with a Pydantic `schema`.
    Retries only temporary API errors; a failing model hands over to the next one."""
    error = None
    for name in _models():  # try the fast model; switch to the fallback only if it fails
        for attempt in range(retries + 1):
            try:
                text = _call(name, max_tokens, prompt)
                return schema.model_validate_json(text[text.find("{"): text.rfind("}") + 1])
            except Exception as e:
                error = e
                if attempt == retries or not is_temporary(e):
                    break  # give up on this model, try the next one
                time.sleep(2 * (attempt + 1))  # free endpoint has short outages (503): wait, then retry
    raise error

# ---------- Task 3: rewriting a recipe (the only place the LLM writes recipe text) ----------
class Rewrite(BaseModel):
    ingredients: list[str]  # full updated ingredient list
    instructions: str       # full revised instructions
    effects: list[str]      # likely effects on taste/texture/preparation (predictions)


REWRITE_SYSTEM = """You adapt a recipe by replacing ONE ingredient with ONE substitute.
Strict rules:
- Keep every other ingredient line exactly as it is. Replace only the target lines.
- Use ONLY the original ingredients plus the given substitute (and an extra ingredient only if the
  conversion names it, e.g. "+ 1 tbsp butter per cup"). Add nothing else.
- Write the substitute line with a concrete amount, using the given conversion.
- Rewrite only the instruction sentences that mention the target; keep the rest word for word.
- effects: 2-4 short sentences on likely changes in taste, texture or preparation. These are predictions.
Reply with ONLY one JSON object: {"ingredients": [...], "instructions": "...", "effects": [...]}"""


def rewrite_recipe(recipe, targets, substitute, amount, role, feedback=None):
    """LLM rewrite of one recipe; substitute.py CHECKS the result before using it."""
    ingredients = "\n".join(f"- {i}" for i in recipe["ingredients"])
    user = (f"Recipe: {recipe['title']}\nIngredients:\n{ingredients}\nInstructions:\n{recipe['instructions']}\n\n"
            f"Replace the lines: {targets}\nSubstitute: {substitute} (conversion: {amount}). "
            f"Role of the replaced ingredient: {role}.")
    if feedback:  # retry after our checks rejected the first answer
        user += f"\n\nYour previous answer was rejected: {'; '.join(feedback)}. Fix exactly these problems."
    return _complete([("system", REWRITE_SYSTEM), ("user", user)], Rewrite, max_tokens=2000)
