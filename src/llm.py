"""The LLM's only job: read the user's latest message and return structured filter CHANGES.
All recipes, numbers and facts in answers come from Python (search.py / stats.py), not the LLM."""

import os
import re
import time
from functools import lru_cache
from typing import Literal

import requests
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field

load_dotenv()
FAST_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"      # ~1-3 s per message
FALLBACK_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"  # bigger and slower; used only if the fast one fails


class Intent(BaseModel):
    """What the latest message asks for. Fixed choices (Literal) stop the model inventing values."""
    task: Literal["search", "match", "stats", "show_recipe", "unsupported", "reset", "other"] = "search"
    query: str | None = Field(None, description="dish name to search in titles, e.g. 'penne pasta'")
    include: list[str] = Field([], description="ingredients that must be present")
    exclude_ingredients: list[str] = Field([], description="specific ingredients to avoid")
    exclude_groups: list[Literal["dairy", "nuts", "gluten"]] = []
    max_steps: int | None = None
    remove: list[str] = Field([], description="earlier filters the user drops: ingredient, "
                                              "group, or 'max_steps'")
    stat: Literal["count", "common_ingredients", "ingredient_count", "rank_steps"] | None = None
    n: int | None = Field(None, description="how many results the user asked for")
    fewest: bool = True
    use_active_filters: bool = Field(False, description="stats question about the current search")
    recipe_number: int | None = Field(None, description="'show me the second one' -> 2")
    missing_attribute: str | None = Field(None, description="for task unsupported, e.g. 'calories'")
    have: list[str] = Field([], description="task match: ingredients the user HAS, as written, with quantities")
    prefer: list[Literal["most_used", "fewest_missing", "fewest_steps"]] = []


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
- task=unsupported: asks for data the dataset doesn't have (calories, nutrition, cooking time, servings, cost, ratings).
- task=reset: "start over", "new search". task=other: greetings, help, anything else.

- EVERY ingredient named in the latest message must appear in include, have or exclude_ingredients.
- Active filters are context only: never copy them into your output.

Reply with ONLY one JSON object. Allowed keys: task, query, include, exclude_ingredients, exclude_groups,
max_steps, remove, stat, n, fewest, use_active_filters, recipe_number, missing_attribute, have, prefer.
Omit empty keys.

Examples:
"Find a penne pasta recipe" -> {"task": "search", "query": "penne pasta"}
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
"""


@lru_cache(maxsize=None)
def _model(name):
    # Thinking OFF: by default the model writes ~375 hidden "reasoning" tokens before a one-line
    # JSON answer, which made every turn ~10x slower. Parsing a message doesn't need it.
    return ChatNVIDIA(model=name, temperature=0, max_tokens=400).bind(
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


def parse(message, active_filters, retries=2):
    """Message + current filters (as text) -> Intent. The model writes JSON as text; Pydantic
    validates it (unknown values like task='recipe' are rejected). Retries only temporary API errors."""
    prompt = [("system", SYSTEM), ("user", f"Active filters: {active_filters}\nLatest message: {message}")]
    error = None
    for name in _models():  # try the fast model; switch to the fallback only if it fails
        for attempt in range(retries + 1):
            try:
                text = _model(name).invoke(prompt).content
                return Intent.model_validate_json(text[text.find("{"): text.rfind("}") + 1])
            except Exception as e:
                error = e
                if attempt == retries or not is_temporary(e):
                    break  # give up on this model, try the next one
                time.sleep(2 * (attempt + 1))  # free endpoint has short outages (503): wait, then retry
    raise error