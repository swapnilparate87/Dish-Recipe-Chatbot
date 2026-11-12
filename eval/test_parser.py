"""Checks that the LLM turns messages into the right filters.  Run:  python -m eval.test_parser
Calls the real LLM (about 10 s per case)."""

from concurrent.futures import ThreadPoolExecutor

from src.llm import parse
from src.matcher import have_name

PENNE = 'dish "penne pasta"; no dairy'
# (message, active filters, expected fields)
CASES = [
    ("Find a penne pasta recipe", "none", dict(task="search", query="penne pasta")),
    ("I cannot have dairy", 'dish "penne pasta"', dict(task="search", exclude_groups=["dairy"], include=[])),
    ("How many recipes contain both garlic and onion? What percentage of the dataset is that?", PENNE,
     dict(task="stats", stat="count", include=["garlic", "onion"])),
    ("Among recipes containing tomato, which five have the fewest instruction steps?", PENNE,
     dict(task="stats", stat="rank_steps", include=["tomato"], n=5)),
    ("I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, "
     "with no peanuts and at most six instruction steps. Prefer recipes that need the fewest additional "
     "ingredients.", "none",
     dict(task="match", have=["eggs", "onions", "tomatoes", "rice"], exclude_ingredients=["peanut"],
          max_steps=6, n=5, include=[])),
    ("What can I make with 3 eggs and 2 cups of rice?", "none",
     dict(task="match", have=["3 eggs", "2 cups of rice"])),
    ("I'd rather have fewer steps", "with egg; without peanut", dict(prefer=["fewest_steps"])),
    ("Exclude eggs too", "with egg; with onion", dict(exclude_ingredients=["egg"])),
    ("dairy is fine now", PENNE, dict(remove=["dairy"])),
    ("show me the third one", 'dish "penne pasta"', dict(task="show_recipe", recipe_number=3)),
    ("what is the average number of ingredients in gluten-free recipes?", "none",
     dict(task="stats", stat="ingredient_count", exclude_groups=["gluten"])),
    ("How many calories does it have?", PENNE, dict(task="unsupported")),
    # Task 3
    ("Take this banana bread recipe and remove eggs. Use only ingredients that appear in the dataset. What "
     "replacements have evidence from related baked recipes, and how might the texture change?", "none",
     dict(task="substitute", replace=["egg"])),
    ("Replace cream in this pasta recipe without dairy. Prefer examples from recipes already in the database.",
     "none; current recipe: Pepperoni Penne", dict(task="substitute", replace=["cream"], exclude_groups=["dairy"])),
    ("Use yogurt instead", "none; current recipe: Banana Bread", dict(task="substitute", use="yogurt")),
    ("Make it dairy-free", "none; current recipe: Pepperoni Penne", dict(task="substitute", exclude_groups=["dairy"])),
    ("Find a penne recipe with heavy cream", "none", dict(task="search", query="penne", include=["heavy cream"])),
]


def check(case):
    message, active, expected = case
    try:
        intent = parse(message, active)
    except Exception as e:
        return False, f"ERROR {type(e).__name__}"
    got = lambda k: [have_name(h) for h in intent.have] if k == "have" else getattr(intent, k)
    want = lambda k: [have_name(h) for h in expected[k]] if k == "have" else expected[k]
    wrong = {k: got(k) for k in expected if got(k) != want(k)}  # "have" compared after normalization
    return not wrong, wrong


if __name__ == "__main__":
    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(check, CASES))
    for (message, _, _), (ok, wrong) in zip(CASES, results):
        print("OK  " if ok else "FAIL", message, "" if ok else f"-> got {wrong}")
    print(f"\n{sum(ok for ok, _ in results)}/{len(CASES)} passed")
