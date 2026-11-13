"""Golden evaluation set: questions with the answer we EXPECT, written by hand (not copied from the system).
Used by eval/run_eval.py. Deterministic components have exact answers; for retrieval, statistics and
Task 2/3 the expected behaviour is checked by independent rules in run_eval.py."""

from eval.test_parser import CASES as PARSER  # 17 message -> expected Intent fields

# 1. Normalization: raw ingredient line -> canonical names
NORMALIZE = [
    ("2 sweet onions (such as Vidalia), finely chopped", ["sweet onion"]),
    ("Kosher salt and freshly ground black pepper", ["salt", "black pepper"]),
    ("3 cloves garlic, minced", ["garlic"]),
    ("1 cup white sugar", ["sugar"]),
    ("4 skinless, boneless chicken breasts", ["chicken breast"]),
    ("1 (14.5 ounce) can diced tomatoes", ["tomato"]),
    ("2 tablespoons peanut butter", ["peanut butter"]),
    ("3 scallions, thinly sliced", ["green onion"]),
    ("1 cup half-and-half", ["half-and-half"]),
    ("2 large eggs, beaten", ["egg"]),
    ("1 cup chicken broth", ["chicken broth"]),
    ("For the sauce:", []),
    ("1/4 cup fresh basil leaves", ["basil"]),
    ("2 teaspoons hot sauce", ["hot sauce"]),
    ("1 1/2 cups milk", ["milk"]),
]

# 2. Ingredient matching: (query, canonical name, strict?, should match?)
MATCHING = [
    ("garlic", "garlic powder", True, True),
    ("tomato", "cherry tomato", True, True),
    ("oil", "olive oil", True, True),
    ("flour", "all-purpose flour", True, True),
    ("butter", "peanut butter", True, False),
    ("egg", "eggplant", True, False),
    ("onion", "green onion", True, False),
    ("chicken", "chicken broth", True, False),   # "with chicken": broth is not chicken
    ("chicken", "chicken broth", False, True),   # "without chicken": broth is removed too
    ("milk", "coconut milk", True, False),
    ("rice", "rice vinegar", True, False),
    ("cheese", "cream cheese", True, True),
]

# 3. Restriction groups: (ingredient names, group, acceptable statuses)
RESTRICTIONS = [
    (["coconut milk"], "dairy", {"ok"}),
    (["peanut butter"], "dairy", {"ok"}),
    (["egg"], "dairy", {"ok"}),
    (["cream of tartar"], "dairy", {"ok"}),
    (["butter"], "dairy", {"contains"}),
    (["parmesan cheese"], "dairy", {"contains"}),
    (["ghee"], "dairy", {"contains"}),
    (["peanut butter cream frosting"], "dairy", {"contains"}),
    (["margarine"], "dairy", {"uncertain"}),
    (["whipped topping"], "dairy", {"uncertain"}),
    (["almond"], "nuts", {"contains"}),
    (["coconut pecan frosting"], "nuts", {"contains"}),
    (["nutmeg"], "nuts", {"ok"}),
    (["water chestnut"], "nuts", {"ok"}),
    (["all-purpose flour"], "gluten", {"contains"}),
    (["gluten-free pasta"], "gluten", {"ok"}),
    (["soy sauce"], "gluten", {"contains", "uncertain"}),  # usually brewed with wheat
    (["corn tortilla"], "gluten", {"ok"}),
]

# 4. Step counting: instructions -> number of steps (a step = one sentence)
STEPS = [
    ("Preheat oven to 350 degrees F. Mix flour and sugar. Bake 30 minutes.", 3),
    ("Bake at 350 degrees F. until golden.", 1),
    ("Add 1.5 cups water. Stir.", 2),
    ("1. Heat the oil. 2. Add onions.", 2),
    ("Mix everything together", 1),
    ("Watch how to make this recipe. Boil water. Add pasta.", 2),
    ("Combine the cheese and\ncornstarch. Toss.", 2),
    ("Cook! Serve hot? Enjoy.", 3),
    ("Heat oil in a pan.\nAdd garlic; cook 1 minute.\nServe.", 3),
    ("Stir in the eggs, one at a time. Fold in flour. Bake for 25 to 30 minutes, or until a toothpick "
     "comes out clean. Cool.", 4),
    ("Place in a 9x13 inch pan. Refrigerate overnight.", 2),
    ("Add 1 tsp. Salt and stir.", 1),  # known limitation: abbreviation + capital letter
]

# 5. Retrieval: search() arguments. Relevance and constraints are judged by independent rules.
SEARCH = [
    dict(query="banana bread"),
    dict(query="pepperoni penne"),
    dict(query="chocolate chip cookies"),
    dict(query="beef stew"),
    dict(query="guacamole"),
    dict(query="macaroni and cheese"),
    dict(query="penne pasta", exclude_groups=["dairy"]),
    dict(query="brownies", exclude_groups=["nuts"]),
    dict(query="pancakes", exclude_ingredients=["egg"]),
    dict(query="tomato soup", max_steps=4),
    dict(query="lasagna", exclude_groups=["gluten"]),
    dict(include=["chicken", "rice"]),
    dict(include=["garlic"], exclude_groups=["dairy", "gluten"]),
    dict(query="penne pasta", include=["chocolate"], max_steps=2),  # no-match case: expect 0 results
]

# 6. Statistics: (function name, kwargs). The expected value is recomputed by a plain-Python oracle.
STATS = [
    ("count", dict(include=["garlic", "onion"])),
    ("count", dict(include=["chicken"])),
    ("count", dict(exclude_groups=["dairy"])),
    ("count", dict(include=["egg"], max_steps=5)),
    ("count", dict(include=["saffron", "marshmallow"])),  # expected 0 or tiny
    ("common", dict(top=5)),
    ("common", dict(top=5, exclude_groups=["dairy"])),
    ("ingredient_count", dict()),
    ("ingredient_count", dict(exclude_groups=["gluten"])),
    ("ingredient_count", dict(include=["saffron", "marshmallow", "lobster"])),  # empty -> no average
    ("rank_steps", dict(n=5, include=["tomato"])),
    ("rank_steps", dict(n=5, fewest=False, include=["chocolate"])),
]

# 7. Task 2 matching: match() arguments + what must hold
MATCH = [
    dict(have=["eggs", "onions", "tomatoes", "rice"], exclude_ingredients=["peanut"], max_steps=6,
         prefer=["most_used", "fewest_missing"]),                                   # PDF example
    dict(have=["onions", "tomatoes", "rice"], exclude_ingredients=["peanut", "egg"], max_steps=6,
         prefer=["most_used", "fewest_missing"]),                                   # "exclude eggs too"
    dict(have=["4 eggs", "1 cup rice", "2 tomatoes"], max_steps=6, n=3),            # quantities
    dict(have=["chicken", "garlic", "lemon"], exclude_groups=["dairy"]),
    dict(have=["pasta", "tomato", "basil"], exclude_groups=["gluten"]),
    dict(have=["flour", "sugar", "butter", "egg"], prefer=["fewest_steps"]),
    dict(have=["potato", "cheese"], exclude_groups=["nuts"], max_steps=4),
    dict(have=["tomato", "onion"], max_steps=5),                                    # tomato != tomato sauce
    dict(have=["rice", "egg", "soy sauce", "green onion"], exclude_ingredients=["garlic"], max_steps=8),
    dict(have=["saffron", "lobster"], exclude_groups=["dairy"], max_steps=1, n=3, expect_none=True),  # no match
]

# 8. Task 3 substitution: (recipe title, replace, kwargs, expected outcome)
SUBSTITUTE = [
    ("Banana Bread", ["egg"], dict(), "adapted"),
    ("Banana Bread", ["egg"], dict(use="flaxseed"), "adapted"),             # weak evidence expected
    ("Banana Bread", ["egg"], dict(exclude_groups=["dairy"]), "adapted"),   # yogurt not allowed
    ("Pepperoni Penne", ["cream"], dict(exclude_groups=["dairy"]), "adapted"),
    ("Chocolate Chip Cookies", ["butter"], dict(), "adapted"),
    ("Pancakes", ["milk"], dict(exclude_groups=["dairy"]), "adapted"),
    ("Brownies", ["flour"], dict(exclude_groups=["gluten"]), "adapted"),
    ("Spinach Dip", ["sour cream"], dict(exclude_groups=["dairy"]), "adapted"),
    ("Guacamole", ["egg"], dict(), "refused"),                              # no egg in the recipe
    ("Paella", ["saffron"], dict(), "refused"),                             # no candidates known
]

# 9. End-to-end conversations (real LLM). Each turn: (message, check name); checks are in run_eval.py
CONVERSATIONS = {
    "Task 1: search + restriction follow-up": [
        ("Find a penne pasta recipe", "search_results"),
        ("I cannot have dairy", "dairy_free_results"),
        ("Show me the second one", "shows_second"),
    ],
    "Task 1: aggregate statistics": [
        ("How many recipes contain both garlic and onion? What percentage of the dataset is that?", "garlic_onion"),
        ("Among recipes containing tomato, which five have the fewest instruction steps?", "tomato_steps"),
        ("What is the average number of ingredients in gluten-free recipes?", "gluten_free_avg"),
    ],
    "Task 2: matching + follow-ups": [
        ("I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, "
         "with no peanuts and at most six instruction steps. Prefer recipes that need the fewest additional "
         "ingredients.", "pdf_match"),
        ("Exclude eggs too", "no_eggs"),
        ("I'd rather have fewer steps", "steps_first"),
    ],
    "No match + unsupported question": [
        ("Find a penne pasta recipe with chocolate in at most 2 steps", "no_match"),
        ("How many calories does it have?", "refuses_calories"),
    ],
    "Task 3: weak evidence": [
        ("Find a banana bread recipe", "search_results"),
        ("Show me the first one", "shows_first"),
        ("Remove the eggs from this recipe", "adapted_no_egg"),
        ("Use flaxseed instead", "flaxseed_weak"),
    ],
    "Task 3: dairy-free cream": [
        ("Find a pepperoni penne recipe", "search_results"),
        ("Show me the first one", "shows_first"),
        ("Replace cream in this pasta recipe without dairy. Prefer examples from recipes already in the "
         "database.", "cream_dairy_free"),
    ],
}

# 10. One-shot LLM baseline: the same model, asked directly with no access to the data
ONE_SHOT = [
    ("How many recipes in the Recipe Box dataset (Allrecipes, Epicurious and Food Network, about 125,000 "
     "recipes) contain both garlic and onion? What percentage of the dataset is that? Give one number.",
     "garlic_onion"),
    ("In the Recipe Box dataset, among recipes containing tomato, which five have the fewest instruction "
     "steps? Give their recipe IDs.", "tomato_ids"),
    ("How many calories are in the Pepperoni Penne recipe from the Recipe Box dataset?", "calories"),
]
