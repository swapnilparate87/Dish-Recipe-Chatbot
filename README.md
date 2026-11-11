# Dish Recipe Chatbot

A conversational chatbot that searches, analyzes, matches and adapts recipes from the
[Recipe Box dataset](https://eightportions.com/datasets/Recipes/). Answers are grounded in the
dataset: facts come from retrieved recipes, and statistics are computed in Python, never guessed
by the LLM.

> Status: Task 1 (search, follow-ups, statistics) and Task 2 (ingredient matching) work end to end
> in the CLI. Task 3 (substitution) is in progress (see [Roadmap](#roadmap)).

## Setup

**1. Install** (Python 3.10+)

```bash
git clone <this-repo-url>
cd Dish-Recipe-Chatbot
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**2. Download the dataset** (about 200 MB, not stored in this repo)

Download `recipes_raw.zip` from https://eightportions.com/datasets/Recipes/ and unzip it so the
files are at:

```
data/recipes_raw/recipes_raw_nosource_ar.json
data/recipes_raw/recipes_raw_nosource_epi.json
data/recipes_raw/recipes_raw_nosource_fn.json
```

**3. API key**

Copy `.env.example` to `.env` and add an [NVIDIA API key](https://build.nvidia.com) (free).
`LLM_MODEL` is optional; the default is the fast `nvidia/nemotron-3.5-lightning-30b-a3b`, with
`nvidia/nemotron-3.5-super-vl-120b-a12b` as an automatic fallback. Many models listed by
NVIDIA are retired or not enabled for every key; if the default fails, pick another model enabled for your key.

**4. Run**

```bash
python -m src.data_loader    # load data, print record count
python -m src.normalize      # run normalization + step-count tests, print top ingredients
python -m src.search         # run example searches (dish name, no dairy, ingredients, no match)
python -m src.matcher        # Task 2 examples (PDF request, exclude eggs, quantities, no match)
streamlit run ui.py         # web chat UI at http://localhost:8501 (chat history + context sidebar)
python app.py               # the same chatbot in the terminal (add --debug to see what the LLM extracted)
python -m eval.test_parser   # LLM parser test cases (calls the LLM)
```

## Project structure

| File | Purpose |
|---|---|
| `src/data_loader.py` | Loads the 3 JSON files into one pandas DataFrame |
| `src/rules.py` | Word lists only (units, descriptors, aliases, exceptions, restriction groups). Edit this file to change normalization behavior |
| `src/normalize.py` | Ingredient normalization, ingredient matching, restriction groups, step counting |
| `src/search.py` | Recipe search: hard filters, then title ranking (BM25) |
| `src/stats.py` | Counts, common ingredients, ingredient-count stats, step rankings |
| `src/matcher.py` | Task 2: rank existing recipes by the ingredients you have |
| `src/substitute.py` | TODO: Task 3, evidence-based substitution |
| `src/llm.py` | LLM parser: user message -> structured filter changes (JSON validated by Pydantic) |
| `src/session.py` | Conversation state, runs search/stats, writes answers from templates |
| `ui.py` | Web chat interface (Streamlit): chat history, recipe cards, charts, follow-up buttons, editable context sidebar |
| `app.py` | Command-line chat interface |
| `eval/` | Evaluation: `test_parser.py` (LLM parsing cases); more to come |

## Design decisions

### 1. Data loading (`src/data_loader.py`)

- All three files are used: `ar` (Allrecipes), `epi` (Epicurious), `fn` (Food Network).
- **Stable ID** = `"<source>:<original key>"`, e.g. `ar:rmK12Uau.ntP510KeImX506H6Mr6jTu`.
  The source prefix keeps IDs unique across files.
- **Original data is preserved**: `title`, `ingredients_raw` (unchanged list) and `instructions`
  (unchanged text). The scraping artifact `ADVERTISEMENT` is removed only in the separate
  `ingredients` column.
- **Dropped records**: 125,164 raw records, of which 122,917 are kept. A record is dropped if
  it has no title, no ingredients (after removing `ADVERTISEMENT`), or instructions with no
  letters (e.g. `";"`), or is one of 5 website test pages (titles like `QATest2`).
- **Denominator**: all dataset percentages use **122,917** recipes.
- Duplicate titles (about 12k, e.g. many "Banana Bread"s) are kept. They are different recipes
  and have different IDs.
- No cache: loading and normalizing takes about 12 seconds once at startup, so a cache file was
  not worth the extra complexity.

### 2. Ingredient normalization (`src/normalize.py`)

Raw strings are reduced to a canonical name:
`"2 sweet onions (such as Vidalia), finely chopped"` → `sweet onion`,
`"Kosher salt and freshly ground black pepper"` → `salt`, `black pepper`.

Rules, in order:

1. Lowercase; skip section headers ending in `:` (e.g. `"For the sauce:"`).
2. Remove text in brackets.
3. Split on commas and use the first part that still contains a name (usually before the
   comma: `"garlic, minced"`; sometimes after: `"4 skinless, boneless chicken breasts"`).
4. Remove numbers and symbols, make each word singular, and drop **units** (cup, tbsp, clove…)
   and **descriptors** (fresh, chopped, large, low-sodium…).
5. Split `X and Y` / `X or Y` into two ingredients, except for single products such as
   `macaroni and cheese` and `half-and-half`.
6. Apply **aliases** for justified equivalents only: `white sugar`→`sugar`,
   `scallion`→`green onion`, `basil leaves`→`basil`, and so on.

**Matching** (`matches(query, name)`): the query must appear as whole words in the canonical
name, minus an explicit exception list:

| Query | Name | Match | Reason |
|---|---|---|---|
| garlic | garlic powder | yes | same ingredient, different form |
| tomato | cherry tomato | yes | a type of tomato |
| butter | peanut butter | **no** | different product |
| egg | eggplant | **no** | whole words only |
| onion | green onion | **no** | different plant part |

**Restriction groups** (`check_group(names, "dairy")`) return `contains` (with evidence),
`uncertain` or `ok`. Products whose contents are unclear from the name, such as
`whipped topping` or `margarine`, are **flagged as uncertain, never assumed safe**. Coconut milk
and peanut butter are correctly not treated as dairy. Each dairy or nut word found in a name is
checked on its own, so `peanut butter cream frosting` is still dairy (because of the cream) and
`coconut pecan frosting` still contains nuts. Names with `<group>-free` (e.g. `gluten-free pasta`)
are ok for that group. Groups so far: dairy, nuts, gluten. Distilled drinks such as rye whiskey
are treated as gluten (strict on purpose: over-excluding is safer for allergies).

**The title is also checked.** Some recipes leave a main ingredient out of their ingredient list
(e.g. *"Scallops and Penne Pasta in a Saffron Broth"* lists no pasta). If the title names a
restricted ingredient that the ingredient list doesn't contain, the recipe is marked **uncertain**
rather than safe.

**How it was validated**: the rules were run on all 122,917 recipes and the 150 most common
canonical names were reviewed by hand. Each junk name was traced to its cause and fixed. For
example, `ground` came from `ground cloves` because "clove" was treated as a garlic unit, and
`sauce` came from `hot sauce` because "hot" was treated as a descriptor.

### 3. Step counting (`count_steps` in `src/normalize.py`)

**A step is one sentence**: text ending in `.`, `!` or `?` followed by whitespace and a capital
letter.

- Sentences instead of lines: about 15,000 recipes put all instructions on one line, so counting
  lines would rank them as 1-step recipes.
- No split on newlines: some recipes wrap lines mid-sentence (`"...Colby cheese and\ncornstarch."`).
- Handled cases: `"350 degrees F. until golden"` (lowercase after the period),
  `"1.5 cups"` (decimal), `"1. Heat the oil"` (the `"1."` fragment has no letters and is
  not counted).
- The website text `"Watch how to make this recipe."` (in 8,387 Food Network recipes) is not
  counted as a step.
- Result: median 11 steps, range 1-265.

### 4. Search (`src/search.py`)

Search runs in two stages:

1. **Filter the full dataset** with hard rules: required ingredients, excluded ingredients,
   restriction groups (e.g. dairy) and maximum steps. This uses no LLM and no top-k cutoff, so
   counts and statistics use the **complete filtered set**.
2. **Rank** the remaining recipes:
   - **Dish-name query** (`"penne pasta"`): exact title matches first, then
     [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) relevance of the title. BM25 gives rare
     words like "penne" more weight than common words like "pasta".
   - **Ingredient-only query** (`"recipes with chicken and rice"`): every recipe that contains
     the ingredients, whatever the title, ordered by fewest ingredients and then fewest steps.

**Inclusion is strict and exclusion is broad.** For "with chicken", `chicken broth` does **not**
count, because it is a product made from chicken (see `DERIVED` in `normalize.py`). For
"without chicken", `chicken broth` **is** removed, because over-excluding is the safe choice
for a restriction.

**Uncertain recipes** (e.g. pesto or chocolate for "no dairy") are not counted as matches. They are
returned in a separate `uncertain` list with the ingredient that caused the flag.

Each result includes its recipe ID, title, step count, the matched ingredient names and the full
ingredient list, so the user can check the evidence.

### 5. Statistics (`src/stats.py`)

All statistics are computed with pandas on the **complete filtered set** from `apply_filters()`.
The LLM never calculates them. Every answer includes the filters, the number of recipes analyzed,
the dataset size (122,917) and the calculation used.

| Function | Example question |
|---|---|
| `count_recipes` | "How many recipes contain garlic and onion? What %?" → 18,662 (15.18%) |
| `common_ingredients` | "Most common ingredients in dairy-free recipes?" (each ingredient counted once per recipe) |
| `ingredient_count_stats` | "Average / median number of ingredients?" → mean 10.39, median 9 |
| `rank_by_steps` | "Five tomato recipes with the fewest steps?" (ties broken by title A-Z, tie count reported) |

**Uncertain recipes** (e.g. pesto for dairy-free) are not included in the analyzed set. Their
number is reported separately as `uncertain_excluded`.

### 6. LLM and conversation (`src/llm.py`, `src/session.py`)

**The LLM only reads the message. It never writes facts.** Each turn:

```
user message -> LLM: structured filter CHANGES (JSON) -> Python merges them into the session's
active filters -> search.py / stats.py -> answer written from a template
```

- **Parser** (`llm.py`): returns an `Intent` (task, query, include, exclude_ingredients,
  exclude_groups, max_steps, remove, stat, ...). Fixed choices (`Literal`) and Pydantic
  validation reject invented values; bad JSON or API errors are retried twice.
- **Answers from templates**: every recipe, ID, count and percentage in an answer comes straight
  from the code. This also means only one LLM call per turn (~5-15 s with this model).
- **Follow-up rules** (`session.py`):
  - New restrictions are **added** to the active search ("I cannot have dairy").
  - A new dish **replaces** the query and clears required ingredients, but **restrictions stay**
    and are always shown in "Active filters".
  - The user can drop a filter ("dairy is fine now"). "No X" overrides an earlier "with X".
  - Statistics questions use only the filters in that question, unless the user says
    "of these" / "among them". They don't change the active search.
- **Unsupported questions** (calories, cooking time, servings...) get a clear refusal: the dataset
  only has titles, ingredients and instructions.

**Why prompt design mattered** (first finding for the evaluation): with only a schema and no rules,
the model turned "I am allergic to peanuts" into a **dairy** restriction and returned
`task="recipe"`. With fixed choices, explicit rules and JSON examples, it passes 10/10 parser cases
(`eval/test_parser.py`). Native structured output (`guided_json`) failed intermittently on the
NVIDIA endpoint, so the model writes JSON as text and Pydantic validates it.

### 7. Ingredient matching, Task 2 (`src/matcher.py`)

*"I have eggs, onions, tomatoes, and rice. Five recipes using as many as possible, no peanuts, at
most six steps, fewest additional ingredients."* Only existing recipes are returned; none are invented.

1. **Hard constraints** (excluded ingredients, restriction groups, max steps) are applied with the
   same `apply_filters()` as search, over all recipes.
2. For each remaining recipe:
   - **used**: your ingredients the recipe uses
   - **not used**: your ingredients it doesn't use
   - **missing**: what the recipe needs that you don't have. Pantry staples (salt, black pepper,
     water) are assumed and not counted as missing.
3. **Ranking is lexicographic and explainable**: sort by your priorities first, then the defaults:
   most of your ingredients used -> fewest missing -> fewest steps -> title A-Z. Each result says why
   it is above the next one (the first sort key that differs), e.g. *"needs fewer extra
   ingredients (8 vs 9)"*.

**Forms and products**: having fresh tomatoes does not cover `tomato sauce`, `tomato paste` or
`ketchup`, and chicken does not cover chicken broth (`PRODUCT_FORMS`, `DERIVED` in `rules.py`).
Equivalent names are matched through normalization (`eggs` = `large egg`, `rice` = `white rice`).

**Quantities**: if you give one ("4 eggs", "1 cup rice") and the recipe line uses the same unit,
it is compared (`enough: have 4, need 2`; `NOT enough: have 1 cup, need 2 cup`). Different or missing
units -> *"quantity not verified"*, showing the recipe line. Without quantities, the answer states
that quantity sufficiency was not verified.

**Never relaxing a constraint silently**: if no recipe meets every hard constraint, the bot says
"no match" and then lists the closest recipes **separately**, each labeled with what it violates
(`12 steps > 1`, `contains dairy (parmesan, heavy cream)`).

**Follow-ups**: "Exclude eggs too" adds the exclusion and removes eggs from what you have. "I'd rather
have fewer steps" moves steps to the first ranking key. A new "I have ..." request resets the step limit
and ranking; if it lists something you excluded earlier, the bot removes that exclusion **and says so**.

Trade-off: lexicographic ranking means a recipe using 4/4 of your ingredients but needing 8 extra ranks
above one using 3/4 and needing 2 extra. That follows "use as many as possible"; a user who cares more
about extras can say "fewest additional ingredients first".

### 8. Latency

A turn went from **25+ s to about 2-5 s**:

| Cause found by measuring | Fix |
|---|---|
| The model "thought" before answering: **375 of 393 output tokens were hidden reasoning** for a one-line JSON | `chat_template_kwargs={"enable_thinking": False}`: 25 output tokens |
| The 120b model was slow (~25 s) and often down (503) on the free endpoint | default to `nemotron-3.5-lightning-30b-a3b` (**1-3 s**); the 120b model is only a fallback |
| Smaller model missed 2 patterns ("gluten-free", "dairy is fine now") | two more JSON examples in the prompt: parser tests back to 12/12 |

Search/statistics code takes 0.8-2.5 s per turn. Removing a filter from the sidebar skips the LLM entirely (<1 s).

## Known limitations

- Normalization is rule-based. About 66,000 distinct names remain; most are rare, specific products.
- `pepper` alone is aliased to `black pepper`, because in this dataset it almost always comes
  from "salt and pepper".
- `cream of chicken soup` becomes `condensed cream chicken soup` and so matches "cream".
  This is acceptable for dairy checks, since it usually contains dairy.
- Step counting miscounts an abbreviation followed by a capital letter (`"Add 1 tsp. Salt"` counts as 2 steps).
- Some instructions refer to other recipes (e.g. "Repeat process as above"); these are kept as-is.
- The dataset has **no nutrition, cooking time or serving data**. The chatbot says so instead of
  estimating.

## Roadmap

- [x] Data loading and cleaning
- [x] Ingredient normalization and matching
- [x] Step counting
- [x] Task 1: search, filters, statistics, follow-up context
- [x] Task 2: ingredient-based matching with hard constraints and explainable scores
- [ ] Task 3: evidence-based substitution
- [ ] Evaluation set (5+ conversations, one-shot LLM comparison)

## Evaluation

TODO: test conversations, results, and what worked or failed.

## AI tools disclosure

Claude Code was used as a guide throughout the project: planning the architecture, reviewing
code and explaining concepts. It wrote the first versions of `data_loader.py` and
`normalize.py` (normalization rules), which I reviewed and tested. I wrote `count_steps`.
<!-- TODO: keep this updated as the project continues -->
