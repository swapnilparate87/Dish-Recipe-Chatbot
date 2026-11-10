# Dish Recipe Chatbot

A conversational chatbot that searches, analyzes, matches and adapts recipes from the
[Recipe Box dataset](https://eightportions.com/datasets/Recipes/). Answers are grounded in the
dataset: facts come from retrieved recipes, and statistics are computed in Python, never guessed
by the LLM.

> Status: data loading, ingredient normalization and step counting are done.
> Search, statistics, matching, substitution and the chat interface are in progress (see [Roadmap](#roadmap)).

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

Copy `.env.example` to `.env` and add your LLM API key. <!-- TODO: list supported providers once llm.py is done -->

**4. Run**

```bash
python -m src.data_loader    # load data, print record count
python -m src.normalize      # run normalization + step-count tests, print top ingredients
python -m src.search         # run example searches (dish name, no dairy, ingredients, no match)
# python app.py              # chatbot (TODO)
```

## Project structure

| File | Purpose |
|---|---|
| `src/data_loader.py` | Loads the 3 JSON files into one pandas DataFrame |
| `src/rules.py` | Word lists only (units, descriptors, aliases, exceptions, restriction groups). Edit this file to change normalization behavior |
| `src/normalize.py` | Ingredient normalization, ingredient matching, restriction groups, step counting |
| `src/search.py` | Recipe search: hard filters, then title ranking (BM25) |
| `src/stats.py` | TODO: counts, averages, rankings |
| `src/matcher.py` | TODO: Task 2, ingredient-based matching |
| `src/substitute.py` | TODO: Task 3, evidence-based substitution |
| `src/llm.py`, `src/session.py` | TODO: LLM calls and conversation state |
| `app.py` | TODO: command-line chat interface |
| `eval/` | TODO: evaluation conversations and results |

## Design decisions

### 1. Data loading (`src/data_loader.py`)

- All three files are used: `ar` (Allrecipes), `epi` (Epicurious), `fn` (Food Network).
- **Stable ID** = `"<source>:<original key>"`, e.g. `ar:rmK12Uau.ntP510KeImX506H6Mr6jTu`.
  The source prefix keeps IDs unique across files.
- **Original data is preserved**: `title`, `ingredients_raw` (unchanged list) and `instructions`
  (unchanged text). The scraping artifact `ADVERTISEMENT` is removed only in the separate
  `ingredients` column.
- **Dropped records**: 125,164 raw records, of which 122,922 are kept. A record is dropped if
  it has no title, no ingredients (after removing `ADVERTISEMENT`), or instructions with no
  letters (e.g. `";"`).
- **Denominator**: all dataset percentages use **122,922** recipes.
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
`coconut pecan frosting` still contains nuts. Groups so far: dairy, nuts.

**How it was validated**: the rules were run on all 122,922 recipes and the 150 most common
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
- [ ] Task 1: search, filters, statistics, follow-up context
- [ ] Task 2: ingredient-based matching with hard constraints and explainable scores
- [ ] Task 3: evidence-based substitution
- [ ] Evaluation set (5+ conversations, one-shot LLM comparison)

## Evaluation

TODO: test conversations, results, and what worked or failed.

## AI tools disclosure

Claude Code was used as a guide throughout the project: planning the architecture, reviewing
code and explaining concepts. It wrote the first versions of `data_loader.py` and
`normalize.py` (normalization rules), which I reviewed and tested. I wrote `count_steps`.
<!-- TODO: keep this updated as the project continues -->
