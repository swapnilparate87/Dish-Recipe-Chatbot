# Dish Recipe Chatbot

A conversational chatbot that searches, analyzes, matches and adapts recipes from the
[Recipe Box dataset](https://eightportions.com/datasets/Recipes/). Answers are grounded in the
data: recipes are retrieved from the dataset, statistics are computed with pandas, and the LLM is
used only to understand the user's message (and, in Task 3, to rewrite a recipe, which code then checks).

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/swapnilparate87/Dish-Recipe-Chatbot.git
cd Dish-Recipe-Chatbot
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

1. **Dataset** (about 200 MB, not in this repo): download `recipes_raw.zip` from
   https://eightportions.com/datasets/Recipes/ and unzip it so these files exist:
   `data/recipes_raw/recipes_raw_nosource_ar.json`, `..._epi.json`, `..._fn.json`.
2. **API key**: copy `.env.example` to `.env` and add a free [NVIDIA API key](https://build.nvidia.com).
3. **Run**:

```bash
streamlit run ui.py     # web chat at http://localhost:8501 (python ui.py also works)
python app.py           # same chatbot in the terminal (--debug shows what the LLM extracted)
```

Startup takes about 15 s (loading and normalizing 122,917 recipes). After that, a turn takes about 2-3 s (median 2.4 s; a Task 3
rewrite takes 5-7 s).

**Model**: the default is `nvidia/nemotron-3.5-lightning-30b-a3b`, with `nvidia/nemotron-3.5-super-vl-120b-a12b`
as an automatic fallback. To use another model, set `LLM_MODEL` in `.env`. Many models in NVIDIA's catalog are
retired or not enabled for every key.

**Other commands**

```bash
python -m eval.test_parser   # LLM parser test cases (calls the LLM)
python -m src.normalize      # normalization examples + step-count tests
python -m src.search         # example searches
python -m src.matcher        # Task 2 examples
python -m src.substitute     # Task 3 example (calls the LLM)
```

## Where each requirement is handled

| PDF requirement | How | Where |
|---|---|---|
| All 3 JSON files, stable ID, original title/ingredients/instructions | ID = `source:key` (e.g. `ar:rmK12...`); raw columns kept unchanged | `data_loader.py` |
| Search by dish name, ingredients, natural language | LLM turns the message into filters; pandas filters; BM25 ranks titles | `llm.py`, `search.py` |
| Follow-ups; restrictions apply to the active search | Session keeps active filters; new restrictions are added and always shown | `session.py` |
| Counts, common ingredients, mean/median, rank by steps | pandas on the **complete filtered set**; every answer states filters, recipes analyzed, calculation | `stats.py` |
| No fabricated attributes | Calories, time, servings, etc. get "the dataset doesn't have this" | `llm.py` (`unsupported`), `session.py` |
| Task 2: ingredients, quantities, hard constraints, preferences | Hard filters first, then lexicographic ranking with a reason per result; quantities compared only when units match | `matcher.py` |
| Task 2: never relax silently | "No match" first, then closest recipes listed separately with what each violates | `matcher.py` |
| Task 3: substitutes with local evidence and cited IDs | Candidates from a role table; evidence and recipe IDs from similar recipes in the dataset | `substitute.py` |
| Task 3: updated recipe, effects, restrictions | LLM rewrite **checked by code**; effects labeled as predictions; no safety claims | `substitute.py`, `llm.py` |
| Runnable interface | Streamlit web UI + CLI | `ui.py`, `app.py` |

## Project structure

| File | Purpose |
|---|---|
| `src/data_loader.py` | Loads and cleans the 3 JSON files into one pandas DataFrame |
| `src/rules.py` | Word lists only: units, descriptors, aliases, restriction groups, substitute table |
| `src/normalize.py` | Ingredient normalization, matching, restriction checks, step counting |
| `src/search.py` | Hard filters over all recipes, then title ranking (BM25) |
| `src/stats.py` | Counts, common ingredients, ingredient-count stats, step rankings |
| `src/matcher.py` | Task 2: rank recipes by the ingredients you have |
| `src/substitute.py` | Task 3: substitutes with dataset evidence, checked rewrite |
| `src/llm.py` | LLM calls: message -> `Intent` JSON (validated by Pydantic); recipe rewrite |
| `src/session.py` | Conversation state; runs the tasks; writes answers from templates |
| `ui.py` / `app.py` | Web UI (Streamlit) / command line |
| `eval/` | `golden.py` (golden set), `run_eval.py` (all metrics -> `RESULTS.md`), `test_parser.py` (parser cases) |

## Design decisions

### Data
- 125,164 raw records -> **122,917 kept**. A record is dropped if it has no title, no ingredients (after removing
  the scraping artifact `ADVERTISEMENT`), instructions with no letters, or is a website test page (`QATest2`).
- **Denominator for all percentages: 122,917.** Duplicate titles are kept: they are different recipes with different IDs.

### Normalization
Each raw ingredient is reduced to a canonical name: `"2 sweet onions (such as Vidalia), finely chopped"` ->
`sweet onion`. Steps: lowercase, drop brackets, take the main comma part, remove numbers, **units** and
**descriptors**, make words singular, split `X and Y` (except products like `half-and-half`), apply **aliases**
only for real equivalents (`scallion` -> `green onion`). The rules were checked by reviewing the 150 most
common names over the whole dataset.

**Matching** uses whole words, with exceptions for different products:

| Query | Name | Match | Why |
|---|---|---|---|
| garlic | garlic powder | yes | same ingredient, different form |
| butter | peanut butter | no | different product |
| egg | eggplant | no | whole words only |
| tomato | tomato sauce | no (Task 2) | a product, not the fresh ingredient |

**Inclusion is strict, exclusion is broad**: "with chicken" doesn't match `chicken broth`; "without chicken"
does remove it. Over-excluding is the safe side for restrictions.

### Step counting
**A step is one sentence** (`.`/`!`/`?` followed by a capital letter). Lines were rejected because about 15,000
recipes put everything on one line. Handles decimals (`1.5 cups`), numbered lists (`1. Heat`) and
`350 degrees F. until`. The boilerplate `"Watch how to make this recipe."` is not counted. Median 11 steps.

### Retrieval and ranking
1. **Hard filters over all 122,917 recipes** (required/excluded ingredients, dairy/nuts/gluten, max steps).
   No top-k cutoff, so statistics always use the complete filtered set.
2. **Ranking**: for a dish name, exact title matches first, then [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
   on titles (rare words like "penne" count more than "pasta"). For an ingredient-only request, every
   recipe containing the ingredients counts, whatever its title.
3. **Task 2 ranking is lexicographic**: the user's priorities first, then most of your ingredients used ->
   fewest missing -> fewest steps -> title. Each result says why it ranks above the next one. Salt, pepper and
   water are assumed to be in the pantry.

### Computation
All numbers come from pandas and are reproducible. Example: recipes with garlic **and** onion = 18,662 / 122,917
= **15.18%**. Rankings break ties by title, and the number of ties at the cutoff is reported.

### Uncertainty
- **Unclear products** (`margarine`, `whipped topping`, `pesto` for dairy) are marked **uncertain**, never
  assumed safe. They are left out of results and statistics and reported separately.
- If a **title** names a restricted ingredient that the ingredient list lacks, the recipe is marked uncertain.
- Task 2: quantities are compared only when the units match; otherwise "quantity not verified".
- Task 3: evidence strength is shown as strong / moderate / indirect / weak / conflicting / none. Effects are labeled
  **predictions**, and allergen safety is never claimed.

### Use of the LLM
- **The LLM reads; Python knows.** The LLM turns each message into an `Intent` JSON (task, filters, stat, ...).
  `Literal` fields and Pydantic validation reject invented values. Answers are written from templates.
- Thinking mode is turned off and a small model is used: latency went from **25+ s to 1-3 s**. The smaller model needed
  more JSON examples in the prompt; the parser now passes **17/17** test cases.
- Only temporary API errors (429/5xx) are retried; after that, the fallback model is used.
- Task 3 is the only place the LLM writes recipe text. Code rejects a rewrite that adds a new ingredient, keeps the
  removed ingredient, or adds a restricted one. The LLM gets one retry; after that, the ingredients are swapped
  mechanically and the answer says so.

### Substitution (Task 3)
1. **Candidates** come from a small role table (`rules.py`, e.g. egg: binding, moisture, lift). It is labeled as
   **general cooking knowledge**; the amounts are rules of thumb.
2. **Evidence**: similar recipes (BM25 title match + Jaccard ingredient overlap >= 0.2), split into versions with
   and without the target ingredient. A candidate is strong when it is used in >= 3 similar recipes without the
   target **and** is clearly more common there (>= 1.5x and +3 points). It is conflicting when it is just as
   common in the versions that keep the target.
3. Results on the PDF examples:

| Request | Evidence | Chosen |
|---|---|---|
| Banana bread without eggs | 719 similar recipes; yogurt in 9 egg-free versions (7% vs 2%) -> strong; applesauce and oil -> conflicting; flaxseed -> weak | yogurt |
| Cream in pasta, no dairy | 94 similar; chicken broth in 5 cream-free versions -> strong; cream cheese and milk -> unusable (dairy) | chicken broth, with a warning that the Parmesan is still dairy |

### External search
**Not used.** The local dataset is the only source of evidence. Candidates without local evidence are labeled
"general cooking knowledge only" instead of being supported with web results.

## Known limitations
- Normalization is rule-based. About 66,000 distinct names remain; most are rare products.
- `"Add 1 tsp. Salt"` counts as 2 steps (an abbreviation followed by a capital letter).
- `pepper` alone becomes `black pepper` (in this dataset it nearly always comes from "salt and pepper").
- Task 3's role table covers 6 ingredients (egg, cream, butter, milk, sour cream, flour). For others, the user names
  the substitute ("use X instead").
- No nutrition, cooking time or serving data: the bot says so instead of estimating.
- Recipe images are not shown: the JSON only has an image ID (`picture_link`) and the separate image archive is not used.

## Evaluation

> **Authorship note:** I defined the evaluation requirements: which components to test, the metrics (accuracy,
> retrieval quality, latency p50/p95, time to first token, tokens), a golden set of 10-15 cases per component, and
> the use of a stronger NVIDIA model as a judge. The golden set, the evaluation script and the results below were
> produced by the AI coding agent. I am now reviewing them and building my own evaluation: checking the golden
> answers by hand and testing further cases and conversations myself. I will update this section with my own findings.

```bash
python -m eval.run_eval            # full run, about 4-5 min (calls the LLM one request at a time)
python -m eval.run_eval --no-llm   # deterministic components only, no API key needed (~2 min)
```

- **Golden set** ([eval/golden.py](eval/golden.py)): 141 hand-written cases over every component, plus 6
  multi-turn conversations: search with a dairy follow-up, aggregate statistics, Task 2 with "exclude eggs too",
  a no-match case with an unsupported question, Task 3 with weak evidence (flaxseed), and Task 3 cream without dairy.
- **Independent checks**: answers are not graded with the system's own code. Restriction checks scan the
  **original** ingredient text with separate keyword lists. Statistics are recomputed with plain loops, one recipe
  at a time. Cited recipe IDs are opened and checked.
- **LLM judge**: Task 3's free text (revised instructions, amounts, predicted effects) has no exact answer. It is scored
  1-5 by the larger `nemotron-3.5-super-vl-120b` model, so the 30b model that wrote it does not grade itself.
- **Latency and tokens**: every LLM call is streamed and logged with latency, time to first token (TTFT) and tokens.
- The full report is in [eval/RESULTS.md](eval/RESULTS.md).

### Results (latest run)

| Component | Passed | Key metrics | Latency p50 / p95 |
|---|---|---|---|
| Normalization | 15/15 | exact match 100% | < 1 ms |
| Ingredient matching | 12/12 | 100% | < 1 ms |
| Restriction groups | 17/18 | unsafe items flagged 100%; safe items wrongly flagged 1/8 | < 1 ms |
| Step counting | 11/12 | 92% | < 1 ms |
| Retrieval (search) | 12/14 | Precision@5 0.92, MRR 1.0, exact title ranked 1st 100%, **constraints met in 61/61 results** | 0.9 / 2.4 s |
| Statistics | 12/12 | all numbers equal the recount; filters, recipes analyzed and calculation always stated | 1.3 / 2.5 s |
| Task 2 matching | 10/10 | constraints met in 43/43 results; ranking order, quantities and no-match correct | 1.8 / 4.3 s |
| LLM parser | 17/17 | task and field accuracy 100% | 1.1 / 2.5 s |
| Task 3 substitution | 8/10 | cited IDs valid 19/19; both refusals correct; LLM rewrite accepted 6/8 | 4.9 / 7.2 s |
| Task 3 quality (120b judge) | 4/8 scored >= 3 on every item | mean scores: instructions 2.9, amount 4.4, effects 3.9 out of 5 | 1.5 / 2.3 s |
| Conversations | 16/18 turns | 4/6 conversations fully correct | 2.4 / 7.5 s per turn |

**Overall: 130 of 138 system checks passed (94%).**

| Latency and tokens | p50 | p95 |
|---|---|---|
| One chatbot answer (end to end) | 2.4 s | 7.5 s |
| One LLM call | 1.3 s | 4.4 s |
| LLM time to first token | 0.9 s | 1.9 s |

- **Tokens:** about 1,390 tokens and 1.2 LLM calls per answer.
- **Whole evaluation:** 263 s, 64 LLM calls, 62,361 tokens.
- **Startup:** about 20 s, once.

The answer is shown in one piece, so the user's time to first token equals the full answer latency.

### One-shot LLM vs this system

The same models were asked the PDF questions directly, with no access to the data:

| Question | One-shot `30b` | One-shot `120b` | This system |
|---|---|---|---|
| Recipes with garlic and onion, and % | "105, 15.6%" (wrong) | "I don't have access" | **18,662 = 15.18%** of 122,917, with filters and method |
| 5 tomato recipes with the fewest steps, with IDs | a made-up ID ("1000000...") | "I don't have access" | 5 real IDs, ranked by the stated step rule |
| Calories in Pepperoni Penne | **"550 calories"** (invented) | refused correctly | refuses: the dataset has no nutrition data |

The small model invents answers; the big one is honest but can't answer. Grounding in the data is what makes
the answers both correct and checkable. A naive text count of "garlic" and "onion" gives 20,241, which shows
that the ingredient normalization changes the result.

### What failed, and why

1. **"I'd rather have fewer steps" tightened the step limit.** When a step limit was already active, the parser
   read it as `max_steps` 4-5 instead of a ranking preference: a hard constraint changed without the user
   asking. The parser test missed this because it ran without an active limit. Fix: a prompt rule.
2. **LLM rewrites sometimes start lines with "- "**, copying the prompt's list format (Brownies, Spinach Dip). This
   is a display bug; the Python check accepted the lines because it compares normalized names.
3. **Task 3 instruction quality is the weakest part** (judge mean 2.9/5). With flaxseed, the rewrite was rejected
   twice and the mechanical fallback swapped words without saying how to prepare a flax egg. Cornstarch for all
   the flour in brownies is a poor swap (judge 2/5) although the dataset has indirect evidence for it.
4. **Retrieval:** when few titles contain every query word ("pepperoni penne"), BM25 fills the top 5 with
   partial matches ("Pepperoni Burgers").
5. **Not reproducible every run:** "Remove the eggs from this recipe" failed once in two runs; the rewrite
   varies between runs although the temperature is 0.
6. **By design:** "Add 1 tsp. Salt" counts as 2 steps; corn tortillas are flagged "uncertain" for gluten
   (over-cautious, but on the safe side).

**Why no DeepEval:** almost every output is structured (IDs, counts, filters, rankings), so exact checks against
a golden answer are stronger than an LLM judge, which costs tokens, varies between runs and needs another API
key by default. A judge is used only for Task 3's free text.

## AI tools disclosure

I used **Claude Code** (an AI coding agent) throughout the project. I did not write every line myself:
building all three tasks line by line was not possible in the 4-day deadline. Instead I directed the agent:
I set the requirements, constraints and conditions for each part, reviewed what it produced, tested the
results and asked for fixes when the behavior was wrong.

**My decisions:**
- LLM provider: NVIDIA, through LangChain
- the LLM only parses messages; every fact and number comes from pandas
- a step is one sentence
- a dish-name search shows the exact title first; an ingredient search ignores titles
- restrictions refine the active search
- no cache or parquet, and code kept short enough to explain
- what the UI must do: chat history, visible context, follow-ups, answers in under 5 s
- images left out

**Written by me:** `count_steps` and its test cases, the first version of `count_recipes`, and the start of the
gluten restriction group.

**Written by the agent, then reviewed and tested by me:**
- data loading
- the word lists and categories in `rules.py` (units, aliases, dairy/nut/gluten lists, substitute table)
- search, statistics, matching, substitution and the LLM prompt
- the Streamlit UI
- two refactors, and the parser and regression tests

The agent also proposed most of the algorithm choices (BM25 title ranking, lexicographic match ranking,
evidence-strength rules). I chose among the options and can explain each one.
