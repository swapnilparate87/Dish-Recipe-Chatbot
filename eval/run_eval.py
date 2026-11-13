"""Evaluates every component on the golden set (eval/golden.py) and writes eval/results.json + eval/RESULTS.md.
Run:  python -m eval.run_eval          (about 5-10 minutes: calls the real LLM, one request at a time)
      python -m eval.run_eval --no-llm (deterministic components only, no API key needed)

Correctness is judged by INDEPENDENT rules, not by re-running the system's own code:
- constraint checks scan the ORIGINAL ingredient lines with plain keyword regexes (ORACLE below)
- statistics are recomputed with plain Python loops (one recipe at a time) and compared
LLM calls are logged by src/llm.py (latency, time to first token, tokens)."""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from eval import golden
from src import llm, stats
from src.data_loader import load_recipes
from src.matcher import SORT_KEYS, have_name, match
from src.normalize import check_group, count_steps, group_flags, matches, normalize_ingredient
from src.rules import PANTRY
from src.search import RecipeSearch
from src.session import Session
from src.substitute import substitute

OUT = Path(__file__).parent
NO_LLM = "--no-llm" in sys.argv

# ---------- independent oracle: keyword regexes over the ORIGINAL ingredient text ----------
ORACLE = {
    "dairy": (r"\b(milk|buttermilk|butter|cheese|cheddar|parmesan|mozzarella|ricotta|cream|yogurt|yoghurt|ghee|"
              r"whey|half-and-half|creme fraiche)\b",
              r"(coconut|almond|soy|rice|oat|cashew)\s+(milk|cream|yogurt)|(peanut|almond|cashew|sunflower|apple|"
              r"nut|seed)\s+butter|cream of tartar|(dairy|lactose)[- ]free \w+( \w+)?|non-?dairy \w+|vegan \w+"),
    "nuts": (r"\b(almonds?|walnuts?|pecans?|cashews?|pistachios?|hazelnuts?|macadamias?|peanuts?|pine nuts?|nuts?|"
             r"praline|marzipan|nutella)\b", r"(nut|peanut)[- ]free \w+"),
    "gluten": (r"\b(flour|wheat|barley|rye|bread|breadcrumbs?|crumbs|panko|pasta|penne|spaghetti|macaroni|noodles?|"
               r"lasagna|couscous|seitan|beer|crackers?|croutons?|tortillas?|soy sauce|biscuits?|cake mix)\b",
               r"gluten[- ]free \w+( \w+)?|(rice|almond|corn|coconut|tapioca|potato|chickpea)\s+(flour|noodles?|"
               r"tortillas?|pasta)"),
}


def oracle_hits(lines, constraint):
    """Words in the original ingredient lines that break a constraint (group name or one ingredient)."""
    text = " ; ".join(lines).lower()
    if constraint in ORACLE:
        bad, safe = ORACLE[constraint]
        return re.findall(bad, re.sub(safe, " ", text))
    return re.findall(rf"\b{re.escape(constraint)}(s|es)?\b", text)


def stem(w):
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def words(text):
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower())}


# ---------- timing helpers ----------
def pct(values, p):
    return round(float(np.percentile(values, p)), 3) if len(values) else None


def llm_summary(calls):
    lat, ttft = [c["latency"] for c in calls], [c["ttft"] for c in calls if c["ttft"] is not None]
    return {"llm_calls": len(calls), "llm_p50_s": pct(lat, 50), "llm_p95_s": pct(lat, 95),
            "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls)}


class Component:
    """Collects cases, wall-clock times and the LLM calls made while one component is evaluated."""
    def __init__(self, name):
        self.name, self.cases, self.times, self.start_call = name, [], [], len(llm.CALLS)
        self.started = time.perf_counter()

    def run(self, fn, *args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        self.times.append(time.perf_counter() - t)
        return result

    def add(self, case, passed, **details):
        self.cases.append({"case": case, "pass": bool(passed), **details})

    def summary(self, **metrics):
        calls = list(llm.CALLS)[self.start_call:]
        return {"component": self.name, "cases": len(self.cases), "passed": sum(c["pass"] for c in self.cases),
                "accuracy": round(sum(c["pass"] for c in self.cases) / max(len(self.cases), 1), 3), **metrics,
                "case_p50_s": pct(self.times, 50), "case_p95_s": pct(self.times, 95),
                "total_s": round(time.perf_counter() - self.started, 2), **llm_summary(calls),
                "failures": [c for c in self.cases if not c["pass"]], "details": self.cases}


# ---------- 1-4: deterministic building blocks ----------
def eval_normalize():
    c = Component("Normalization")
    for raw, want in golden.NORMALIZE:
        got = list(c.run(normalize_ingredient, raw))
        c.add(raw, got == want, expected=want, got=got)
    return c.summary()


def eval_matching():
    c = Component("Ingredient matching")
    for q, name, strict, want in golden.MATCHING:
        got = c.run(matches, q, name, strict)
        c.add(f"{q} ~ {name} ({'strict' if strict else 'broad'})", got == want, expected=want, got=got)
    return c.summary()


def eval_restrictions():
    c = Component("Restriction groups")
    for names, group, want in golden.RESTRICTIONS:
        got = c.run(check_group, names, group)[0]
        c.add(f"{names[0]} / {group}", got in want, expected=sorted(want), got=got)
    unsafe = [x for x in c.cases if x["expected"] != ["ok"]]
    safe = [x for x in c.cases if x["expected"] == ["ok"]]
    return c.summary(unsafe_flagged_recall=round(sum(x["got"] != "ok" for x in unsafe) / len(unsafe), 3),
                     safe_false_alarm_rate=round(sum(x["got"] != "ok" for x in safe) / len(safe), 3))


def eval_steps():
    c = Component("Step counting")
    for text, want in golden.STEPS:
        got = c.run(count_steps, text)
        c.add(text[:60], got == want, expected=want, got=got)
    return c.summary()


# ---------- 5: retrieval ----------
def eval_search(s):
    c = Component("Retrieval (search)")
    precisions, hits, rrs, checked, violating = [], [], [], 0, []
    for kw in golden.SEARCH:
        r = c.run(s.search, top_n=5, **kw)
        res, label = r["results"], json.dumps(kw)
        constraints = kw.get("exclude_groups", []) + kw.get("exclude_ingredients", [])
        bad = []
        for x in res:  # constraints, checked on the ORIGINAL lines; IDs must point to the same title
            row = s.get(x["recipe_id"])
            hits_ = {k: oracle_hits(row["ingredients_raw"], k) for k in constraints}
            hits_ = {k: v for k, v in hits_.items() if v}
            for inc in kw.get("include", []):
                if not oracle_hits(row["ingredients_raw"], inc):
                    hits_[f"missing {inc}"] = True
            if kw.get("max_steps") and row["n_steps"] > kw["max_steps"]:
                hits_["steps"] = row["n_steps"]
            if row["title"] != x["title"]:
                hits_["id/title mismatch"] = True
            if hits_:
                bad.append({"id": x["recipe_id"], "title": x["title"], "problems": hits_})
        checked += len(res)
        violating += bad
        if kw.get("max_steps") == 2 and "chocolate" in kw.get("include", []):  # the no-match case
            c.add(label, r["total_matches"] == 0 and not res, expected="no match", got=r["total_matches"])
            continue
        q = kw.get("query")
        rel = [words(q) <= words(x["title"]) for x in res] if q else [True] * len(res)
        p5 = sum(rel) / len(res) if res else 0.0
        rr = next((1 / (i + 1) for i, ok in enumerate(rel) if ok), 0.0)
        eligible = s.apply_filters(**{k: v for k, v in kw.items() if k != "query"})[0]
        exact = bool(q) and (eligible["title"].str.lower() == q).any()  # an exact-title recipe passes the filters
        hit1 = (res[0]["title"].lower() == q) if exact and res else None
        precisions.append(p5), rrs.append(rr)
        if hit1 is not None:
            hits.append(hit1)
        c.add(label, res and p5 >= 0.8 and not bad and hit1 is not False, precision_at_5=round(p5, 2), rr=rr,
              exact_title_top1=hit1, total_matches=r["total_matches"], violations=bad,
              top=[x["title"] for x in res])
    return c.summary(precision_at_5=round(float(np.mean(precisions)), 3), mrr=round(float(np.mean(rrs)), 3),
                     exact_title_hit_at_1=round(float(np.mean(hits)), 3) if hits else None,
                     constraint_satisfaction=round(1 - len(violating) / max(checked, 1), 3),
                     results_checked=checked)


# ---------- 6: statistics vs a plain-Python oracle ----------
def oracle_rows(s, include=(), exclude_ingredients=(), exclude_groups=(), max_steps=None):
    """Re-implementation of 'filter the complete set', one recipe at a time (no pandas filtering)."""
    rows, uncertain = [], 0
    for r in s.df.itertuples():
        ns = r.ingredients_norm
        if not all(any(matches(i, n, True) for n in ns) for i in include):
            continue
        if any(matches(e, n) for e in exclude_ingredients for n in ns):
            continue
        if max_steps is not None and r.n_steps > max_steps:
            continue
        flags = group_flags(ns, exclude_groups, r.title)
        if any(st == "contains" for _, st, _ in flags):
            continue
        if flags:
            uncertain += 1
            continue
        rows.append(r)
    return rows, uncertain


def eval_stats(s):
    c = Component("Statistics")
    fns = {"count": stats.count_recipes, "common": stats.common_ingredients,
           "ingredient_count": stats.ingredient_count_stats, "rank_steps": stats.rank_by_steps}
    disclosed = 0
    for kind, kw in golden.STATS:
        got = c.run(fns[kind], s, **dict(kw))
        filt = {k: v for k, v in kw.items() if k not in ("top", "n", "fewest")}
        rows, unc = oracle_rows(s, **filt)
        if kind == "count":
            want = {"count": len(rows), "percent": round(100 * len(rows) / len(s.df), 2)}
            have = {"count": got["count"], "percent": got["percent"]}
        elif kind == "common":  # each ingredient counted once per recipe
            cnt = Counter(n for r in rows for n in set(r.ingredients_norm))
            want, have = cnt.most_common(kw["top"]), [(n, k) for n, k, _ in got["top"]]
            want = [list(x) for x in want]
            have = [list(x) for x in have]
        elif kind == "ingredient_count":
            sizes = sorted(len(r.ingredients) for r in rows)
            want = ({"mean": round(sum(sizes) / len(sizes), 2), "median": float(np.median(sizes))} if sizes
                    else {"mean": None, "median": None})
            have = {"mean": got["mean"], "median": got["median"]}
        else:
            sign = 1 if kw.get("fewest", True) else -1
            ranked = sorted(rows, key=lambda r: (sign * r.n_steps, r.title))  # sorted() is stable
            want = [r.recipe_id for r in ranked[: kw["n"]]]
            have = [rid for rid, _, _ in got["results"]]
        ok_disclosure = all(k in got for k in ("filters", "recipes_analyzed", "calculation", "dataset_size")) \
            and got["recipes_analyzed"] == len(rows) and got["uncertain_excluded"] == unc
        disclosed += ok_disclosure
        c.add(f"{kind} {json.dumps(kw)}", want == have and ok_disclosure, expected=want, got=have,
              recipes_analyzed=got["recipes_analyzed"], oracle_analyzed=len(rows))
    # context: what a naive substring count on raw text gives for the PDF question
    naive = sum(bool(re.search(r"\bgarlic\b", " ".join(x).lower())) and bool(re.search(r"\bonions?\b", " ".join(x).lower()))
                for x in s.df["ingredients_raw"])
    return c.summary(disclosure_complete=round(disclosed / len(golden.STATS), 3),
                     naive_raw_text_count_garlic_onion=naive)


# ---------- 7: Task 2 ----------
def eval_match(s):
    c = Component("Task 2 matching")
    checked, broken = 0, 0
    for kw in golden.MATCH:
        kw = dict(kw)
        expect_none = kw.pop("expect_none", False)
        r = c.run(match, s, **kw)
        res, problems = r["results"], []
        names = [have_name(h) for h in kw["have"]]
        if expect_none:
            ok = not res and r["alternatives"] and all(a["violates"] for a in r["alternatives"])
            c.add(json.dumps(kw), ok, expected="no match + labelled alternatives",
                  got=f"{len(res)} results, {len(r['alternatives'])} alternatives")
            continue
        order = [SORT_KEYS[k] for k in dict.fromkeys([*kw.get("prefer", []), *SORT_KEYS])]
        for i, x in enumerate(res):
            row = s.get(x["recipe_id"])
            p = [f"{k}: {h}" for k in kw.get("exclude_groups", []) + kw.get("exclude_ingredients", [])
                 if (h := oracle_hits(row["ingredients_raw"], k))]
            if kw.get("max_steps") and row["n_steps"] > kw["max_steps"]:
                p.append(f"{row['n_steps']} steps")
            checked += 1
            broken += bool(p)
            if not set(x["used"]) <= set(names) or set(x["used"]) | set(x["unused"]) != set(names):
                p.append("used/unused inconsistent with what you have")
            if any(m not in row["ingredients_norm"] or m in PANTRY for m in x["missing"]):
                p.append("missing list has a name not in the recipe (or a pantry staple)")
            if i + 1 < len(res):  # sort keys in the declared order; smaller tuple = better
                def key(y):
                    vals = {"n_used": len(y["used"]), "n_missing": len(y["missing"]), "n_steps": y["n_steps"]}
                    return tuple(vals[col] * (1 if asc else -1) for col, asc, _ in order)
                if key(x) > key(res[i + 1]):
                    p.append("ranked above a better recipe")
                if not x["reason"]:
                    p.append("no ranking reason")
            for text in kw["have"]:  # quantities: compared only in the same unit, otherwise "not verified"
                note = x["quantities"].get(have_name(text))
                if note and not re.match(r"(enough|NOT enough): have|quantity not verified", note):
                    p.append(f"bad quantity note: {note}")
            if "tomato" in x["used"] and not any(re.search(r"\btomato(es)?\b(?! (sauce|paste|juice|soup|puree))", l.lower())
                                                 and "ketchup" not in l.lower() for l in row["ingredients_raw"]):
                p.append("tomato counted from a tomato product")
            if p:
                problems.append({"id": x["recipe_id"], "title": x["title"], "problems": p})
        c.add(json.dumps(kw), res and not problems, results=len(res), eligible=r["eligible_count"], problems=problems)
    return c.summary(constraint_satisfaction=round(1 - broken / max(checked, 1), 3), results_checked=checked)


# ---------- 8: Task 3 ----------
def pick(s, title, replace, expected):
    rows = s.df[s.df["title"] == title]
    if expected == "adapted" or replace == ["saffron"]:
        rows = rows[rows["ingredients_norm"].apply(lambda ns: any(matches(t, n) for t in replace for n in ns))]
    return rows.iloc[0].to_dict()


ADAPTED = []  # successful Task 3 results, for the judge


def eval_substitute(s):
    c = Component("Task 3 substitution")
    llm_ok = adapted = cites_ok = cites_total = 0
    for title, replace, kw, expected in golden.SUBSTITUTE:
        recipe = pick(s, title, replace, expected)
        r = c.run(substitute, s, recipe, replace, **kw)
        outcome = "adapted" if "chosen" in r else "refused"
        p = [] if outcome == expected else [f"expected {expected}, got {outcome}: {r.get('error')}"]
        detail = {"recipe": f"{title} [{recipe['recipe_id']}]"}
        if outcome == "adapted":
            adapted += 1
            ch, orig = r["chosen"], set(recipe["ingredients"])
            new_lines = [l for l in r["ingredients"] if l not in orig]
            for g in kw.get("exclude_groups", []):
                if oracle_hits(new_lines, g):
                    p.append(f"added {g}: {oracle_hits(new_lines, g)}")
            target_re = "|".join(rf"\b{re.escape(t)}(s|es)?\b" for t in replace)
            left = [l for l in r["ingredients"] if re.search(target_re, l.lower())
                    and not re.search(rf"\b{re.escape(ch['substitute'])}", l.lower())]
            if left:
                p.append(f"target still in ingredients: {left}")
            if re.search(target_re, r["instructions"].lower()):
                p.append("target still mentioned in instructions")
            sub_word = ch["substitute"].split()[-1]
            # the conversion may name a helper ingredient ("1 tbsp ground flaxseed + 3 tbsp water")
            helpers = {w for w in re.findall(r"[a-z]{4,}", ch["amount"].lower())} - {
                "amount", "start", "same", "rule", "thumb", "each", "with", "cups", "teaspoon", "tablespoon",
                "tablespoons", "teaspoons", "less", "more", "reduce", "about", "instead", "plus"}
            invented = [l for l in new_lines if sub_word not in l.lower() and not any(h in l.lower() for h in helpers)]
            if invented:
                p.append(f"lines not from the recipe or the substitute: {invented}")
            for rid, _ in ch["ids"]:  # every cited recipe exists, lacks the target and uses the substitute
                row = s.get(rid)
                cites_total += 1
                good = row is not None and not re.search(target_re, " ".join(row["ingredients_raw"]).lower()) \
                    and sub_word in " ".join(row["ingredients_raw"]).lower()
                cites_ok += good
                if not good:
                    p.append(f"citation {rid} does not support the swap")
            weak = ch["strength"] in ("weak", "conflicting", "none")
            if weak != r["weak_evidence"]:
                p.append("weak-evidence warning missing")
            llm_ok += r["note"] is None
            ADAPTED.append(r)  # graded for quality by the judge model later
            detail.update(chosen=ch["substitute"], strength=ch["strength"], llm_rewrite_accepted=r["note"] is None,
                          still_contains=r["still_contains"], effects=len(r["effects"]))
        c.add(f"{title}: replace {replace} {json.dumps(kw)}", not p, problems=p, **detail)
    return c.summary(llm_rewrite_acceptance=round(llm_ok / max(adapted, 1), 3),
                     citation_validity=round(cites_ok / max(cites_total, 1), 3), citations_checked=cites_total)


# ---------- 9: parser (LLM) ----------
def eval_parser():
    c = Component("LLM parser")
    fields = fields_ok = task_ok = 0
    for message, active, want in golden.PARSER:
        try:
            got = c.run(llm.parse, message, active)
        except Exception as e:
            c.add(message, False, error=type(e).__name__)
            continue
        norm = lambda k, v: [have_name(h) for h in v] if k == "have" else v
        wrong = {k: getattr(got, k) for k in want if norm(k, getattr(got, k)) != norm(k, want[k])}
        fields, fields_ok = fields + len(want), fields_ok + len(want) - len(wrong)
        task_ok += got.task == want.get("task", got.task)
        c.add(message, not wrong, wrong=wrong)
    return c.summary(task_accuracy=round(task_ok / len(golden.PARSER), 3),
                     field_accuracy=round(fields_ok / max(fields, 1), 3))


# ---------- 10: end-to-end conversations ----------
def eval_conversations(s, oracle):
    c = Component("End-to-end conversations")
    turns = []
    for name, script in golden.CONVERSATIONS.items():
        session = Session(s)
        for message, check in script:
            before, start = list(session.last_results), len(llm.CALLS)
            t = time.perf_counter()
            answer = session.handle(message)
            latency = time.perf_counter() - t
            calls = list(llm.CALLS)[start:]
            v, data = session.view or {}, (session.view or {}).get("data") or {}
            ok = CHECKS[check](s, session, v, data, answer, before, oracle)
            turns.append({"latency": latency, "calls": calls})
            c.times.append(latency)
            c.add(f"{name} | {message[:70]}", ok, check=check, latency_s=round(latency, 2),
                  tokens=sum(x["input_tokens"] + x["output_tokens"] for x in calls), answer=answer[:400])
    convs = Counter()
    for x in c.cases:
        convs[x["case"].split(" | ")[0]] += not x["pass"]
    lat = [t["latency"] for t in turns]
    per_turn = {"turn_p50_s": pct(lat, 50), "turn_p95_s": pct(lat, 95),
                "avg_llm_calls_per_turn": round(sum(len(t["calls"]) for t in turns) / len(turns), 2),
                "avg_tokens_per_turn": round(sum(x["input_tokens"] + x["output_tokens"]
                                                 for t in turns for x in t["calls"]) / len(turns), 1)}
    return c.summary(conversations=len(convs), conversations_fully_passed=sum(v == 0 for v in convs.values()),
                     **per_turn)


def no_hits(s, ids, k):
    return all(not oracle_hits(s.get(i)["ingredients_raw"], k) for i in ids)


CHECKS = {
    "search_results": lambda s, ss, v, d, a, b, o: v.get("kind") == "search" and len(d["results"]) > 0,
    "dairy_free_results": lambda s, ss, v, d, a, b, o: v.get("kind") == "search" and d["results"]
        and "dairy" in ss.filters["exclude_groups"] and "penne" in (ss.filters["query"] or "")
        and no_hits(s, [x["recipe_id"] for x in d["results"]], "dairy"),
    "shows_second": lambda s, ss, v, d, a, b, o: v.get("kind") == "recipe" and len(b) > 1 and d["recipe_id"] == b[1],
    "shows_first": lambda s, ss, v, d, a, b, o: v.get("kind") == "recipe" and b and d["recipe_id"] == b[0],
    "garlic_onion": lambda s, ss, v, d, a, b, o: d.get("count") == o["garlic_onion"] and "%" in a
        and str(len(s.df)) in a,
    "tomato_steps": lambda s, ss, v, d, a, b, o: [x[0] for x in d.get("results", [])] == o["tomato_steps"],
    "gluten_free_avg": lambda s, ss, v, d, a, b, o: d.get("mean") == o["gluten_free_mean"],
    "pdf_match": lambda s, ss, v, d, a, b, o: v.get("kind") == "match" and len(d["results"]) == 5
        and all(x["n_steps"] <= 6 for x in d["results"]) and no_hits(s, [x["recipe_id"] for x in d["results"]], "peanut"),
    "no_eggs": lambda s, ss, v, d, a, b, o: v.get("kind") == "match" and d["results"]
        and no_hits(s, [x["recipe_id"] for x in d["results"]], "egg"),
    "steps_first": lambda s, ss, v, d, a, b, o: v.get("kind") == "match" and d["ranking"].startswith("has fewer steps"),
    "no_match": lambda s, ss, v, d, a, b, o: v.get("kind") == "search" and d["total_matches"] == 0 and "No recipe" in a,
    "refuses_calories": lambda s, ss, v, d, a, b, o: "won't guess" in a and not re.search(r"\d+\s*(k?cal|calories)", a),
    "adapted_no_egg": lambda s, ss, v, d, a, b, o: v.get("kind") == "substitute" and "chosen" in d
        and not oracle_hits(d["ingredients"], "egg"),
    "flaxseed_weak": lambda s, ss, v, d, a, b, o: "chosen" in d and "flax" in d["chosen"]["substitute"]
        and d["weak_evidence"] == (d["chosen"]["strength"] in ("weak", "conflicting", "none"))
        and (not d["weak_evidence"] or "WARNING" in a),
    "cream_dairy_free": lambda s, ss, v, d, a, b, o: "chosen" in d
        and not oracle_hits([d["chosen"]["substitute"]], "dairy")
        and (not oracle_hits(d["ingredients"], "dairy") or "dairy" in d["still_contains"]),
}


# ---------- 11: LLM judge for the free text of Task 3 (the only part with no exact answer) ----------
JUDGE_MODEL = llm.FALLBACK_MODEL  # the bigger model grades the smaller one's rewrites (not self-grading)
JUDGE_SYSTEM = """You are a strict professional chef reviewing an adapted recipe. One ingredient was replaced.
Score each item from 1 (bad) to 5 (excellent):
- instructions: are the revised instructions coherent, complete and correct for the substitute?
- amount: is the substitute's amount reasonable for the amount replaced?
- effects: are the predicted effects on taste/texture plausible and specific?
Reply with ONLY JSON: {"instructions": n, "amount": n, "effects": n, "comment": "one sentence"}"""


class Verdict(BaseModel):
    instructions: int
    amount: int
    effects: int
    comment: str


def eval_judge():
    c = Component(f"Task 3 quality (judge: {JUDGE_MODEL.split('/')[-1]})")
    scores = []
    for r in ADAPTED:
        rec, ch = r["recipe"], r["chosen"]
        user = (f"Original recipe: {rec['title']}\nIngredients:\n" + "\n".join(rec["ingredients"])
                + f"\nInstructions:\n{rec['instructions']}\n\nReplaced: {'; '.join(r['targets'])} -> {ch['substitute']}"
                + "\n\nAdapted ingredients:\n" + "\n".join(r["ingredients"])
                + f"\nAdapted instructions:\n{r['instructions']}\nPredicted effects:\n" + "\n".join(r["effects"]))
        try:
            text = c.run(llm._call, JUDGE_MODEL, 400, [("system", JUDGE_SYSTEM), ("user", user)])
            v = Verdict.model_validate_json(text[text.find("{"): text.rfind("}") + 1])
        except Exception as e:
            c.add(f"{rec['title']}: {ch['substitute']}", False, error=type(e).__name__)
            continue
        scores.append(v)
        c.add(f"{rec['title']}: {'/'.join(r['replace'])} -> {ch['substitute']}",
              min(v.instructions, v.amount, v.effects) >= 3, **v.model_dump())
    avg = lambda k: round(float(np.mean([getattr(v, k) for v in scores])), 2) if scores else None
    return c.summary(avg_instructions=avg("instructions"), avg_amount=avg("amount"), avg_effects=avg("effects"))


# ---------- 12: one-shot LLM baseline ----------
def eval_one_shot(s, oracle):
    c = Component("One-shot LLM baseline")
    for (question, kind), model in [(q, m) for m in (llm.FAST_MODEL, llm.FALLBACK_MODEL) for q in golden.ONE_SHOT]:
        answer = c.run(llm._call, model, 600, [("user", question)])
        if kind == "garlic_onion":
            numbers = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", answer)]
            ok, note = oracle["garlic_onion"] in numbers, f"stated {numbers[:3]}, true count {oracle['garlic_onion']}"
        elif kind == "tomato_ids":
            ids = re.findall(r"\b(?:ar|epi|fn):[\w./]+", answer)
            real = [i for i in ids if s.get(i) is not None]
            ok, note = ids[:5] == oracle["tomato_steps"], f"{len(ids)} IDs given, {len(real)} exist in the dataset"
        else:
            invented = re.findall(r"\d[\d,]*\s*(?:k?cal|calories)", answer, re.I)
            ok, note = not invented, f"invented values: {invented[:3]}" if invented else "no number invented"
        c.add(f"[{model.split('/')[-1]}] {question[:70]}", ok, check=kind, note=note, answer=answer[:300])
    return c.summary(grounded_answers=sum(x["pass"] for x in c.cases))


# ---------- report ----------
def markdown(report):
    L = ["# Evaluation results", "", f"Generated by `python -m eval.run_eval`. Dataset: {report['dataset_size']} "
         f"recipes; LLM: `{report['model']}`; startup (load + index): {report['startup_s']} s.", "",
         "## Summary", "", "| Component | Cases | Passed | Accuracy | Key metrics | Case p50 / p95 (s) | LLM calls | "
         "LLM p50 / p95 (s) | TTFT p50 / p95 (s) | Tokens in / out |", "|---|---|---|---|---|---|---|---|---|---|"]
    skip = {"component", "cases", "passed", "accuracy", "case_p50_s", "case_p95_s", "total_s", "llm_calls", "llm_p50_s",
            "llm_p95_s", "ttft_p50_s", "ttft_p95_s", "input_tokens", "output_tokens", "failures", "details"}
    for r in report["components"]:
        key = "; ".join(f"{k} {v}" for k, v in r.items() if k not in skip)
        L.append(f"| {r['component']} | {r['cases']} | {r['passed']} | {r['accuracy']:.0%} | {key or '-'} | "
                 f"{r['case_p50_s']} / {r['case_p95_s']} | {r['llm_calls']} | {r['llm_p50_s']} / {r['llm_p95_s']} | "
                 f"{r['ttft_p50_s']} / {r['ttft_p95_s']} | {r['input_tokens']} / {r['output_tokens']} |")
    t = report["totals"]
    L += ["", "## Whole evaluation", "", f"- Wall time: {t['wall_s']} s; LLM calls: {t['llm_calls']}; tokens: "
          f"{t['input_tokens']} in + {t['output_tokens']} out = {t['input_tokens'] + t['output_tokens']}",
          f"- All LLM calls: latency p50 {t['llm_p50_s']} s, p95 {t['llm_p95_s']} s; "
          f"time to first token p50 {t['ttft_p50_s']} s, p95 {t['ttft_p95_s']} s"]
    if "one_response" in report:
        o = report["one_response"]
        L += ["", "## One chatbot response (end-to-end conversation turns)", "",
              f"- Latency p50 {o['turn_p50_s']} s, p95 {o['turn_p95_s']} s "
              "(the answer is shown in one piece, so this is also the user's time to first token)",
              f"- On average {o['avg_llm_calls_per_turn']} LLM calls and {o['avg_tokens_per_turn']} tokens per turn"]
    L += ["", "## Failures", ""]
    for r in report["components"]:
        for f in r["failures"]:
            L.append(f"- **{r['component']}**: `{f['case']}`: "
                     + json.dumps({k: v for k, v in f.items() if k not in ("case", "pass")}, default=str)[:500])
    return "\n".join(L) + "\n"


def main():
    t0 = time.perf_counter()
    s = RecipeSearch(load_recipes())
    startup = round(time.perf_counter() - t0, 1)
    print(f"loaded {len(s.df)} recipes in {startup} s")
    parts = [eval_normalize(), eval_matching(), eval_restrictions(), eval_steps()]
    print("building blocks done")
    parts += [eval_search(s), eval_stats(s), eval_match(s)]
    print("retrieval / statistics / matching done")
    report = {"dataset_size": len(s.df), "model": llm._models()[0], "startup_s": startup}
    if not NO_LLM:
        rows, _ = oracle_rows(s, include=["garlic", "onion"])
        tomato, _ = oracle_rows(s, include=["tomato"])
        gf, _ = oracle_rows(s, exclude_groups=["gluten"])
        oracle = {"garlic_onion": len(rows),
                  "tomato_steps": [r.recipe_id for r in sorted(tomato, key=lambda r: (r.n_steps, r.title))[:5]],
                  "gluten_free_mean": round(sum(len(r.ingredients) for r in gf) / len(gf), 2)}
        parts.append(eval_parser())
        print("parser done")
        parts.append(eval_substitute(s))
        parts.append(eval_judge())
        print("substitution done")
        conv = eval_conversations(s, oracle)
        parts.append(conv)
        report["one_response"] = {k: conv[k] for k in ("turn_p50_s", "turn_p95_s", "avg_llm_calls_per_turn",
                                                       "avg_tokens_per_turn")}
        parts.append(eval_one_shot(s, oracle))
    calls = list(llm.CALLS)
    report["components"] = parts
    report["totals"] = {"wall_s": round(time.perf_counter() - t0, 1), **llm_summary(calls)}
    (OUT / "results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT / "RESULTS.md").write_text(markdown(report), encoding="utf-8")
    for r in parts:
        print(f"{r['component']:28} {r['passed']}/{r['cases']}")
    print("written: eval/results.json, eval/RESULTS.md")


if __name__ == "__main__":
    main()
