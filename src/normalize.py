"""Ingredient normalization, ingredient matching and step counting.

"2 sweet onions (such as Vidalia), finely chopped"  ->  ("sweet onion",)
"Kosher salt and freshly ground black pepper"       ->  ("salt", "black pepper")
"""

import re
from functools import lru_cache

from src.rules import (ALIASES, BOILERPLATE, DERIVED, DESCRIPTORS, FREE_OF, GROUPS, KEEP_TOGETHER,
                       NOT_A, NOT_PLURAL, UNITS)

FREE_WORDS = sorted({w for ws in FREE_OF.values() for w in ws}, key=len, reverse=True)


def singular(word):
    """onions -> onion, tomatoes -> tomato, berries -> berry, leaves -> leaf"""
    if word in NOT_PLURAL or len(word) <= 3:
        return word
    if word == "halves":
        return "half"
    for end, new in [("leaves", "leaf"), ("ies", "y"), ("oes", "o"), ("ches", "ch"),
                     ("shes", "sh"), ("xes", "x"), ("sses", "ss"), ("s", "")]:
        if word.endswith(end) and not word.endswith("ss"):
            return word[: -len(end)] + new
    return word


def clean_words(text):
    """Drop units, descriptors and hyphenated descriptors ("low-sodium"); keep the rest."""
    words, prev = [], ""
    for w in map(singular, text.split()):
        is_spice = w == "clove" and prev in ("ground", "whole")  # ground cloves, not garlic
        prev = w
        if not is_spice and (w in UNITS or w in DESCRIPTORS
                             or all(p in DESCRIPTORS for p in w.split("-"))):
            continue
        words.append(w)
    return " ".join(words).strip()


@lru_cache(maxsize=None)  # the same strings repeat thousands of times -> compute once
def normalize_ingredient(text):
    """Raw ingredient string -> tuple of canonical names (usually 1, sometimes 2)."""
    t = text.lower().strip()
    if t.endswith(":"):                                  # header like "For the sauce:"
        return ()
    t = re.sub(r"\(.*?\)", " ", t).replace("half and half", "half-and-half")
    # "dairy free cheese" -> "dairy-free cheese" ("free" alone is dropped as a descriptor)
    t = re.sub(rf"\b({'|'.join(FREE_WORDS)})\s+free\b", r"\1-free", t)
    name = ""
    for part in re.split(r"[,;]", t):                    # name is usually before the comma,
        part = re.sub(r"[^a-z\s-]", " ", part)           # ...but not always: use first
        part = re.sub(r"(?<!\w)-|-(?!\w)", " ", part)    # part that survives cleaning
        name = re.sub(r"^((and|or)(\s+|$))+|(\s+(and|or))+$", "", clean_words(part))  # stray and/or
        if name:
            break
    if not name:
        return ()
    if name in KEEP_TOGETHER:
        return (name,)
    parts = re.split(r" (?:and|or) ", name)              # "salt and black pepper" -> 2 names
    return tuple(ALIASES.get(p, p) for p in parts if p)


def normalize_list(ingredients):
    """All canonical names for one recipe, duplicates removed, order kept."""
    return list(dict.fromkeys(n for ing in ingredients for n in normalize_ingredient(ing)))


def has_phrase(phrase, name):
    return re.search(rf"\b{re.escape(phrase)}\b", name) is not None


@lru_cache(maxsize=None)
def matches(query, name, strict=False):
    """Does canonical `name` count as ingredient `query`?
    'garlic' ~ 'garlic powder' yes; 'butter' ~ 'peanut butter' no (NOT_A).
    strict=True for "WITH chicken": 'chicken broth' doesn't count (DERIVED).
    strict=False for "WITHOUT chicken": it does count, safer to over-exclude."""
    q = clean_words(query.lower()) or query.lower()
    if not has_phrase(q, name) or any(has_phrase(x, name) for x in NOT_A.get(q, [])):
        return False
    return not (strict and DERIVED & set(name.split(q, 1)[1].split()))


@lru_cache(maxsize=None)
def group_regex(group, status):
    """All words of one list as ONE regex, longest first: r'\\b(sour cream|cream|...)\\b'"""
    words = sorted(GROUPS[group][status], key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")


@lru_cache(maxsize=None)
def name_status(name, group):
    """'contains', 'uncertain' or 'ok' for ONE canonical name in a group (e.g. 'dairy').
    Each word found is re-checked with matches() (NOT_A exceptions) and the group's own
    exceptions ("rice noodle" is not gluten)."""
    if re.search(rf"\b({'|'.join(FREE_OF[group])})[- ]free\b", name):  # "nut-free", "gluten free"
        return "ok"
    exceptions = GROUPS[group].get("exceptions", {})

    def really(word):
        return matches(word, name) and not any(has_phrase(x, name) for x in exceptions.get(word, []))

    for status in ("contains", "uncertain"):
        if any(really(word) for word in group_regex(group, status).findall(name)):
            return status
    return "ok"


def check_group(names, group):
    """Classify a recipe's names: ('contains' | 'uncertain' | 'ok', evidence names)."""
    for status in ("contains", "uncertain"):
        hits = [n for n in names if name_status(n, group) == status]
        if hits:
            return status, hits
    return "ok", []


def count_steps(text):
    """Step = sentence ending in . ! ? followed by whitespace + capital letter.
    Pieces without letters ("1.") and website boilerplate are not steps."""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)]
    return sum(1 for p in pieces if any(c.isalpha() for c in p) and p not in BOILERPLATE)


if __name__ == "__main__":
    for text in ["3 tbsp low-sodium tamari soy sauce", "Kosher salt and freshly ground black pepper",
                 "1 (16 ounce) package frozen whipped topping (such as Cool Whip®), thawed",
                 "1/4 cup (1/2 stick) butter", "2 garlic cloves, minced", "1 cup half-and-half",
                 "4 skinless, boneless chicken breast halves", "1/2 teaspoon ground cloves"]:
        print(f"{text!r:75} -> {normalize_ingredient(text)}")

    print()
    for text, expected in [
        ("Mix flour and sugar. Bake 20 minutes.", 2),
        ("Bake at 350 degrees F. until golden. Cool.", 2),   # lowercase after "F."
        ("1. Heat the oil. Add shells.\n2. Stir well.", 3),  # "1." is not a step
        ("Add 1.5 cups water. Boil!", 2),                    # decimal is not a split
        ("Combine everything", 1),                           # no final period
        ("Watch how to make this recipe. Mix well.", 1),     # boilerplate ignored
        (";", 0),                                            # junk
    ]:
        got = count_steps(text)
        print("OK  " if got == expected else "FAIL", got, expected, repr(text))
