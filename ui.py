"""Web chat interface.  Run:  streamlit run ui.py   (opens http://localhost:8501)

Streamlit reruns this whole file on every click/message. So:
- the dataset is loaded ONCE and shared (st.cache_resource),
- each browser tab keeps its own Session + chat history (st.session_state).
Answers are drawn from session.view (structured results): recipe cards, badges, metrics, a chart.
"""

import time

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loader import load_recipes
from src.llm import Intent
from src.matcher import have_name
from src.normalize import matches
from src.search import RecipeSearch
from src.session import Session, describe

st.set_page_config(page_title="Recipe Chatbot", page_icon="🍳", layout="wide")
BAR_COLOR = "#2a78d6"  # one series -> one color (dataviz reference palette, slot 1)
EXAMPLES = {
    "🔎 Search": ["Find a penne pasta recipe", "Find a banana bread recipe"],
    "🥕 What can I cook?": ["I have eggs, onions, tomatoes and rice. No peanuts, at most 6 steps.",
                           "What can I make with 4 eggs and 1 cup of rice?"],
    "📊 Statistics": ["How many recipes contain both garlic and onion?",
                      "Most common ingredients in gluten-free recipes?"],
}
FOLLOW_UPS = {
    "search": ["I cannot have dairy", "Make it gluten-free", "At most 5 steps", "Show me the first one"],
    "match": ["I'd rather have fewer steps", "Fewest extra ingredients first", "Show me the first one"],
    "stats": ["How many of these are there?", "Which five have the fewest steps?"],
    "recipe": ["Show me the second one", "How many calories does it have?"],
}


@st.cache_resource(show_spinner="Loading and indexing 122,917 recipes (about 30 seconds, once)...")
def get_searcher():
    return RecipeSearch(load_recipes())


def new_conversation():
    st.session_state.session = Session(get_searcher())
    st.session_state.messages = []  # [{"role", "content", "view", "intent", "seconds"}]


if "session" not in st.session_state:
    new_conversation()
session: Session = st.session_state.session
searcher = get_searcher()


def ask(text):
    """Queue a message (from a button) to be answered on this rerun."""
    st.session_state.pending = text


def remove_filter(item, label):
    """Sidebar chip: remove a filter directly, no LLM call (instant)."""
    start = time.time()
    answer = session.respond(Intent(task="search", remove=[item]))
    st.session_state.messages += [
        {"role": "user", "content": f"✕ remove filter: {label}"},
        {"role": "assistant", "content": answer, "view": session.view, "seconds": time.time() - start}]


def badges(items, color):
    return " ".join(f":{color}-badge[{x}]" for x in items) if items else "_none_"


def recipe_details(recipe_id):
    r = searcher.get(recipe_id)
    left, right = st.columns([2, 3])
    left.markdown("**Ingredients**\n" + "\n".join(f"- {i}" for i in r["ingredients"]))
    right.markdown("**Instructions**")
    right.write(r["instructions"])


# ---------- answer renderers (one per kind of result) ----------
def show_search(view):
    d = view["data"]
    st.caption(f"Active filters: **{view['filters']}** · ranked by {d['ranking']}")
    if d["total_matches"] == 0:
        st.warning("No recipe in the dataset matches all of these filters. Remove one in the sidebar.")
    else:
        st.markdown(f"**{d['total_matches']:,} matching recipes.** Top {len(d['results'])}:")
    for i, x in enumerate(d["results"], 1):
        with st.expander(f"**{i}. {x['title']}** · {x['n_steps']} steps", expanded=(i == 1)):
            st.caption(f"ID `{x['recipe_id']}`")
            for ing, names in x["matched"].items():
                st.markdown(f"{ing} → {badges(names, 'green')}")
            recipe_details(x["recipe_id"])
    if d["uncertain_count"]:
        examples = "\n".join(f"- {x['title']}: {', '.join(x['flags'])}" for x in d["uncertain"])
        st.warning(f"**{d['uncertain_count']} recipes not shown** because they *may* break your "
                   f"restrictions:\n{examples}", icon="⚠️")


def show_match(view):
    d = view["data"]
    st.caption(f"You have: **{', '.join(d['have'])}** · hard constraints: **{view['filters']}** · "
               f"ranking: {d['ranking']} · pantry assumed: salt, black pepper, water")
    if not d["results"]:
        st.error("No recipe meets all hard constraints and uses any of your ingredients.")
        for x in d["alternatives"]:
            st.warning(f"**{x['title']}** `{x['recipe_id']}` uses {', '.join(x['used'])}  \n"
                       f"❌ violates: {'; '.join(x['violates'])}")
        return
    st.markdown(f"**{d['eligible_count']:,} eligible recipes** use at least one of your ingredients.")
    for i, x in enumerate(d["results"], 1):
        used, total = len(x["used"]), len(d["have"])
        with st.expander(f"**{i}. {x['title']}** · uses {used}/{total} · needs {len(x['missing'])} more · "
                         f"{x['n_steps']} steps", expanded=(i == 1)):
            st.progress(used / total if total else 0, text=f"uses {used} of your {total} ingredients")
            st.markdown(f"**Uses:** {badges(x['used'], 'green')}  \n"
                        f"**Not used:** {badges(x['unused'], 'gray')}  \n"
                        f"**You'd also need:** {badges(x['missing'], 'orange')}")
            for name, note in x["quantities"].items():
                st.markdown(f"⚖️ {name}: {note}")
            if x["reason"]:
                st.caption(f"Rank: {x['reason']}")
            recipe_details(x["recipe_id"])
    if not any(x["quantities"] for x in d["results"]):
        st.caption("Matching is by ingredient name; quantity sufficiency was not verified.")


def show_stats(view):
    d, stat = view["data"], view["stat"]
    if stat == "count":
        a, b, c = st.columns(3)
        a.metric("Matching recipes", f"{d['count']:,}")
        b.metric("Share of dataset", f"{d['percent']}%")
        c.metric("Dataset size", f"{d['dataset_size']:,}")
        for ing, names in d["matched_names"].items():
            st.markdown(f"**{ing}** counted as: {badges(names, 'blue')} …")
    elif stat == "common_ingredients":
        df = pd.DataFrame(d["top"], columns=["ingredient", "recipes", "percent"])
        chart = alt.Chart(df, title="Most common ingredients (% of recipes analyzed)").mark_bar(
            color=BAR_COLOR, cornerRadiusEnd=4, size=16).encode(
            x=alt.X("percent:Q", title="% of recipes"),
            y=alt.Y("ingredient:N", sort="-x", title=None),
            tooltip=["ingredient", alt.Tooltip("recipes:Q", format=","), alt.Tooltip("percent:Q", format=".1f")],
        ).properties(height=34 * len(df) + 40)
        st.altair_chart(chart, width="stretch")
        with st.expander("Table view"):
            st.dataframe(df, hide_index=True, width="stretch")
    elif stat == "ingredient_count":
        if d["mean"] is None:
            st.warning("No recipe matches these filters, so there is no average.")
        else:
            for col, (label, key) in zip(st.columns(4), [("Mean", "mean"), ("Median", "median"),
                                                         ("Min", "min"), ("Max", "max")]):
                col.metric(f"{label} ingredients", d[key])
    elif stat == "rank_steps":
        df = pd.DataFrame(d["results"], columns=["recipe_id", "title", "steps"])
        df.insert(0, "#", range(1, len(df) + 1))
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"{d['ties_at_cutoff']} recipes have the same step count as the last one shown.")
    st.caption(f"Filters: **{view['filters']}** · recipes analyzed: {d['recipes_analyzed']:,} "
               f"(+{d['uncertain_excluded']:,} uncertain excluded) · calculation: {d['calculation']}")


def show_recipe(view):
    r = view["data"]
    st.subheader(r["title"])
    st.caption(f"ID `{r['recipe_id']}` · {r['n_steps']} steps")
    recipe_details(r["recipe_id"])


RENDER = {"search": show_search, "match": show_match, "stats": show_stats, "recipe": show_recipe}


def show_message(msg):
    with st.chat_message(msg["role"]):
        view = msg.get("view")
        if view:
            RENDER[view["kind"]](view)
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            footer = f"⏱ {msg['seconds']:.1f} s" if msg.get("seconds") else ""
            st.caption(footer)
            if msg.get("intent"):
                with st.expander("What I understood"):
                    st.json(msg["intent"])


# ---------- sidebar: the conversation's context (editable) ----------
with st.sidebar:
    st.header("Conversation context")
    mode = "ingredient matching (Task 2)" if session.mode == "match" else "recipe search (Task 1)"
    st.markdown(f"**Mode:** {mode}")
    f = session.filters
    chips = ([(g, f"no {g}") for g in f["exclude_groups"]]
             + [(i, f"without {i}") for i in f["exclude_ingredients"]]
             + [(i, f"with {i}") for i in f["include"]]
             + ([("max_steps", f"≤ {f['max_steps']} steps")] if f["max_steps"] is not None else []))
    if f["query"] and session.mode == "search":  # matching ignores the dish name
        st.markdown(f"**Dish:** {f['query']}")
    st.markdown("**Active filters:**" + ("" if chips else " none"))
    for item, label in chips:
        st.button(f"✕ {label}", key=f"chip-{item}", on_click=remove_filter, args=(item, label),
                  help="Remove this filter (instant, no LLM call)")
    if session.mode == "match":
        excluded = f["exclude_ingredients"]  # kept in `have`, but not used while excluded
        items = [f"~~{h}~~ (excluded)" if any(matches(e, have_name(h)) for e in excluded) else h
                 for h in session.have]
        st.markdown(f"**You have:** {', '.join(items) or 'nothing yet'}")
        st.markdown(f"**Ranking priorities:** {', '.join(session.prefer) or 'default'}")
    times = [m["seconds"] for m in st.session_state.messages if m.get("seconds")]
    if times:
        st.caption(f"Average answer time: {sum(times) / len(times):.1f} s over {len(times)} answers")
    st.divider()
    st.button("🗑 New conversation", on_click=new_conversation, width="stretch")

# ---------- main: chat ----------
st.title("🍳 Recipe Chatbot")
st.caption(f"Grounded in {len(searcher.df):,} recipes from the Recipe Box dataset. The LLM only reads "
           "your message; every recipe and number is computed from the data.")

if not st.session_state.messages:
    st.markdown("#### Try one of these")
    for col, (group, prompts) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        col.markdown(f"**{group}**")
        for p in prompts:
            col.button(p, key=f"ex-{p}", on_click=ask, args=(p,), width="stretch")

for msg in st.session_state.messages:
    show_message(msg)

# one-click follow-ups for the latest answer
last = st.session_state.messages[-1] if st.session_state.messages else None
if last and last["role"] == "assistant" and last.get("view"):
    st.caption("Follow up:")
    options = FOLLOW_UPS[last["view"]["kind"]]
    for col, p in zip(st.columns(len(options)), options):
        col.button(p, key=f"fu-{len(st.session_state.messages)}-{p}", on_click=ask, args=(p,),
                   width="stretch")

prompt = st.chat_input("Ask about recipes, ingredients you have, or statistics...") \
    or st.session_state.pop("pending", None)
if prompt:
    show_message({"role": "user", "content": prompt})
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        start = time.time()
        answer = session.handle(prompt)
        seconds = time.time() - start
    intent = session.last_intent.model_dump(exclude_unset=True) if session.last_intent else None
    st.session_state.messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer, "view": session.view, "intent": intent, "seconds": seconds}]
    st.rerun()  # redraw: new messages, updated sidebar, new follow-up buttons
