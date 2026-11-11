"""Command-line recipe chatbot.  Run:  python app.py        (add --debug to see parsed intents)"""

import sys

from src.data_loader import load_recipes
from src.search import RecipeSearch
from src.session import Session


def main():
    print("Loading and indexing recipes (about 30 seconds)...")
    session = Session(RecipeSearch(load_recipes()), debug="--debug" in sys.argv)
    print("Ready. Ask about recipes. Type 'quit' to exit.")
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in ("quit", "exit"):
            break
        if message:
            print("\nBot:", session.handle(message))


if __name__ == "__main__":
    main()
