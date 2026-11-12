"""Word lists used by normalize.py. Data only, no logic: edit these to change behavior."""

# Removed from ingredient strings: units/containers...
UNITS = set("""
cup c tablespoon tbsp tbs tbl teaspoon tsp ounce oz pound lb gram g kg kilogram ml l liter
litre lbs ozs kgs tbsps tsps quart qt pint pt gallon fluid fl package pkg packet can jar bottle box bag container
carton envelope stick clove head bunch sprig slice piece pinch dash drop handful stalk rib
sheet strip loaf inch cube scoop splash whole half quarter dozen of a an about plus more
additional each to taste optional x
""".split())

# ...and words describing size, state or preparation
DESCRIPTORS = set("""
fresh freshly large small medium extra big jumbo thin thick chopped minced diced sliced grated
shredded crushed finely coarsely roughly thinly cut peeled seeded trimmed softened melted
cooked uncooked raw beaten lightly well room temperature cold warm chilled thawed drained
rinsed packed firmly loosely sifted divided plain pure kosher fine coarse low sodium fat free
reduced lean boneless skinless halved quartered cubed torn julienned mashed pitted toasted
dried dry frozen canned prepared good quality organic ripe very such as into for garnish
serving needed desired if at in with the inch-thick bite-size bite-sized store-bought homemade lukewarm
level heaping scant generous approximately unsalted salted
""".split())

# Words ending in "s" that are not plurals
NOT_PLURAL = set("""
molasses hummus asparagus couscous swiss grits lemongrass watercress brussels citrus
hibiscus bitters schnapps less
""".split())

# Contain "and" but are ONE product (not split into two ingredients)
KEEP_TOGETHER = {"macaroni and cheese", "sweet and sour sauce", "salt and vinegar"}

# Same product, different wording -> one name
ALIASES = {
    "all purpose flour": "all-purpose flour",
    "white sugar": "sugar", "granulated sugar": "sugar",
    "confectioner sugar": "powdered sugar", "icing sugar": "powdered sugar",
    "scallion": "green onion",
    "pepper": "black pepper",  # mostly from "salt and pepper" in this dataset
    "ground black pepper": "black pepper", "ground pepper": "black pepper",
    "cracked black pepper": "black pepper",
    "heavy whipping cream": "heavy cream", "whipping cream": "heavy cream",
    "parmesan cheese": "parmesan", "cheddar cheese": "cheddar",
    "mozzarella cheese": "mozzarella", "feta cheese": "feta",
    "extra-virgin olive oil": "olive oil", "virgin olive oil": "olive oil",
    "juice lemon": "lemon juice", "juice lime": "lime juice", "juice orange": "orange juice",
    "flat-leaf parsley": "parsley", "italian parsley": "parsley",
    "bay": "bay leaf",
}
for herb in "basil parsley cilantro mint thyme sage oregano tarragon rosemary dill coriander".split():
    ALIASES[f"{herb} leaf"] = herb  # "basil leaves" -> "basil"

# Contain the word but are NOT that ingredient: "peanut butter" is not "butter"
NOT_A = {
    "butter": ["peanut butter", "cocoa butter", "almond butter", "apple butter",
               "cashew butter", "nut butter", "butternut", "butterscotch", "buttermilk"],
    "milk": ["coconut milk", "almond milk", "soy milk", "rice milk", "oat milk", "buttermilk"],
    "cream": ["cream of tartar", "coconut cream", "cream cheese", "sour cream", "ice cream"],
    "egg": ["eggplant"],
    "onion": ["green onion", "spring onion"],  # different plant part
    "pepper": ["bell pepper", "red pepper flake"],
}

# Products MADE FROM an ingredient: "chicken broth" is not chicken (used for "recipes WITH x")
DERIVED = set("""
broth stock bouillon soup juice cocktail gravy base seasoning flavoring drippings fat liqueur
schnapps cereal vinegar wine flour noodle paper syrup milk extract
""".split())

# Restriction groups. "uncertain" = may or may not contain it -> flag it, never assume safe.
# Exceptions (coconut milk, peanut butter...) come from NOT_A above, plus a group's own "exceptions".
GROUPS = {
    "dairy": {
        "contains": ["milk", "butter", "cream", "cheese", "yogurt", "buttermilk", "ghee", "whey",
                     "parmesan", "mozzarella", "cheddar", "ricotta", "mascarpone", "half-and-half",
                     "creme fraiche", "sour cream", "cream cheese", "ice cream", "feta", "brie",
                     "gruyere", "provolone", "goat cheese", "monterey jack", "romano", "custard"],
        "uncertain": ["whipped topping", "margarine", "chocolate", "creamer", "pesto", "cool whip"],
    },
    "nuts": {
        "contains": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia",
                     "peanut", "pine nut", "brazil nut", "nut"],
        "uncertain": ["praline", "marzipan", "nougat", "granola"],
    },
    "gluten": {
        "contains": ["wheat", "flour", "bread", "bread crumb", "panko", "crouton", "pasta",
                     "noodle", "spaghetti", "penne", "macaroni", "linguine", "fettuccine",
                     "lasagna", "rigatoni", "ziti", "orzo", "tortellini", "ravioli", "couscous",
                     "fusilli", "linguini", "capellini", "angel hair", "orecchiette", "farfalle",
                     "rotini", "bucatini", "vermicelli", "pappardelle", "tagliatelle", "gnocchi",
                     "cavatappi", "ditalini", "manicotti", "cannelloni", "udon", "ramen", "lo mein",
                     "barley", "rye", "semolina", "bulgur", "farro", "spelt", "seitan", "malt",
                     "beer", "cracker", "cookie", "biscuit", "pastry", "pie crust", "phyllo",
                     "croissant", "bagel", "pita", "bun", "pretzel", "wonton wrapper",
                     "cake mix", "brownie mix", "graham cracker", "matzo", "wheat germ"],
        "uncertain": ["soy sauce", "oat", "teriyaki sauce", "hoisin sauce", "worcestershire sauce",
                      "soba", "tortilla", "cream of mushroom soup", "cream chicken soup",
                      "condensed", "gravy", "bouillon", "seasoning mix", "cereal", "granola",
                      "breading", "stuffing"],
        # gluten-free products that contain a gluten word. ONLY for the gluten check:
        # "no flour" (an ingredient the user named) still removes almond flour.
        "exceptions": {
            "flour": ["almond flour", "rice flour", "coconut flour", "corn flour", "potato flour",
                      "chickpea flour", "tapioca flour", "buckwheat flour", "cassava flour",
                      "oat flour"],
            "pasta": ["pasta sauce", "rice pasta"],
            "noodle": ["rice noodle", "glass noodle", "cellophane noodle", "bean thread noodle",
                       "shirataki noodle"],
            "cracker": ["rice cracker"],
            "vermicelli": ["rice vermicelli"],
            "spaghetti": ["spaghetti squash"],  # a vegetable
        },
    },
}

# "<word>-free" / "<word> free" means safe for that group: "nut-free", "dairy free", "gluten-free"
FREE_OF = {"dairy": ["dairy", "lactose", "milk"], "nuts": ["nut", "peanut"],
           "gluten": ["gluten", "wheat"]}

# Task 2 matching -------------------------------------------------------------------------
# Assumed to be in every kitchen: not counted as "missing" (listed separately in answers)
PANTRY = {"salt", "black pepper", "water"}

# Having the fresh ingredient does NOT give you these products: tomato -> not "tomato sauce"
PRODUCT_FORMS = set("sauce paste puree ketchup powder salt soup mix".split())

# Units we can compare quantities in (other units -> "quantity not verified")
MEASURES = {"cup": "cup", "tablespoon": "tbsp", "tbsp": "tbsp", "tbs": "tbsp", "teaspoon": "tsp",
            "tsp": "tsp", "ounce": "oz", "oz": "oz", "pound": "lb", "lb": "lb", "gram": "g",
            "g": "g", "kg": "kg", "ml": "ml", "liter": "l", "l": "l", "clove": "clove",
            "can": "can", "stick": "stick", "slice": "slice",
            "lbs": "lb", "ozs": "oz", "kgs": "kg", "tbsps": "tbsp", "tsps": "tsp"}  # short plurals

# Website text that is not a cooking step (in 8,387 Food Network recipes)
BOILERPLATE = {"Watch how to make this recipe."}

# Task 3 substitution --------------------------------------------------------------------
# CANDIDATE substitutes per ingredient, with the ingredient's role and a usual amount.
# This is general cooking knowledge (labeled as such in answers). The DATASET supplies the
# evidence: how often similar recipes without the ingredient use each candidate.
SUBSTITUTES = {
    "egg": {"role": "binding, moisture and some lift in baking",
            "options": {"applesauce": "1/4 cup per egg", "banana": "1/4 cup mashed per egg",
                        "yogurt": "1/4 cup per egg", "buttermilk": "1/4 cup per egg",
                        "flaxseed": "1 tbsp ground + 3 tbsp water per egg",
                        "tofu": "1/4 cup silken, blended, per egg",
                        "vegetable oil": "3 tbsp per egg (moisture only, no binding)"}},
    "cream": {"role": "richness, body and thickening of sauces",
              "options": {"coconut milk": "same amount (full-fat)", "milk": "same amount + 1 tbsp butter per cup",
                          "chicken broth": "same amount, reduce longer (lighter sauce)",
                          "cream cheese": "half the amount, whisked in", "olive oil": "2-3 tbsp + starchy pasta water",
                          "white bean": "1 cup pureed per cup of cream", "cashew": "1 cup soaked + blended per cup"}},
    "butter": {"role": "fat, flavor and tenderness",
               "options": {"vegetable oil": "3/4 of the amount", "olive oil": "3/4 of the amount",
                           "coconut oil": "same amount", "margarine": "same amount",
                           "shortening": "same amount", "applesauce": "half the amount (baking only)"}},
    "milk": {"role": "liquid, mild richness",
             "options": {"almond milk": "same amount", "soy milk": "same amount", "oat milk": "same amount",
                         "coconut milk": "same amount", "rice milk": "same amount", "water": "same amount (less rich)"}},
    "sour cream": {"role": "tang, moisture, creaminess",
                   "options": {"yogurt": "same amount", "cream cheese": "same amount, thinned",
                               "coconut cream": "same amount + squeeze of lemon", "mayonnaise": "same amount"}},
    "flour": {"role": "structure (gluten) or thickening",
              "options": {"rice flour": "same amount", "almond flour": "same amount (denser)",
                          "oat flour": "1 1/3 x the amount", "cornstarch": "half the amount (thickening only)",
                          "buckwheat flour": "same amount"}},
}

# Extra ingredients a conversion above needs ("milk: + 1 tbsp butter per cup"). The rewrite may
# add these (water is a pantry staple, so it needs no entry), and they must pass the restrictions too.
EXTRAS = {"milk": ["butter"], "coconut cream": ["lemon"]}

# In instruction TEXT, the word used as a verb, not the ingredient: "cream the butter and sugar"
VERB_USES = {"cream": r"cream(?=\s+(?:the|together|butter|shortening|margarine)\b)"}
