"""Curated value banks for AgentLife.

Invariant enforced by tests: within one attribute domain, no canonical value
matches inside another value of the same domain (word-boundary check), so the
scorer can rely on plain boundary matching without longest-match arbitration.
"""
from __future__ import annotations

# (name, gender) — the first three pairs are deliberate confusable twins.
PERSONS = [
    ("Sofia Almeida", "f"),
    ("Sofia Duarte", "f"),
    ("Marcus Chen", "m"),
    ("Marcus Webb", "m"),
    ("Elena Petrova", "f"),
    ("Elena Marsh", "f"),
    ("Ravi Kumar", "m"),
    ("Tom Eriksen", "m"),
    ("Yuki Tanaka", "f"),
    ("Amara Okafor", "f"),
    ("Lucas Ferreira", "m"),
    ("Nina Kowalski", "f"),
    ("Omar Haddad", "m"),
    ("Grace Liu", "f"),
    ("Diego Ramos", "m"),
    ("Hana Novak", "f"),
]

# "Auriga" vs "Aurora" are deliberate confusables (not substrings).
PROJECTS = [
    "Project Falcon",
    "Project Auriga",
    "Project Aurora",
    "Project Basil",
    "Project Cinder",
    "Project Dune",
    "Project Ember",
    "Project Fathom",
]

CITIES = [
    "Lisbon", "Oslo", "Kyoto", "Denver", "Marseille", "Toronto", "Auckland",
    "Zurich", "Valencia", "Krakow", "Osaka", "Tallinn", "Boulder", "Nairobi",
    "Cusco", "Galway", "Bergen", "Sapporo", "Antwerp", "Havana", "Adelaide",
    "Quito", "Tbilisi", "Fresno", "Leipzig", "Cork", "Windhoek", "Bristol",
    "Trondheim", "Matera", "Gdansk", "Boise", "Kanazawa", "Split", "Yerevan",
    "Tampere", "Luang Prabang", "Salta", "Chefchaouen", "Hobart",
]

COMPANIES = [
    "Nortex Labs", "Helioform", "Quandry Systems", "Vantalux", "Ferrowind",
    "Optiline", "Zephyr Logic", "Cobalt Harbor", "Brightmesh", "Kernelworks",
    "Solvantic", "Peregrine Data", "Mistral Forge", "Cedarline Group",
    "Ionflow", "Tesserac Analytics", "Windrose Media", "Halcyon Grid",
]

RESTAURANTS = [
    "Casa Verde", "Miso Garden", "Petit Four", "La Braise", "Golden Fern",
    "The Copper Pot", "Saffron Alley", "Juniper Table", "Oaxaca Corner",
    "Blue Heron Bistro",
]

COFFEES = [
    "flat white", "cold brew", "oat latte", "double espresso",
    "matcha latte", "cortado",
]

WIFI_PASSWORDS = [
    "maple-toast-42", "violet-canyon-7", "copper-flamingo-19",
    "quiet-harbor-88", "neon-cactus-3", "silver-otter-61",
]

DIETARY = ["vegetarian", "vegan", "gluten-free", "pescatarian"]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]

PROJECT_STATUSES = ["active", "paused", "completed", "cancelled"]

NOISE_SENTENCES = [
    "Watched a documentary about deep sea creatures tonight.",
    "The weather was surprisingly nice for a walk this afternoon.",
    "Tried a new podcast about urban planning during the commute.",
    "Spent an hour reorganizing the bookshelf, very satisfying.",
    "The neighbor's dog kept barking most of the morning.",
    "Finally finished that novel that had been sitting on the desk.",
    "Traffic was terrible today, took forever to get across town.",
    "Made soup from scratch, it turned out better than expected.",
    "The gym was packed, could barely get on any machine.",
    "Caught a beautiful sunset from the balcony this evening.",
    "Power went out for twenty minutes during dinner.",
    "Found a great secondhand record store downtown.",
    "The elevator in the building is being repaired again.",
    "Binge-watched a cooking competition show way too late.",
    "A street musician played violin near the market, lovely.",
    "Repotted the plants on the windowsill this weekend.",
    "The bakery on the corner had a line out the door.",
    "Went for a long bike ride along the river path.",
    "Cleaned out the garage, found boxes from three moves ago.",
    "The new phone update changed all the settings around.",
]

UNKNOWN_MARKERS = [
    "unknown", "not known", "not mentioned", "never mentioned",
    "i don't know", "don't know", "no information", "not stated",
    "not provided", "never provided", "not sure", "no idea",
    "cannot recall", "can't recall", "not recorded",
]


def accepted_variants(attribute: str, value: str) -> list:
    """All answer strings accepted as correct for a canonical value."""
    if attribute in ("birthday", "deadline"):
        # canonical form: "June 5"
        month, day = value.rsplit(" ", 1)
        return [
            f"{month} {day}",
            f"{month} {day}th" if not day.endswith(("1", "2", "3")) or day in ("11", "12", "13")
            else f"{month} {day}{'st' if day.endswith('1') else 'nd' if day.endswith('2') else 'rd'}",
            f"{day} {month}",
            f"{day}th of {month}" if not day.endswith(("1", "2", "3")) or day in ("11", "12", "13")
            else f"{day}{'st' if day.endswith('1') else 'nd' if day.endswith('2') else 'rd'} of {month}",
        ]
    if attribute == "status" and value == "cancelled":
        return ["cancelled", "canceled"]
    return [value]
