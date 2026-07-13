"""Verbalization templates.

Every template MUST verbalize the canonical value(s) it sets verbatim —
the generator asserts this, so the answer is always extractable from text.
Placeholders: {name} {value} {old} {sub} {poss} {client} {lead}
"""
from __future__ import annotations

PRONOUNS = {"m": ("he", "his"), "f": ("she", "her"), "": ("they", "their")}

INTRODUCE_PERSON = [
    "Met {name} today at a friend's dinner. {sub_cap} works at {employer} and lives in {lives_in}.",
    "New contact: {name}. {sub_cap} is at {employer} these days and is based in {lives_in}.",
    "Had coffee with {name}, who works at {employer} and lives in {lives_in}.",
    "Got introduced to {name} — works at {employer}, lives in {lives_in}.",
]

INTRODUCE_PROJECT = [
    "Kicked off {name} today. The client is {client} and {lead} is leading it. The project is now active.",
    "We started {name} this week — {client} is the client, {lead} is the lead, and it is officially active.",
    "{name} launched: client {client}, led by {lead}. Status: active.",
]

STATE = {
    ("person", "birthday"): [
        "{name} mentioned that {poss} birthday is {value}.",
        "Noted: {name}'s birthday is {value}.",
        "Turns out {name} was born on {value}.",
    ],
    ("person", "hometown"): [
        "{name} grew up in {value}, apparently.",
        "{name} told me {poss} hometown is {value}.",
        "Fun fact from lunch: {name} is originally from {value}.",
    ],
    ("person", "favorite_restaurant"): [
        "{name}'s favorite restaurant is {value}.",
        "{name} keeps recommending {value} — {poss} favorite spot.",
        "If you ever book dinner with {name}, {poss} favorite place is {value}.",
    ],
    ("person", "works_on"): [
        "{name} is now working on {value}.",
        "{name} joined the team on {value}.",
        "Heard that {name} has been assigned to {value}.",
    ],
    ("project", "deadline"): [
        "The deadline for {name} is {value}.",
        "{name} is due on {value}.",
        "Reminder to self: {name} must ship by {value}.",
    ],
    ("project", "budget"): [
        "The budget for {name} was set at {value}.",
        "{name} got a budget of {value} approved.",
        "Finance confirmed {value} for {name}.",
    ],
    ("user", "parking_spot"): [
        "I parked the car at spot {value} today.",
        "Left the car in spot {value} this morning.",
        "Car is at spot {value} for now.",
    ],
    ("user", "desk_booking"): [
        "Booked {value} at the coworking space for this week.",
        "I'm sitting at {value} this week.",
        "Got {value} reserved at the office.",
    ],
    ("user", "gym_day"): [
        "My gym day is {value} from now on.",
        "I settled on {value} as my gym day.",
        "Locked in {value} for the gym every week.",
    ],
    ("user", "dietary_restriction"): [
        "Quick note: I'm {value}, so keep that in mind for recipes.",
        "I've decided to go {value} — please remember that.",
        "For any food suggestions: I'm {value} now.",
    ],
    ("user", "wifi_password"): [
        "The home wifi password is {value}.",
        "I changed the wifi password to {value}.",
        "New wifi password: {value}.",
    ],
    ("user", "favorite_coffee"): [
        "My usual coffee order is a {value}.",
        "These days my go-to coffee is a {value}.",
        "If you ever order for me: {value}.",
    ],
}

UPDATE = {
    ("person", "employer"): [
        "Big news: {name} left {old} and joined {value}.",
        "{name} changed jobs — no longer at {old}, now at {value}.",
        "{name} told me {sub} moved from {old} to {value}.",
    ],
    ("person", "lives_in"): [
        "{name} moved from {old} to {value}.",
        "{name} relocated — {sub} now lives in {value}, not {old} anymore.",
        "Update: {name} just moved to {value} (was in {old}).",
    ],
    ("person", "favorite_restaurant"): [
        "{name} has a new favorite restaurant: {value} (dethroned {old}).",
        "{name} says {value} is {poss} new favorite, replacing {old}.",
    ],
    ("person", "works_on"): [
        "{name} was moved off {old} and onto {value}.",
        "Reorg: {name} switched from {old} to {value}.",
    ],
    ("project", "lead"): [
        "Leadership change on {name}: {value} is taking over from {old}.",
        "{old} stepped down; {value} now leads {name}.",
        "{name} has a new lead — {value} replaces {old}.",
    ],
    ("project", "deadline"): [
        "The deadline for {name} slipped from {old} to {value}.",
        "{name} was rescheduled: new deadline {value} (was {old}).",
        "Heads up: {name} is now due {value} instead of {old}.",
    ],
    ("project", "budget"): [
        "The budget for {name} was revised from {old} to {value}.",
        "{name}'s budget changed: {value} now, up from {old}.",
    ],
    ("project", "status"): [
        "{name} status change: it is now {value} (was {old}).",
        "Update on {name}: the project moved from {old} to {value}.",
    ],
    ("user", "parking_spot"): [
        "Moved the car — it's at spot {value} now.",
        "I parked at spot {value} this time.",
        "Car update: spot {value}.",
    ],
    ("user", "desk_booking"): [
        "Switched desks — I'm at {value} now, not {old}.",
        "New week, new spot: {value} (was at {old}).",
    ],
    ("user", "gym_day"): [
        "I switched my gym day from {old} to {value}.",
        "Gym day is now {value} instead of {old}.",
    ],
    ("user", "wifi_password"): [
        "I rotated the wifi password: it's {value} now, {old} won't work.",
        "New wifi password is {value} (the old {old} is dead).",
    ],
    ("user", "favorite_coffee"): [
        "I'm off the {old} phase — my coffee order is a {value} now.",
        "Switched my usual coffee from {old} to {value}.",
    ],
}

REVOKE = {
    ("user", "dietary_restriction"): [
        "By the way, I'm not {old} anymore — no dietary restrictions at all now.",
        "You can stop filtering recipes: I gave up being {old}, no restrictions now.",
    ],
    ("project", "status"): [
        "Bad news: {name} was cancelled today.",
        "{name} is dead — the client pulled the plug, project cancelled.",
        "They killed {name}. Officially cancelled.",
    ],
}

MENTION = {
    ("person", "employer"): [
        "Bumped into {name} near the {value} office.",
        "{name} was complaining about meetings at {value} again.",
    ],
    ("person", "lives_in"): [
        "{name} invited me to visit {value} sometime.",
        "Postcard from {name} — {value} looks lovely.",
    ],
    ("person", "works_on"): [
        "{name} said {value} is keeping {poss} team busy.",
        "Long call with {name} about {value} today.",
    ],
    ("project", "lead"): [
        "{lead} ran the {name} standup this morning.",
        "Quick sync with {lead} about {name}.",
    ],
    ("user", "gym_day"): [
        "Went to the gym as usual — {value} really is the quietest day.",
    ],
    ("user", "favorite_coffee"): [
        "Started the day with my usual {value}.",
    ],
}

ANSWER_HINTS = {
    "birthday": "Answer with a date like 'June 5'.",
    "hometown": "Answer with a city name.",
    "lives_in": "Answer with a city name.",
    "employer": "Answer with a company name.",
    "favorite_restaurant": "Answer with the restaurant name.",
    "works_on": "Answer with the project name.",
    "lead": "Answer with the person's full name.",
    "deadline": "Answer with a date like 'June 5'.",
    "budget": "Answer with the amount, like '$120k'.",
    "status": "Answer with one word: active, paused, completed, or cancelled.",
    "client": "Answer with the company name.",
    "parking_spot": "Answer with the spot code, like 'B47'.",
    "desk_booking": "Answer with the desk, like 'Desk 14'.",
    "gym_day": "Answer with a day of the week.",
    "dietary_restriction": "Answer with the restriction, or 'none'.",
    "wifi_password": "Answer with the password.",
    "favorite_coffee": "Answer with the drink name.",
}

UNKNOWN_SUFFIX = "If this was never mentioned, answer 'unknown'."
