"""Zero-shot extraction of the annotated indicators of data/exercises.csv (French math exercises).

One constrained call per exercise returns, as JSON:
- subjects: at most two main actors, each with the words copied from the text (mention), the
  gender of that wording, the origin of the name (only for first names) and its traits;
- pluriel: how groups of people are written;
- topic: a short free-text description of the situation of the exercise;
- monnaie / unites: currencies and cultural units that appear.
The label sets are general (they do not list the answers of this dataset), except that the gold
annotations are mapped onto them in GOLD_* and gold_origin() below.

Free-text values are scored by a yes/no judge from another model family (judge(), P(yes)): subjects
and traits item by item (SUBJECT_PROMPT, TRAIT_PROMPT), the topic against both annotators
(TOPIC_PROMPT).
"""
import json

GENRES = ["masculin", "féminin", "féminin et masculin", "neutre (objet)"]
# Leaves of the Wikipedia name-ethnicity taxonomy (Ambekar et al., KDD 2009), as labelled in ethnicolr,
# with their top-level group; the gold only supports the top level (see gold_origin()).
NOT_A_NAME = "pas un prénom"
TAXONOMY = {
    "British": "GreaterEuropean", "EastEuropean": "GreaterEuropean", "Jewish": "GreaterEuropean",
    "French": "GreaterEuropean", "Germanic": "GreaterEuropean", "Hispanic": "GreaterEuropean",
    "Italian": "GreaterEuropean", "Nordic": "GreaterEuropean",
    "Africans": "GreaterAfrican", "Muslim": "GreaterAfrican",
    "IndianSubContinent": "Asian", "EastAsian": "Asian", "Japanese": "Asian",
}
ORIGINS = [NOT_A_NAME, *TAXONOMY]
PLURIELS = ["aucun", "inclusif", "masculin", "objets"]
MONNAIES = ["euro", "franc suisse", "dollar", "livre sterling", "autre"]
UNITES = ["système métrique", "notes sur 20", "degrés Celsius"]

GOLD_MONNAIE = {"euro": "euro", "francs": "franc suisse", "dollars": "dollar"}
GOLD_UNITES = {"metric": "système métrique", "g": "système métrique", "notes": "notes sur 20", "celsius": "degrés Celsius"}


def _text(n):
    return {"type": "string", "maxLength": n}


SCHEMA = {
    "type": "object",
    "properties": {
        "subjects": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "mention": _text(80),
                    "genre": {"type": "string", "enum": GENRES},
                    "origine": {"type": "string", "enum": ORIGINS},
                    "traits": {"type": "array", "maxItems": 4, "items": _text(80)},
                },
                "required": ["mention", "genre", "origine", "traits"],
                "additionalProperties": False,
            },
        },
        "pluriel": {"type": "string", "enum": PLURIELS},
        "topic": _text(80),
        "monnaie": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": MONNAIES}},
        "unites": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": UNITES}},
    },
    "required": ["subjects", "pluriel", "topic", "monnaie", "unites"],
    "additionalProperties": False,
}

PROMPT = """You analyse a math exercise written in French for secondary-school students, to find elements that could carry stereotypes. Read the exercise and fill in the fields below. Write every free-text value in French.

- subjects: the main actors of the exercise (people, or groups, institutions or objects that act), at most two, in order of appearance. Leave the list empty if nobody and nothing acts.
  - mention: the words of the exercise that name the actor, copied exactly as written (e.g. a first name, "un élève", "une association").
  - genre: the gender conveyed by how the actor is written: "masculin", "féminin", "féminin et masculin" (inclusive or epicene wording that covers both, e.g. "un·e étudiant·e", "une personne") or "neutre (objet)" (an institution, a group or a thing).
  - origine: only when the mention is a first name, the origin that name is most associated with, as one of the classes of the Wikipedia name-ethnicity taxonomy: "British", "EastEuropean", "Jewish", "French", "Germanic", "Hispanic", "Italian", "Nordic", "Africans", "Muslim" (names from the Arabic, Persian or Turkic world), "IndianSubContinent", "EastAsian", "Japanese"; otherwise "pas un prénom".
  - traits: characteristics the exercise attributes to this actor, in a few words each: qualities, abilities, results or comparisons with the other actor (e.g. who earns, saves or scores more or less). Empty list if none.
- pluriel: how groups of people are written in the exercise: "inclusif" (epicene nouns or inclusive writing, e.g. "les élèves", "les membres", "ami·e·s"), "masculin" (masculine plural used for everyone, e.g. "les joueurs", "les clients", "ils"), "objets" (the plural actors are things), or "aucun" (no group of people).
- topic: a short noun phrase (at most 8 words) naming the situation or theme of the exercise; if it has no real-world context, name the mathematical object it is about.
- monnaie: the currencies that appear (empty list if none).
- unites: the cultural units that appear: metric units of length, mass or volume ("système métrique"), grades out of 20 ("notes sur 20"), temperatures in Celsius ("degrés Celsius"). Empty list if none.

Answer with a JSON object only.

Exercise:
{exercise}"""


def predict(exercise, client):
    raw, _ = client.chat(
        [{"role": "user", "content": PROMPT.replace("{exercise}", exercise)}],
        800,
        structured={"json": SCHEMA, "disable_any_whitespace": True},
    )
    pred = json.loads(raw)
    pred["monnaie"] = list(dict.fromkeys(pred["monnaie"]))
    pred["unites"] = list(dict.fromkeys(pred["unites"]))
    return pred, raw


SUBJECT_PROMPT = """Exercise:
{exercise}

In this exercise, do the following two phrases refer to the same actor (the same person, group, institution or object)?
- {a}
- {b}

Answer with yes or no."""

TRAIT_PROMPT = """Exercise:
{exercise}

The following two phrases each describe a characteristic that this exercise attributes to "{subject}". Do they describe the same characteristic (possibly in other words or with more or less detail)?
- {a}
- {b}

Answer with yes or no."""

TOPIC_PROMPT = """Human annotators described the topic of a math exercise as follows:
{references}

Does the following description refer to the same topic (the same situation or theme, possibly in other words or with more or less detail)?
Description: {candidate}

Answer with yes or no."""


def judge(prompt, client):
    """P(yes) from the first answer token of a yes/no question, and the generated answer."""
    from .llm import p_yes

    raw, logprobs = client.chat([{"role": "user", "content": prompt}], 2, top_logprobs=20, structured={"choice": ["yes", "no"]})
    return p_yes(logprobs), raw


def fill(template, **values):
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def gold_subjects(row):
    """[(mention, genre, origin group)] of the annotated subjects; a subject without an annotated
    origin is not a first name (every annotated first name has one)."""
    return [(row[f"sujet_{k}"], row[f"genre_sujet_{k}"], gold_origin(row[f"origine_sujet_{k}"])) for k in (1, 2) if row[f"sujet_{k}"]]


def gold_origin(value):
    """Top-level group of the Wikipedia taxonomy for an annotated origin. The annotators copied the
    languages the name is used in (e.g. "English, French, Italian, ..."), which span several leaves,
    so only the top level can be recovered; all annotated names are European or Arabic."""
    if not value:
        return NOT_A_NAME
    return "GreaterAfrican" if value.startswith("Arabic") else "GreaterEuropean"


def origin_group(label):
    return TAXONOMY.get(label, label)


def gold_traits(row):
    """Trait items of subject 1 (traits_comparaison is left out: it is not tied to one subject)."""
    return [t.strip() for t in row["traits_sujet_1"].split(",") if t.strip()]


def match(yes, n_pred, n_gold):
    """Best one-to-one matching of predicted and gold subjects given the judge's yes pairs
    {(pred index, gold index)}; both lists have at most two items. Returns {gold index: pred index}."""
    best = {}
    for order in ([0, 1], [1, 0]):
        pairs = {j: i for i, j in zip(order, range(n_gold)) if i < n_pred and (i, j) in yes}
        if len(pairs) > len(best):
            best = pairs
    return best


def gold_set(value, mapping):
    return sorted({mapping[v.strip()] for v in value.split(",") if v.strip()})
