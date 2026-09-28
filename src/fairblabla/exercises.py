"""Zero-shot extraction of the annotated indicators of data/exercises.csv (French math exercises).

One constrained call per exercise returns, as JSON:
- subjects: at most two main actors, each with the words copied from the text (mention), the
  gender of that wording, the coarse origin of the name (only for first names) and its traits;
- pluriel: how groups of people are written;
- topic: a short free-text description of the situation of the exercise;
- monnaie / unites: currencies and cultural units that appear.
The label sets are general (they do not list the answers of this dataset), except that the gold
annotations are mapped onto them in GOLD_* below.

The predicted topic is scored by a yes/no judge (P(yes), same served model) and, separately, by
embedding similarity (scripts/embed_topics.py).
"""
import json
import re
import unicodedata

GENRES = ["masculin", "féminin", "féminin et masculin", "neutre (objet)"]
ORIGINS = ["pas un prénom", "européen", "arabe / maghrébin", "africain subsaharien", "asiatique", "autre"]
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
  - origine: only when the mention is a first name, the cultural origin that name is most associated with; otherwise "pas un prénom".
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


JUDGE_PROMPT = """Human annotators described the topic of a math exercise as follows:
{references}

Does the following description refer to the same topic (the same situation or theme, possibly in other words or with more or less detail)?
Description: {candidate}

Answer with yes or no."""


def judge(candidate, references, client):
    """P(yes) that the candidate topic matches the references, from the first answer token."""
    from .llm import p_yes

    refs = "\n".join(f"- {r}" for r in references)
    prompt = JUDGE_PROMPT.replace("{references}", refs).replace("{candidate}", candidate)
    raw, logprobs = client.chat([{"role": "user", "content": prompt}], 2, top_logprobs=20, structured={"choice": ["yes", "no"]})
    return p_yes(logprobs), raw


ARTICLES = r"^(un|une|le|la|les|l|des|du|de|sa|son|ses|leur|leurs)\s+"


def normalise(mention):
    """Lowercase, no accents, no inclusive-writing dots or punctuation, no leading article."""
    text = unicodedata.normalize("NFKD", mention.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[·•.]e\b|[·•.]", "", text)  # étudiant·e -> etudiant
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(ARTICLES, "", text).strip()


def same_subject(pred, gold):
    """A predicted mention matches a gold one when one contains the other, word-wise, after normalise()."""
    p, g = f" {normalise(pred)} ", f" {normalise(gold)} "
    return p.strip() != "" and g.strip() != "" and (p in g or g in p)


def match_subjects(pred, gold):
    """Best one-to-one matching of predicted and gold subjects (both lists have at most two items).
    Returns a list of (pred index, gold index)."""
    best = []
    for order in ([0, 1], [1, 0]):
        pairs = [(i, j) for i, j in zip(order, range(len(gold))) if i < len(pred) and same_subject(pred[i], gold[j])]
        if len(pairs) > len(best):
            best = pairs
    return best


def gold_origin(value):
    """Coarse group of an annotated name origin (the annotators copied the origins listed for the name)."""
    if not value:
        return None
    return "arabe / maghrébin" if value.startswith("Arabic") else "européen"


def gold_set(value, mapping):
    return sorted({mapping[v.strip()] for v in value.split(",") if v.strip()})
