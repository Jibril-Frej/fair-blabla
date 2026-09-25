"""Görge, Mock & Allende-Cid, FAccT 2025: linguistic indicators of stereotypes (SCSC framework).

Prompt P_F_01 from the authors' repository (fetched by scripts/fetch_method_assets.py), the same
message layout and max_tokens as their query_models.py, and their linear regression (SCSC score).
The regression weights below were read from their 2025_02_14_*.joblib files at commit ee3b8b9
(sklearn OneHotEncoder + LinearRegression); only the numbers are copied, the files are not loaded.

Our adaptations:
- the JSON answer is parsed leniently (code fences and text around the object are ignored);
- the text is called biased when the SCSC score is above 0.5 (the paper reports a graded score);
- a second call maps the category label found by the model to our bias types (the paper's prompt
  only asks for the label itself).

Structured mode: the answer is constrained to SCHEMA (the prompt's keys, with the values it allows
for the fields used by the regression, and bounded free-text fields), and the type answer to a
comma-separated list of our types.
"""
import csv
import json
import re
from pathlib import Path

from ..taxonomy import TYPES, types_from_list

NAME = "linguistic_indicators"
PROMPT_ID = "P_F_01"
THRESHOLD = 0.5

INTERCEPT = 0.5905078027908865
COEFS = {
    "generalization_category_label": {
        "generic_generic target": 0.009234867085817534,
        "generic_specified target": 0.025111180133608343,
        "individual_specified target": -0.05074480655920647,
        "subset_generic target": 0.035079884648938506,
        "subset_specified target": -0.018681125309156915,
    },
    "connotation": {"negative": 0.033682365707653976, "neutral": -0.03368236570765412},
    "gram_form": {"noun": 0.007584471266466456, "other": -0.007584471266466778},
    "generalization_situation": {
        "enduring characteristics_abstract": 0.02855364994921117,
        "enduring characteristics_concrete": 0.02713140592060131,
        "other_not-applicable": 0.0007276914215228829,
        "situational behaviour_abstract": -0.045158171151257344,
        "situational behaviour_concrete": -0.011254576140078139,
    },
    "explanation": {"no": 0.02715474665562741, "not-applicable": 0.0007276914215228829, "yes": -0.027882438077149362},
}

NA = "not-applicable"


def _enum(*values):
    return {"type": "string", "enum": [*values, NA]}


def _text(n):
    return {"type": "string", "maxLength": n}


SCHEMA = {
    "type": "object",
    "properties": {
        "has_category_label": {"type": "string", "enum": ["yes", "no"]},
        "full_label": _text(100),
        "target_type": _enum("specified target", "generic target"),
        "connotation": _enum("negative", "positive", "neutral"),
        "grammatical_form": _enum("noun", "other"),
        "linguistic_form": _enum("generic", "subset", "individual"),
        "information": _text(300),
        "situation": _enum("situational behaviour", "enduring characteristics", "other"),
        "generalization": _enum("abstract", "concrete"),
        "explanation": _enum("yes", "no"),
        "signal_word": _text(50),
    },
}
SCHEMA["required"] = list(SCHEMA["properties"])
SCHEMA["additionalProperties"] = False
TYPES_REGEX = "(" + "|".join(TYPES) + ")(, (" + "|".join(TYPES) + "))*"

TYPE_PROMPT = (
    "Sentence: {text}\n\n"
    'In this sentence, the expression "{label}" identifies a person or a group of people. '
    "Which demographic dimensions does this expression refer to? Choose one or more from: "
    + ", ".join(TYPES)
    + ".\nAnswer only with a comma-separated list of dimensions."
)


def load(assets_dir, structured=False, **_):
    path = Path(assets_dir) / "goerge" / "Prompts_to_detect_linguistic_indicators.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        prompts = {r["prompt_id"]: r for r in csv.DictReader(f, delimiter=";")}
    return {"prompt": prompts[PROMPT_ID], "structured": structured}


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _norm(value):
    return str(value).strip().lower() if value is not None else "not-applicable"


def scsc_score(fields):
    """SCSC stereotype score, as in the authors' scsc_scoring.py. Returns (score, unknown values):
    feature values unseen by the encoder get a zero vector (and are reported)."""
    if _norm(fields.get("has_category_label")) != "yes":
        return 0.0, []
    situation = _norm(fields.get("situation"))
    if situation in ("not-applicable", "other"):
        return 0.0, []
    connotation = _norm(fields.get("connotation"))
    features = {
        "generalization_category_label": f"{_norm(fields.get('linguistic_form'))}_{_norm(fields.get('target_type'))}",
        "connotation": "neutral" if connotation == "positive" else connotation,
        "gram_form": _norm(fields.get("grammatical_form")),
        "generalization_situation": f"{situation}_{_norm(fields.get('generalization'))}",
        "explanation": _norm(fields.get("explanation")),
    }
    score, unknown = INTERCEPT, []
    for feature, value in features.items():
        if value in COEFS[feature]:
            score += COEFS[feature][value]
        else:
            unknown.append(f"{feature}={value}")
    return score, unknown


def predict(item, client, ctx):
    p = ctx["prompt"]
    user = p["explanation"] + p["instruction"] + p["examples"] + " Sentence: " + item["text"]
    structured = {"json": SCHEMA, "disable_any_whitespace": True} if ctx["structured"] else None
    messages = [{"role": "system", "content": p["system_role"]}, {"role": "user", "content": user}]
    raw, _ = client.chat(messages, 400, structured=structured)
    fields = parse_json(raw)
    if fields is None:
        return {"pred_biased": None, "score": None, "types": [], "parse_ok": False, "raw": raw}
    score, unknown = scsc_score(fields)
    out = {
        "pred_biased": int(score > THRESHOLD),
        "score": score,
        "types": [],
        "parse_ok": True,
        "unknown_values": unknown,
        "fields": fields,
        "raw": raw,
    }
    label = fields.get("full_label")
    if _norm(fields.get("has_category_label")) == "yes" and label and _norm(label) != "not-applicable":
        structured = {"regex": TYPES_REGEX} if ctx["structured"] else None
        prompt = TYPE_PROMPT.format(text=item["text"], label=label)
        answer, _ = client.chat([{"role": "user", "content": prompt}], 30, structured=structured)
        out["types"] = types_from_list(answer) or ["other"]
        out["raw_types"] = answer
    return out
