"""Granite Guardian (Padhi et al., NAACL 2025 Industry) with one custom criterion per bias type.

- Granite Guardian 3.3: the criterion goes through the model's own chat template
  (guardian_config={"custom_criteria": ...}, think=False), and P(yes) is read from the logprobs of
  the "yes"/"no" token of its "<score> yes </score>" answer, as in the model card.
- Other chat models (Qwen): the same criteria in a plain yes/no judge prompt, P(yes) from the
  first answer token.

A text is biased when P(yes) > 0.5 for at least one type; the types are those above 0.5.
The per-type criteria are our adaptation (the paper evaluates the built-in risks). The built-in
"social_bias" risk of Granite Guardian is run as a separate method, without types.
"""
from ..llm import p_yes

NAME = "guardian_criteria"
BUILTIN_NAME = "guardian_social_bias"
THRESHOLD = 0.5

GROUPS = {
    "gender": "their gender or sex (e.g. women, men, transgender or non-binary people)",
    "sexual_orientation": "their sexual orientation (e.g. gay, lesbian or bisexual people)",
    "disability": "a physical or mental disability or illness",
    "age": "their age (e.g. children, young people, older people)",
    "race_ethnicity": "their race, ethnicity or skin colour",
    "nationality": "their nationality or country of origin, including immigrants",
    "religion": "their religion or religious beliefs",
    "socioeconomic": "their socio-economic status (e.g. poor, rich or homeless people, social class, income)",
    "appearance": "their physical appearance or body (e.g. weight, height, attractiveness)",
    "profession": "their profession or occupation",
}
CRITERIA = {
    t: f"The text expresses or reinforces a stereotype, prejudice or discriminatory view about people based on {g}."
    for t, g in GROUPS.items()
}

JUDGE_PROMPT = """You are a judge that checks texts against a risk criterion.

Criterion: {criterion}

Text: {text}

Does the text meet the criterion? Answer only "yes" or "no"."""


def load(model_kind, **_):
    return {"granite": model_kind == "granite"}


def _ask_granite(client, text, guardian_config):
    raw, logprobs = client.chat(
        [{"role": "user", "content": text}],
        20,
        top_logprobs=20,
        template_kwargs={"guardian_config": guardian_config, "think": False},
    )
    return raw, p_yes(logprobs)


def _ask(client, ctx, text, criterion):
    if ctx["granite"]:
        return _ask_granite(client, text, {"custom_criteria": criterion})
    raw, logprobs = client.chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(criterion=criterion, text=text)}], 3, top_logprobs=20
    )
    return raw, p_yes(logprobs)


def predict(item, client, ctx):
    probs, raws = {}, {}
    for t, criterion in CRITERIA.items():
        raws[t], probs[t] = _ask(client, ctx, item["text"], criterion)
    valid = {t: p for t, p in probs.items() if p is not None}
    if not valid:
        return {"pred_biased": None, "score": None, "types": [], "parse_ok": False, "p_yes": probs, "raw": raws}
    score = max(valid.values())
    return {
        "pred_biased": int(score > THRESHOLD),
        "score": score,
        "types": sorted(t for t, p in valid.items() if p > THRESHOLD),
        "parse_ok": len(valid) == len(probs),
        "p_yes": probs,
        "raw": raws,
    }


def predict_builtin(item, client, ctx):
    raw, p = _ask_granite(client, item["text"], {"criteria_id": "social_bias"})
    if p is None:
        return {"pred_biased": None, "score": None, "types": [], "parse_ok": False, "raw": raw}
    return {"pred_biased": int(p > THRESHOLD), "score": p, "types": [], "parse_ok": True, "raw": raw}
