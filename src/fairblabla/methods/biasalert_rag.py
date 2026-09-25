"""BiasAlert (Fan et al., EMNLP 2024) with an off-the-shelf chat model instead of their fine-tuned judge.

The instruction template and the bias database (bias_doc.tsv) come from the authors' repository
(fetched by scripts/fetch_method_assets.py). The 5 closest database entries are retrieved with
facebook/contriever-msmarco by scripts/retrieve.py. The repository does not show where the retrieved
entries go in the prompt; we put them between the instruction and the sentence.
The instruction asks the model to think step by step, so the verdict is the last "Yes/No, the
following SENTENCE is ..." in the answer, and the score is P("Yes") at that verdict token.
"""
import csv
import json
import re
from pathlib import Path

from ..llm import p_yes
from ..taxonomy import types_from_text

NAME = "biasalert_rag"
K = 5


def load(assets_dir, retrieval_dir, **_):
    base = Path(assets_dir) / "biasalert"
    with open(base / "bias_doc.tsv", encoding="utf-8", newline="") as f:
        docs = {r["id"]: r["text"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    instruction = (base / "instruction.txt").read_text(encoding="utf-8")
    neighbours = {}
    with open(Path(retrieval_dir) / "biasalert.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            neighbours[(r["dataset"], r["id"])] = r["neighbours"][:K]
    return {"docs": docs, "instruction": instruction, "neighbours": neighbours}


def build_prompt(instruction, text, references):
    refs = "\n".join(f"{i}. {r}" for i, r in enumerate(references, 1))
    return f"{instruction}\nReferences:\n{refs}\nThe SENTENCE is: '{text}'"


def parse(raw):
    """Returns (biased, bias type, association) from the last verdict, or (None, None, None)."""
    matches = list(re.finditer(r"\b(Yes|No), the following SENTENCE is", raw, re.I))
    m = matches[-1] if matches else re.match(r"\W*(yes|no)\b", raw, re.I)
    if not m:
        return None, None, None
    biased = int(m.group(1).lower() == "yes")
    tail = raw[m.start():]
    kind = re.search(r"type of bias is\s*\[?([^\].\n]+)", tail, re.I)
    assoc = re.search(r"associating\s*(.+)", tail, re.I)
    return biased, kind.group(1).strip() if kind else None, assoc.group(1).strip(" '.") if assoc else None


def verdict_position(logprobs):
    """Index of the last "Yes"/"No" token directly followed by "," (the template's verdict)."""
    pos = 0
    for i in range(len(logprobs) - 1):
        if logprobs[i]["token"].strip().lower() in ("yes", "no") and logprobs[i + 1]["token"].startswith(","):
            pos = i
    return pos


def predict(item, client, ctx):
    ids = ctx["neighbours"][(item["dataset"], item["id"])]
    prompt = build_prompt(ctx["instruction"], item["text"], [ctx["docs"][i] for i in ids])
    raw, logprobs = client.chat([{"role": "user", "content": prompt}], 4096, top_logprobs=20)
    biased, kind, assoc = parse(raw)
    out = {"parse_ok": biased is not None, "bias_type": kind, "association": assoc, "raw": raw, "docs": ids}
    if biased is None:
        return out | {"pred_biased": None, "score": None, "types": []}
    p = p_yes(logprobs[verdict_position(logprobs):][:1])
    types = types_from_text(kind) if biased and kind else []
    return out | {"pred_biased": biased, "score": p if p is not None else float(biased), "types": types}
