"""Load the evaluation samples (data/samples/*.csv) as a common list of items.

Each item is a dict with:
  dataset, id    -- id is the row position in the sample (CrowS: "<row>-more" / "<row>-less")
  text           -- the text given to the detectors
  gold_biased    -- 1 / 0, or None when the dataset has no binary label (CrowS)
  gold_types     -- list of shared types (see taxonomy.py) the text is about; [] if unknown
  gold_score     -- graded human score (ToxiGen), else None
  pair_id, stereo -- CrowS only: pair id and whether this is the stereotypical sentence of the pair
"""
import csv
import json
from pathlib import Path

from .taxonomy import types_from_text

DATASETS = ["emgsd", "stereodetect", "sbic", "crows_pairs", "toxigen"]

EMGSD_TYPES = {
    "nationality": ["nationality"],
    "profession": ["profession"],
    "gender": ["gender"],
    "lgbtq+": ["sexual_orientation", "gender"],
    "religion": ["religion"],
    "race": ["race_ethnicity"],
}
STEREODETECT_TYPES = {
    "profession": ["profession"],
    "gender": ["gender"],
    "race": ["race_ethnicity"],
    "religion": ["religion"],
    "neutral": [],
}
CROWS_TYPES = {
    "race-color": ["race_ethnicity"],
    "gender": ["gender"],
    "socioeconomic": ["socioeconomic"],
    "nationality": ["nationality"],
    "religion": ["religion"],
    "age": ["age"],
    "sexual-orientation": ["sexual_orientation"],
    "physical-appearance": ["appearance"],
    "disability": ["disability"],
}
SBIC_TYPES = {
    "race": ["race_ethnicity"],
    "gender": ["gender"],
    "disabled": ["disability"],
    "body": ["appearance"],
    "social": ["socioeconomic", "other"],
    "victim": ["other"],
}
# ToxiGen: mean human toxicity (1-5) at or above this counts as biased.
TOXIGEN_THRESHOLD = 3.0


def read_csv(path):
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _item(dataset, i, text, biased=None, types=(), score=None, **extra):
    return {
        "dataset": dataset,
        "id": str(i),
        "text": text,
        "gold_biased": biased,
        "gold_types": sorted(set(types)),
        "gold_score": score,
        **extra,
    }


def sbic_types(row):
    """Types from SBIC's targetCategory, refined with the free-text targetMinority."""
    cats = json.loads(row["targetCategory"] or "[]")
    minority = " ".join(json.loads(row["targetMinority"] or "[]"))
    types = set()
    for c in cats:
        if c == "culture":
            found = set(types_from_text(minority)) & {"religion", "race_ethnicity", "nationality"}
            types |= found or {"religion", "race_ethnicity", "nationality"}
        else:
            types |= set(SBIC_TYPES.get(c, ["other"]))
    # e.g. targetCategory "gender" with targetMinority "gay men"
    types |= set(types_from_text(minority)) - {"other"} if minority else set()
    return sorted(types)




def load_sample(name, data_dir="data"):
    rows = read_csv(Path(data_dir) / "samples" / f"{name}.csv")
    items = []
    for i, r in enumerate(rows):
        if name == "emgsd":
            biased = int(r["category"] == "stereotype")
            items.append(_item(name, i, r["text"], biased, EMGSD_TYPES[r["stereotype_type"]]))
        elif name == "stereodetect":
            biased = int(r["labels"] in ("1", "4"))
            items.append(_item(name, i, r["Sentence"], biased, STEREODETECT_TYPES[r["Category"].lower()]))
        elif name == "sbic":
            biased = int(r["hasBiasedImplication"] == "0")  # inverted in SBIC: 0 = biased
            items.append(_item(name, i, r["post"], biased, sbic_types(r)))
        elif name == "toxigen":
            tox = float(r["toxicity_human"])
            items.append(_item(name, i, r["text"], int(tox >= TOXIGEN_THRESHOLD), types_from_text(r["target_group"]), tox))
        elif name == "crows_pairs":
            types = CROWS_TYPES[r["bias_type"]]
            stereo_is_more = r["stereo_antistereo"] == "stereo"
            items.append(_item(name, f"{i}-more", r["sent_more"], None, types, pair_id=str(i), stereo=stereo_is_more))
            items.append(_item(name, f"{i}-less", r["sent_less"], None, types, pair_id=str(i), stereo=not stereo_is_more))
        else:
            raise ValueError(f"unknown dataset {name}")
    return items


def load_all(data_dir="data", datasets=None):
    return [it for name in (datasets or DATASETS) for it in load_sample(name, data_dir)]


# --- Retrieval pool for the demographic-axes method (Majumdar et al.) ---------------------------

# Shared type -> policy code of the 9 demographic axes (profession has no axis).
AXIS_CODES = {
    "gender": "S1",
    "sexual_orientation": "S2",
    "disability": "S3",
    "age": "S4",
    "race_ethnicity": "S5",
    "nationality": "S6",
    "religion": "S7",
    "socioeconomic": "S8",
    "appearance": "S9",
}
SAFE_CODE = "S10"


def _codes(types):
    return sorted({AXIS_CODES[t] for t in types if t in AXIS_CODES}, key=lambda c: int(c[1:]))


def build_axes_pool(data_dir="data"):
    """Labelled examples from the train splits of EMGSD, StereoDetect, SBIC and ToxiGen, with the
    answer in the policy's format ("S1,S5" or "S10"). Biased rows whose group maps to no axis
    (e.g. profession) are left out, as are texts that also appear in an evaluation sample.
    Returns a list of {"pool_id", "text", "answer"} in a fixed order."""
    data = Path(data_dir)
    exclude = {it["text"].strip().lower() for it in load_all(data_dir)}
    pool, seen = [], set()

    def add(pool_id, text, biased, types):
        key = text.strip().lower()
        if not key or key in exclude or key in seen:
            return
        codes = _codes(types) if biased else [SAFE_CODE]
        if not codes:
            return
        seen.add(key)
        pool.append({"pool_id": pool_id, "text": text, "answer": ",".join(codes)})

    for i, r in enumerate(read_csv(data / "emgsd/train.csv")):
        add(f"emgsd:{i}", r["text"], r["category"] == "stereotype", EMGSD_TYPES[r["stereotype_type"]])
    for i, r in enumerate(read_csv(data / "stereodetect/train.csv")):
        add(f"stereodetect:{i}", r["Sentence"], r["labels"] in ("1", "4"), STEREODETECT_TYPES[r["Category"].lower()])
    for i, r in enumerate(read_csv(data / "sbic/SBIC.v2.agg.trn.csv")):
        add(f"sbic:{i}", r["post"], r["hasBiasedImplication"] == "0", sbic_types(r))
    for i, r in enumerate(read_csv(data / "toxigen/annotated_train.csv")):
        tox = float(r["toxicity_human"])
        if 2 < tox < 4:  # ambiguous middle of the scale
            continue
        add(f"toxigen:{i}", r["text"], tox >= 4, types_from_text(r["target_group"]))
    return pool
