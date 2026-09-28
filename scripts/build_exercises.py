"""Merge the annotation sheets of data/Final labels.xlsx into one CSV, one row per exercise.

Each sheet has the exercise, one label per annotator (Mai and Tuan Tu; Mai and Paul in
"eurometric") and a "Décision" column. The final label of a sheet is:
- "check": the annotators agree. If one of them left the cell empty, the other value is kept;
  if both wrote something different (wording only, e.g. "taxi" / "un taxi"), Tuan Tu's (or
  Paul's) value is kept.
- "Mai" / "Tuan Tu": the value of that annotator.
- "additional info": the values of both annotators (only one is usually filled).
Empty cells, "NA" and "x" all mean "none" and are written as an empty string.

Mai also wrote the cultural setting of the exercise (européen, système métrique, notes sur 20, ...)
in the "Origine sujet 1" and "Traits sujet 1" sheets. These values go to `contexte_culturel`
instead of the origin and trait columns.

`contexte_mai` and `contexte_tuan_tu` keep each annotator's own description of the topic, used as
the two references when scoring predicted topics.

Usage: python scripts/build_exercises.py [--xlsx "data/Final labels.xlsx"] [--out data/exercises.csv]
"""
import argparse
import csv
import re

import openpyxl

EMPTY = {"", "na", "x"}
CULTURAL = re.compile(r"europ|syst[eè]me|m[ée]trique|suisse|notes sur 20|pays|celsius|litres", re.I)
FRENCH = {"français", "françaos", "french"}

# output column -> (sheet, first annotator column, second annotator column, decision column)
FIELDS = {
    "objets": ("Objects", 2, 3, 4),
    "pluriel": ("Pluriels", 2, 3, 4),
    "sujet_1": ("Sujet 1", 2, 3, 4),
    "genre_sujet_1": ("Genre Sujet 1", 2, 3, 4),
    "origine_sujet_1": ("Origine sujet 1", 2, 3, 4),
    "traits_sujet_1": ("Traits sujet 1", 2, 3, 4),
    "sujet_2": ("Sujet 2", 2, 3, 4),
    "genre_sujet_2": ("Genre Sujet 2", 2, 3, 4),
    "origine_sujet_2": ("Origine Sujet 2", 2, 3, 4),
    "contexte": ("Contexte", 2, 3, 4),
    "monnaie": ("eurometric", 2, 4, 6),
    "unites": ("eurometric", 3, 5, 6),
}
COLUMNS = [
    "id", "rid", "exercice", *FIELDS, "traits_comparaison", "contexte_label", "contexte_culturel",
    "contexte_mai", "contexte_tuan_tu",
]


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value = re.sub(r"\s+", " ", str(value)).strip().strip(",").strip()
    value = re.sub(r"\s*\[\d+\]$", "", value)  # footnote marks copied with name origins
    return "" if value.lower() in EMPTY else value


def decide(first, second, decision):
    if decision == "Mai":
        return first
    if decision == "Tuan Tu":
        return second
    if decision == "additional info":
        return ", ".join(unique([first, second]))
    if decision != "check":
        raise ValueError(f"unknown decision {decision!r}")
    return second or first


def unique(values):
    """Non-empty values without repeats (ignoring case), in order."""
    return list({v.lower(): v for v in reversed([v for v in values if v])}.values())[::-1]


def rows(ws):
    return [r for r in ws.iter_rows(min_row=2, values_only=True) if any(v is not None for v in r)]


def split_cultural(value):
    """Split a comma-separated value into (non-cultural part, cultural part)."""
    keep, cultural = [], []
    for part in (p.strip() for p in value.split(",")):
        if part:
            (cultural if CULTURAL.search(part) else keep).append(part)
    return ", ".join(keep), ", ".join(cultural)


def french(value):
    return "French" if value.lower() in FRENCH else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default="data/Final labels.xlsx")
    parser.add_argument("--out", default="data/exercises.csv")
    args = parser.parse_args()

    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    sheets = {ws.title: rows(ws) for ws in wb.worksheets}
    base = sheets["Objects"]
    for name, sheet in sheets.items():
        assert [r[:2] for r in sheet] == [r[:2] for r in base], f"sheet {name} is not aligned"

    out = []
    for i, row in enumerate(base):
        rec = {"id": f"ex{i + 1:03d}", "rid": int(row[0]), "exercice": str(row[1]).strip()}
        for col, (sheet, a, b, d) in FIELDS.items():
            r = sheets[sheet][i]
            rec[col] = decide(clean(r[a]), clean(r[b]), clean(r[d]) or "check")

        cultural = []
        for col in ("origine_sujet_1", "traits_sujet_1"):
            rec[col], found = split_cultural(rec[col])
            cultural.append(found)
        traits = sheets["Traits sujet 1"][i]
        notes = ", ".join(v for v in map(clean, traits[5:]) if v)
        rec["traits_comparaison"], found = split_cultural(notes)
        cultural.append(found)
        rec["contexte_culturel"] = ", ".join(unique(c for f in cultural for c in f.split(", ")))
        for col in ("origine_sujet_1", "origine_sujet_2"):
            rec[col] = french(rec[col])
        ctx = sheets["Contexte"][i]
        rec["contexte_label"] = clean(ctx[5])
        rec["contexte_mai"], rec["contexte_tuan_tu"] = clean(ctx[2]), clean(ctx[3])
        out.append(rec)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out)
    print(f"{len(out)} exercises -> {args.out}")


if __name__ == "__main__":
    main()
