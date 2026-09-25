"""Metrics for the bias detection runs in results/<model-slug>/<method>/<dataset>.jsonl.

Per (method, model, dataset):
  n, n_parse_fail  -- unparsed answers count as "not biased" (score 0) in the metrics below
  macro_f1, auroc  -- binary gold label (EMGSD, StereoDetect, SBIC, ToxiGen, GUS)
  spearman         -- score vs graded human score (FSB, ToxiGen)
  pair_acc         -- CrowS-Pairs: share of pairs where the stereotypical sentence gets the higher
                      score (ties count 0.5)
  type_hit         -- on gold-biased items with known gold types: share where the method flags the
                      text AND one of its predicted types is a gold type
  primary          -- the metric used in the README table (macro-F1, FSB: Spearman, CrowS: pair_acc)

Usage: python -m fairblabla.evaluate [--results results] [--readme README.md]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .data import DATASETS
from .methods import LABELS

PRIMARY = {"fifty_shades_of_bias": "spearman", "crows_pairs": "pair_acc"}
DATASET_LABELS = {
    "emgsd": "EMGSD",
    "stereodetect": "StereoDetect",
    "fifty_shades_of_bias": "FSB",
    "sbic": "SBIC",
    "crows_pairs": "CrowS-Pairs",
    "toxigen": "ToxiGen",
    "gus": "GUS",
}
FIELDS = ["method", "model", "dataset", "n", "n_parse_fail", "macro_f1", "auroc", "spearman", "pair_acc", "type_hit", "n_type", "primary"]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def ranks(x):
    """Ranks starting at 1, ties get their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def macro_f1(y, p):
    y, p = np.asarray(y), np.asarray(p)
    f1s = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return float(np.mean(f1s))


def auroc(y, s):
    y = np.asarray(y)
    pos, neg = y.sum(), len(y) - y.sum()
    if not pos or not neg:
        return math.nan
    r = ranks(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0  # constant predictions: no correlation
    return float(np.corrcoef(ra, rb)[0, 1])


def read_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r  # last row per id wins (retried calls)
    return list(rows.values())


def metrics(dataset, rows):
    pred = [r["pred_biased"] or 0 for r in rows]
    score = [r["score"] if r["score"] is not None else 0.0 for r in rows]
    m = {"n": len(rows), "n_parse_fail": sum(not r["parse_ok"] for r in rows)}
    if dataset == "crows_pairs":
        pairs = {}
        for r, s in zip(rows, score):
            pairs.setdefault(r["pair_id"], {})[r["stereo"]] = s
        wins = [1.0 if p[True] > p[False] else 0.5 if p[True] == p[False] else 0.0 for p in pairs.values() if len(p) == 2]
        m["pair_acc"] = float(np.mean(wins))
        m["n"] = len(wins)
    else:
        if rows[0]["gold_biased"] is not None:
            gold = [r["gold_biased"] for r in rows]
            m["macro_f1"] = macro_f1(gold, pred)
            m["auroc"] = auroc(gold, score)
        if rows[0]["gold_score"] is not None:
            m["spearman"] = spearman([r["gold_score"] for r in rows], score)
    typed = [(r, p) for r, p in zip(rows, pred) if (r["gold_biased"] == 1 or r.get("stereo")) and set(r["gold_types"]) - {"other"}]
    if typed and any(r.get("types") for r in rows):
        m["type_hit"] = float(np.mean([bool(p and set(r["types"]) & set(r["gold_types"])) for r, p in typed]))
        m["n_type"] = len(typed)
    m["primary"] = m.get(PRIMARY.get(dataset, "macro_f1"))
    return m


def collect(results):
    results = Path(results)
    out = []
    for run_file in sorted(results.glob("*/run.json")):
        run = json.loads(run_file.read_text())
        for method in sorted(LABELS):
            for dataset in DATASETS:
                path = run_file.parent / method / f"{dataset}.jsonl"
                if path.exists():
                    out.append({"method": method, "model": run_file.parent.name, "dataset": dataset} | metrics(dataset, read_rows(path)))
    return out


def fmt(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def table(rows, results, metric):
    """Markdown table: one row per (method, model), one column per dataset, then the average."""
    displays = {p.parent.name: json.loads(p.read_text())["display"] for p in Path(results).glob("*/run.json")}
    by_run = {}
    for r in rows:
        by_run.setdefault((r["method"], r["model"]), {})[r["dataset"]] = r.get(metric)
    order = list(LABELS)
    lines = [
        "| Method | Model | " + " | ".join(DATASET_LABELS[d] for d in DATASETS) + " | Average |",
        "|---|---|" + "---:|" * (len(DATASETS) + 1),
    ]
    for (method, model), cells in sorted(by_run.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        values = [cells.get(d) for d in DATASETS]
        known = [v for v in values if v is not None and not math.isnan(v)]
        if not known:
            continue
        avg = float(np.mean(known)) if len(known) == len(DATASETS) or metric == "type_hit" else None
        lines.append(f"| {LABELS[method]} | {displays.get(model, model)} | " + " | ".join(fmt(v) for v in values) + f" | {fmt(avg)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--readme", help="replace the text between the results markers in this file")
    args = parser.parse_args()

    rows = collect(args.results)
    with open(Path(args.results) / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    primary = table(rows, args.results, "primary")
    types = table(rows, args.results, "type_hit")
    print(primary, "\n\n", types, sep="")
    if args.readme:
        path = Path(args.readme)
        text = path.read_text(encoding="utf-8")
        head, rest = text.split(START)
        _, tail = rest.split(END)
        body = f"{START}\n{primary}\n\n**Bias type found** (type hit rate, see below):\n\n{types}\n{END}"
        path.write_text(head + body + tail, encoding="utf-8")


if __name__ == "__main__":
    main()
