"""Score the zero-shot predictions of scripts/run_exercises.py against data/exercises.csv.

Single-label classification, macro-F1 (classes absent from both gold and predictions are skipped):
- genre: on predicted subjects matched to a gold subject;
- origine: on matched subjects, at the top level of the Wikipedia taxonomy (the level the gold
  supports, see gold_origin()); a gold subject without an annotated origin is "pas un prénom";
- pluriel: per exercise (empty gold = "aucun").
Multi-label classification, micro-F1 over (exercise, value) pairs: monnaie, unites.
Free text, with the judge's answers (judge.jsonl, yes = P(yes) > 0.5; a failed call counts as no):
- subjects and traits (lists): micro-F1, a predicted item being correct if the judge matched it to a
  gold item, a gold item being found if the judge matched a predicted item to it. Traits compare the
  predicted subject matched to gold subject 1 with gold traits_sujet_1; if gold subject 1 was not
  found, its gold traits are missed.
- topic (one value): share of yes against both annotators.
Matched subjects: best one-to-one matching of the judge's yes pairs (match() in exercises.py).
Failed predictions count as empty ones.

Writes results/<slug>/exercises/metrics.json and review.csv (gold and predicted values side by side).

Usage: python scripts/evaluate_exercises.py --slug qwen3.8-27b-fp8-structured
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fairblabla.exercises import (  # noqa: E402
    GENRES, GOLD_MONNAIE, GOLD_UNITES, NOT_A_NAME, PLURIELS, TAXONOMY, gold_set, gold_subjects, gold_traits, match,
    origin_group,
)

EMPTY = {"subjects": [], "pluriel": "aucun", "topic": "", "monnaie": [], "unites": []}
ORIGIN_GROUPS = [NOT_A_NAME, *dict.fromkeys(TAXONOMY.values())]


def prf(tp_pred, n_pred, tp_gold, n_gold):
    """Precision from the correct predictions, recall from the gold items found (the same count for
    one-to-one matches)."""
    p = tp_pred / n_pred if n_pred else 0.0
    r = tp_gold / n_gold if n_gold else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def macro_f1(gold, pred, labels):
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        n_pred, n_gold = pred.count(c), gold.count(c)
        if n_gold or n_pred:
            f1s.append(prf(tp, n_pred, tp, n_gold)["f1"])
    return round(sum(f1s) / len(f1s), 3)


def single_label(gold, pred, labels):
    return {"n": len(gold), "macro_f1": macro_f1(gold, pred, labels), "confusion": {f"{g} -> {p}": n for (g, p), n in sorted(Counter(zip(gold, pred)).items())}}


def multi_label(gold_sets, pred_sets):
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold_sets, pred_sets))
    n_pred, n_gold = sum(map(len, pred_sets)), sum(map(len, gold_sets))
    return {"n_gold": n_gold, "n_pred": n_pred, **prf(tp, n_pred, tp, n_gold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--data", default="data/exercises.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out) / args.slug / "exercises"

    with open(args.data, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(out / "predictions.jsonl", encoding="utf-8") as f:
        preds = {r["id"]: r.get("pred") for r in map(json.loads, f)}
    n_failed = sum(not preds.get(r["id"]) for r in rows)
    preds = {r["id"]: preds.get(r["id"]) or EMPTY for r in rows}
    with open(out / "judge.jsonl", encoding="utf-8") as f:
        answers = [json.loads(line) for line in f]
    judges = sorted({a["judge"] for a in answers})
    yes = {(a["kind"], a["id"], a.get("pred"), a.get("gold")) for a in answers if (a["p_yes"] or 0) > 0.5}

    counts = Counter()  # subject_ / trait_ + n_pred, tp_pred, n_gold, tp_gold
    genre_gold, genre_pred, origin_gold, origin_pred, topic_yes = [], [], [], [], []
    review = []
    for row in rows:
        id_, pred = row["id"], preds[row["id"]]
        subjects, gold = pred["subjects"], gold_subjects(row)
        pairs = {(i, j) for kind, x, i, j in yes if kind == "subject" and x == id_}
        counts["subject_n_pred"] += len(subjects)
        counts["subject_tp_pred"] += len({i for i, _ in pairs})
        counts["subject_n_gold"] += len(gold)
        counts["subject_tp_gold"] += len({j for _, j in pairs})
        matched = match(pairs, len(subjects), len(gold))
        for j, i in matched.items():
            genre_gold.append(gold[j][1])
            genre_pred.append(subjects[i]["genre"])
            origin_gold.append(gold[j][2])
            origin_pred.append(origin_group(subjects[i]["origine"]))

        traits_gold = gold_traits(row)
        traits_pred = subjects[matched[0]]["traits"] if 0 in matched else []
        tpairs = {(k, j) for kind, x, k, j in yes if kind == "trait" and x == id_}
        counts["trait_n_pred"] += len(traits_pred)
        counts["trait_tp_pred"] += len({k for k, _ in tpairs})
        counts["trait_n_gold"] += len(traits_gold)
        counts["trait_tp_gold"] += len({j for _, j in tpairs})

        if row["contexte_mai"] or row["contexte_tuan_tu"]:
            topic_yes.append(("topic", id_, None, None) in yes)

        review.append({
            "id": id_,
            "exercice": row["exercice"],
            "gold_subjects": " | ".join(f"{s} ({g}, {o})" for s, g, o in gold),
            "pred_subjects": " | ".join(f"{s['mention']} ({s['genre']}, {s['origine']})" for s in subjects),
            "matched": " | ".join(f"{gold[j][0]} = {subjects[i]['mention']}" for j, i in sorted(matched.items())),
            "gold_traits_1": " | ".join(traits_gold),
            "pred_traits_1": " | ".join(traits_pred),
            "gold_pluriel": row["pluriel"] or "aucun",
            "pred_pluriel": pred["pluriel"],
            "gold_topic": " | ".join(v for v in (row["contexte_mai"], row["contexte_tuan_tu"]) if v),
            "pred_topic": pred["topic"],
            "topic_yes": ("topic", id_, None, None) in yes,
            "gold_monnaie": ", ".join(gold_set(row["monnaie"], GOLD_MONNAIE)),
            "pred_monnaie": ", ".join(pred["monnaie"]),
            "gold_unites": ", ".join(gold_set(row["unites"], GOLD_UNITES)),
            "pred_unites": ", ".join(pred["unites"]),
        })

    def list_scores(kind):
        c = {k: counts[f"{kind}_{k}"] for k in ("n_pred", "tp_pred", "n_gold", "tp_gold")}
        return {"n_gold": c["n_gold"], "n_pred": c["n_pred"], **prf(c["tp_pred"], c["n_pred"], c["tp_gold"], c["n_gold"])}

    metrics = {
        "n": len(rows),
        "n_failed_predictions": n_failed,
        "judge": ", ".join(judges),
        "n_failed_judge_calls": sum(a["p_yes"] is None for a in answers),
        "genre": single_label(genre_gold, genre_pred, GENRES),
        "origine": single_label(origin_gold, origin_pred, ORIGIN_GROUPS),
        "pluriel": single_label([r["pluriel"] or "aucun" for r in rows], [preds[r["id"]]["pluriel"] for r in rows], PLURIELS),
        "monnaie": multi_label([gold_set(r["monnaie"], GOLD_MONNAIE) for r in rows], [preds[r["id"]]["monnaie"] for r in rows]),
        "unites": multi_label([gold_set(r["unites"], GOLD_UNITES) for r in rows], [preds[r["id"]]["unites"] for r in rows]),
        "subjects": list_scores("subject"),
        "traits": list_scores("trait"),
        "topic": {"n": len(topic_yes), "share_yes": round(sum(topic_yes) / len(topic_yes), 3)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with open(out / "review.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
