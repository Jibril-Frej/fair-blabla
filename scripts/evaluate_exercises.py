"""Score the zero-shot predictions of scripts/run_exercises.py against data/exercises.csv.

- subjects: predicted and gold mentions matched one-to-one, order-free (same_subject() in
  src/fairblabla/exercises.py); precision / recall / F1 over mentions.
- genre: accuracy over matched subjects with a gold gender, and exact match of the set of genders
  of the subjects per exercise.
- origine: accuracy over matched subjects whose name origin was annotated (coarse group); plus how
  many matched subjects without an annotated origin got a name origin.
- pluriel: accuracy and macro-F1 over the four values (empty gold = "aucun").
- traits: per exercise, "has a trait" (any trait for any subject) vs a non-empty gold traits_sujet_1.
- monnaie, unites: exact match of the set per exercise, and micro P / R / F1 over (exercise, value).
- topic: mean P(yes) of the judge (a model of another family, see scripts/run_exercises.py) and share above 0.5; mean BGE-M3 cosine similarity. For each pair
  kind (the model vs the annotators, one annotator vs the other, the model vs another exercise).
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
    GOLD_MONNAIE, GOLD_UNITES, PLURIELS, gold_origin, gold_set, match_subjects,
)

EMPTY = {"subjects": [], "pluriel": "aucun", "topic": "", "monnaie": [], "unites": []}


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def macro_f1(gold, pred, labels):
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        n_pred, n_gold = pred.count(c), gold.count(c)
        if n_gold or n_pred:
            f1s.append(prf(tp, n_pred, n_gold)["f1"])
    return round(sum(f1s) / len(f1s), 3)


def confusion(gold, pred):
    return {f"{g} -> {p}": n for (g, p), n in sorted(Counter(zip(gold, pred)).items())}


def set_scores(gold_sets, pred_sets):
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold_sets, pred_sets))
    exact = sum(set(g) == set(p) for g, p in zip(gold_sets, pred_sets)) / len(gold_sets)
    return {"exact_match": round(exact, 3), **prf(tp, sum(map(len, pred_sets)), sum(map(len, gold_sets)))}


def topic_scores(path, key):
    if not path.exists():
        return None
    by_kind, judges = {}, set()
    with open(path, encoding="utf-8") as f:
        for r in map(json.loads, f):
            by_kind.setdefault(r["kind"], []).append(r[key])
            judges.add(r.get("judge"))
    scores = {"judge": ", ".join(sorted(judges))} if key == "p_yes" else {}
    for kind, values in by_kind.items():
        values = [v for v in values if v is not None]
        scores[kind] = {"n": len(values), "mean": round(sum(values) / len(values), 3)}
        if key == "p_yes":
            scores[kind]["share_yes"] = round(sum(v > 0.5 for v in values) / len(values), 3)
    return scores


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

    n_pred_subj = n_gold_subj = n_matched = 0
    genre_gold, genre_pred, origin_gold, origin_pred = [], [], [], []
    extra_origins, genre_sets = 0, []
    review = []
    for row in rows:
        pred = preds[row["id"]]
        subjects = pred["subjects"]
        gold = [(row[f"sujet_{k}"], row[f"genre_sujet_{k}"], row[f"origine_sujet_{k}"]) for k in (1, 2)]
        gold = [g for g in gold if g[0]]
        pairs = match_subjects([s["mention"] for s in subjects], [g[0] for g in gold])
        n_pred_subj += len(subjects)
        n_gold_subj += len(gold)
        n_matched += len(pairs)
        for i, j in pairs:
            if gold[j][1]:
                genre_gold.append(gold[j][1])
                genre_pred.append(subjects[i]["genre"])
            if gold[j][2]:
                origin_gold.append(gold_origin(gold[j][2]))
                origin_pred.append(subjects[i]["origine"])
            elif subjects[i]["origine"] != "pas un prénom":
                extra_origins += 1
        genre_sets.append({g[1] for g in gold if g[1]} == {s["genre"] for s in subjects})
        review.append({
            "id": row["id"],
            "exercice": row["exercice"],
            "gold_subjects": " | ".join(f"{s} ({g}{', ' + gold_origin(o) if o else ''})" for s, g, o in gold),
            "pred_subjects": " | ".join(f"{s['mention']} ({s['genre']}, {s['origine']})" for s in subjects),
            "gold_traits": " | ".join(v for v in (row["traits_sujet_1"], row["traits_comparaison"]) if v),
            "pred_traits": " | ".join(f"{s['mention']}: {', '.join(s['traits'])}" for s in subjects if s["traits"]),
            "gold_pluriel": row["pluriel"] or "aucun",
            "pred_pluriel": pred["pluriel"],
            "gold_topic": " | ".join(v for v in (row["contexte_mai"], row["contexte_tuan_tu"]) if v),
            "pred_topic": pred["topic"],
            "gold_monnaie": ", ".join(gold_set(row["monnaie"], GOLD_MONNAIE)),
            "pred_monnaie": ", ".join(pred["monnaie"]),
            "gold_unites": ", ".join(gold_set(row["unites"], GOLD_UNITES)),
            "pred_unites": ", ".join(pred["unites"]),
        })

    pl_gold = [r["pluriel"] or "aucun" for r in rows]
    pl_pred = [preds[r["id"]]["pluriel"] for r in rows]
    tr_gold = [bool(r["traits_sujet_1"]) for r in rows]
    tr_pred = [any(s["traits"] for s in preds[r["id"]]["subjects"]) for r in rows]
    metrics = {
        "n": len(rows),
        "n_failed": n_failed,
        "subjects": {"n_gold": n_gold_subj, "n_pred": n_pred_subj, **prf(n_matched, n_pred_subj, n_gold_subj)},
        "genre": {
            "n": len(genre_gold),
            "accuracy": round(sum(g == p for g, p in zip(genre_gold, genre_pred)) / len(genre_gold), 3),
            "set_exact_match": round(sum(genre_sets) / len(rows), 3),
            "confusion": confusion(genre_gold, genre_pred),
        },
        "origine": {
            "n": len(origin_gold),
            "accuracy": round(sum(g == p for g, p in zip(origin_gold, origin_pred)) / max(len(origin_gold), 1), 3),
            "origin_given_to_unannotated_subject": extra_origins,
            "confusion": confusion(origin_gold, origin_pred),
        },
        "pluriel": {
            "accuracy": round(sum(g == p for g, p in zip(pl_gold, pl_pred)) / len(rows), 3),
            "macro_f1": macro_f1(pl_gold, pl_pred, PLURIELS),
            "confusion": confusion(pl_gold, pl_pred),
        },
        "traits": {
            "n_gold": sum(tr_gold),
            "accuracy": round(sum(g == p for g, p in zip(tr_gold, tr_pred)) / len(rows), 3),
            **prf(sum(g and p for g, p in zip(tr_gold, tr_pred)), sum(tr_pred), sum(tr_gold)),
        },
        "monnaie": set_scores([gold_set(r["monnaie"], GOLD_MONNAIE) for r in rows], [preds[r["id"]]["monnaie"] for r in rows]),
        "unites": set_scores([gold_set(r["unites"], GOLD_UNITES) for r in rows], [preds[r["id"]]["unites"] for r in rows]),
        "topic_judge": topic_scores(out / "topic_judge.jsonl", "p_yes"),
        "topic_similarity": topic_scores(out / "topic_similarity.jsonl", "similarity"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with open(out / "review.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
