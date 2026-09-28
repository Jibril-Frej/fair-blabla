"""Zero-shot extraction of the indicators of data/exercises.csv, and the topic judge, against an
OpenAI-compatible server (vLLM). See src/fairblabla/exercises.py for the prompt and label sets.

Two steps, each run by the served model:
- predict: writes results/<slug>/exercises/predictions.jsonl, one line per exercise (id, parsed
  prediction, raw answer).
- judge --predictions <other slug>: judges the topics predicted by another model (the judge should
  come from another model family) and writes results/<other slug>/exercises/topic_judge.jsonl with
  P(yes) for each pair in topic_pairs() (predicted topic vs the annotators' topics, one annotator vs
  the other, and predicted topic vs the topics of another exercise).

Usage:
  python scripts/run_exercises.py predict --base-url http://127.0.0.1:8000/v1 --model model \\
      --slug qwen3.8-27b-fp8-structured --no-thinking
  python scripts/run_exercises.py judge --base-url http://127.0.0.1:8000/v1 --model model \\
      --slug mistral-small-3.2-24b-structured --predictions qwen3.8-27b-fp8-structured
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fairblabla.exercises import judge, predict  # noqa: E402
from fairblabla.llm import Client  # noqa: E402

SHIFT = 37  # the "other exercise" of exercise i is exercise (i + SHIFT) % n


def references(row):
    return [r for r in (row["contexte_mai"], row["contexte_tuan_tu"]) if r]


def topic_pairs(rows, preds):
    """(pair kind, id, candidate, references) for the judge and the embedding similarity."""
    pairs = []
    n = len(rows)
    for i, row in enumerate(rows):
        topic = preds[row["id"]]["topic"] if row["id"] in preds else ""
        other = rows[(i + SHIFT) % n]
        pairs += [
            ("model_vs_both", row["id"], topic, references(row)),
            ("model_vs_mai", row["id"], topic, [row["contexte_mai"]]),
            ("model_vs_tuan_tu", row["id"], topic, [row["contexte_tuan_tu"]]),
            ("mai_vs_tuan_tu", row["id"], row["contexte_mai"], [row["contexte_tuan_tu"]]),
            ("model_vs_other_exercise", row["id"], topic, references(other)),
        ]
    return [p for p in pairs if p[2] and all(p[3])]


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["pred"] for r in map(json.loads, f) if r.get("pred")}


def run_predict(args, client, rows):
    out = Path(args.out) / args.slug / "exercises"
    out.mkdir(parents=True, exist_ok=True)

    def run_one(row):
        try:
            pred, raw = predict(row["exercice"], client)
            return {"id": row["id"], "pred": pred, "raw": raw}
        except Exception as e:  # recorded; counted as an empty prediction by the evaluation
            return {"id": row["id"], "pred": None, "error": repr(e)}

    with ThreadPoolExecutor(args.workers) as pool, open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in pool.map(run_one, rows):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parsed = len(load_predictions(out / "predictions.jsonl"))
    print(f"{args.slug} predictions: {parsed}/{len(rows)} parsed", flush=True)
    if not parsed:
        sys.exit("every prediction failed, see the errors in predictions.jsonl")


def run_judge(args, client, rows):
    out = Path(args.out) / args.predictions / "exercises"
    preds = load_predictions(out / "predictions.jsonl")

    def judge_one(pair):
        kind, id_, candidate, refs = pair
        try:
            p, raw = judge(candidate, refs, client)
        except Exception as e:  # recorded; left out of the means by the evaluation
            p, raw = None, repr(e)
        return {"kind": kind, "id": id_, "judge": args.slug, "candidate": candidate, "references": refs, "p_yes": p, "raw": raw}

    n_ok = 0
    with ThreadPoolExecutor(args.workers) as pool, open(out / "topic_judge.jsonl", "w", encoding="utf-8") as f:
        for r in pool.map(judge_one, topic_pairs(rows, preds)):
            n_ok += r["p_yes"] is not None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.slug} judged the topics of {args.predictions}: {n_ok} pairs scored", flush=True)
    if not n_ok:
        sys.exit("every judge call failed, see the raw answers in topic_judge.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict", "judge"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--slug", required=True, help="results sub-directory (predict) or name (judge) of the served model")
    parser.add_argument("--predictions", help="judge: slug of the model whose topics are judged")
    parser.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False to the chat template")
    parser.add_argument("--data", default="data/exercises.csv")
    parser.add_argument("--limit", type=int, help="first N exercises (smoke tests)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.step == "judge" and not args.predictions:
        parser.error("judge needs --predictions")

    client = Client(args.base_url, args.model, {"enable_thinking": False} if args.no_thinking else None)
    rows = load_rows(args.data)[: args.limit]
    (run_predict if args.step == "predict" else run_judge)(args, client, rows)


if __name__ == "__main__":
    main()
