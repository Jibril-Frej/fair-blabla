"""Zero-shot extraction of the indicators of data/exercises.csv, and the judge of the free-text
indicators, against an OpenAI-compatible server (vLLM). See src/fairblabla/exercises.py for the
prompts and label sets.

Two steps, each run by the served model:
- predict: writes results/<slug>/exercises/predictions.jsonl, one line per exercise (id, parsed
  prediction, raw answer).
- judge --predictions <other slug>: judges the free-text values predicted by another model (the
  judge should come from another model family) and writes results/<other slug>/exercises/judge.jsonl,
  one yes/no question per line with P(yes):
  - subject: every (predicted subject, gold subject) pair of an exercise;
  - trait: every (predicted trait, gold trait of subject 1) pair, for the predicted subject that the
    subject answers match to gold subject 1 (match() in exercises.py);
  - topic: the predicted topic against the topics of both annotators.

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
from fairblabla.exercises import (  # noqa: E402
    SUBJECT_PROMPT, TOPIC_PROMPT, TRAIT_PROMPT, fill, gold_subjects, gold_traits, judge, match, predict,
)
from fairblabla.llm import Client  # noqa: E402


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["pred"] for r in map(json.loads, f) if r.get("pred")}


def subject_questions(rows, preds):
    questions = []
    for row in rows:
        subjects = preds.get(row["id"], {}).get("subjects", [])
        for i, s in enumerate(subjects):
            for j, (gold, _, _) in enumerate(gold_subjects(row)):
                prompt = fill(SUBJECT_PROMPT, exercise=row["exercice"], a=s["mention"], b=gold)
                questions.append(({"kind": "subject", "id": row["id"], "pred": i, "gold": j, "candidate": s["mention"], "reference": gold}, prompt))
    return questions


def trait_questions(rows, preds, subject_answers):
    """Needs the subject answers: the traits compared are those of the subject matched to gold subject 1."""
    questions = []
    for row in rows:
        subjects = preds.get(row["id"], {}).get("subjects", [])
        gold = gold_subjects(row)
        yes = {(a["pred"], a["gold"]) for a in subject_answers if a["id"] == row["id"] and (a["p_yes"] or 0) > 0.5}
        i = match(yes, len(subjects), len(gold)).get(0)
        if i is None:
            continue
        for k, trait in enumerate(subjects[i]["traits"]):
            for j, ref in enumerate(gold_traits(row)):
                prompt = fill(TRAIT_PROMPT, exercise=row["exercice"], subject=gold[0][0], a=trait, b=ref)
                questions.append(({"kind": "trait", "id": row["id"], "pred": k, "gold": j, "candidate": trait, "reference": ref}, prompt))
    return questions


def topic_questions(rows, preds):
    questions = []
    for row in rows:
        topic = preds.get(row["id"], {}).get("topic", "")
        refs = [r for r in (row["contexte_mai"], row["contexte_tuan_tu"]) if r]
        if topic and refs:
            prompt = fill(TOPIC_PROMPT, references="\n".join(f"- {r}" for r in refs), candidate=topic)
            questions.append(({"kind": "topic", "id": row["id"], "candidate": topic, "reference": refs}, prompt))
    return questions


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

    def ask(question):
        record, prompt = question
        try:
            p, raw = judge(prompt, client)
        except Exception as e:  # recorded; counted as a "no" by the evaluation
            p, raw = None, repr(e)
        return {**record, "judge": args.slug, "p_yes": p, "raw": raw}

    with ThreadPoolExecutor(args.workers) as pool:
        subjects = list(pool.map(ask, subject_questions(rows, preds)))
        rest = list(pool.map(ask, trait_questions(rows, preds, subjects) + topic_questions(rows, preds)))
    answers = subjects + rest
    with open(out / "judge.jsonl", "w", encoding="utf-8") as f:
        for r in answers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(r["p_yes"] is not None for r in answers)
    print(f"{args.slug} judged {args.predictions}: {n_ok}/{len(answers)} questions answered", flush=True)
    if not n_ok:
        sys.exit("every judge call failed, see the raw answers in judge.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["predict", "judge"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--slug", required=True, help="results sub-directory (predict) or name (judge) of the served model")
    parser.add_argument("--predictions", help="judge: slug of the model whose predictions are judged")
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
