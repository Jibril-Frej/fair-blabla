"""Cosine similarity of BGE-M3 embeddings (CLS token, normalised) between predicted and annotated
topics, for the same pairs as the topic judge (see topic_pairs() in scripts/run_exercises.py).
With several references, the similarity is the highest one.

Writes results/<slug>/exercises/topic_similarity.jsonl. Needs torch + transformers.

Usage: python scripts/embed_topics.py --slug qwen3.8-27b-fp8-structured
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import embed  # noqa: E402
from run_exercises import load_predictions, load_rows, topic_pairs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--data", default="data/exercises.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out) / args.slug / "exercises"

    rows = load_rows(args.data)
    pairs = topic_pairs(rows, load_predictions(out / "predictions.jsonl"))
    texts = sorted({t for _, _, c, refs in pairs for t in [c, *refs]})
    vectors = dict(zip(texts, embed("BAAI/bge-m3", texts, "cls")))
    with open(out / "topic_similarity.jsonl", "w", encoding="utf-8") as f:
        for kind, id_, candidate, refs in pairs:
            sim = max(float(vectors[candidate] @ vectors[r]) for r in refs)
            row = {"kind": kind, "id": id_, "candidate": candidate, "references": refs, "similarity": round(sim, 4)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {out / 'topic_similarity.jsonl'}")


if __name__ == "__main__":
    main()
