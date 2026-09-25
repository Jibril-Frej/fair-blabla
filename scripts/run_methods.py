"""Run bias detection methods on the evaluation samples against an OpenAI-compatible server (vLLM).

Writes results/<model-slug>/<method>/<dataset>.jsonl (one line per item: gold labels, prediction,
raw model output) and results/<model-slug>/run.json (model and settings).
Items already in an output file are skipped (except failed calls), so an interrupted run can be resumed.

Usage:
  python scripts/run_methods.py --base-url http://127.0.0.1:8000/v1 --model qwen \\
      --slug qwen3.8-27b-bf16 --display "Qwen3.8-27B (BF16)" --model-id Qwen/Qwen3.8-27B \\
      --kind chat --methods linguistic_indicators demographic_axes guardian_criteria biasalert_rag
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fairblabla.data import DATASETS, load_sample  # noqa: E402
from fairblabla.llm import Client  # noqa: E402
from fairblabla.methods import METHODS  # noqa: E402


def done_ids(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # rows that failed with an exception are retried (evaluate.py keeps the last row per id)
    return {r["id"] for r in rows if "error" not in r}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--slug", required=True, help="results sub-directory for this model")
    parser.add_argument("--display", required=True, help="model name shown in the results table")
    parser.add_argument("--model-id", required=True, help="Hugging Face id of the weights")
    parser.add_argument("--precision", default="", help="e.g. BF16 or FP8, for run.json")
    parser.add_argument("--kind", choices=["chat", "granite"], default="chat")
    parser.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False to the chat template")
    parser.add_argument("--methods", nargs="+", required=True, choices=sorted(METHODS))
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--limit", type=int, help="first N items per dataset (smoke tests)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data", default="data")
    parser.add_argument("--assets", default="data/methods")
    parser.add_argument("--retrieval", default="results/retrieval")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    client = Client(args.base_url, args.model, {"enable_thinking": False} if args.no_thinking else None)
    out = Path(args.out) / args.slug
    out.mkdir(parents=True, exist_ok=True)
    run_file = out / "run.json"
    run = json.loads(run_file.read_text()) if run_file.exists() else {}
    run |= {
        "display": args.display,
        "model_id": args.model_id,
        "precision": args.precision,
        "kind": args.kind,
        "chat_template_kwargs": client.template_kwargs,
        "decoding": "temperature 0, top_p 1, seed 0",
        "date": date.today().isoformat(),
    }
    run_file.write_text(json.dumps(run, indent=2) + "\n")

    for method in args.methods:
        load, predict = METHODS[method]
        state = load(data_dir=args.data, assets_dir=args.assets, retrieval_dir=args.retrieval, model_kind=args.kind)
        for dataset in args.datasets:
            items = load_sample(dataset, args.data)[: args.limit]
            dest = out / method / f"{dataset}.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            todo = [it for it in items if it["id"] not in done_ids(dest)]
            start = time.time()

            def run_one(it):
                try:
                    return it | predict(it, client, state)
                except Exception as e:  # keep going; the failure is recorded and counted as unparsed
                    return it | {"pred_biased": None, "score": None, "types": [], "parse_ok": False, "error": repr(e)}

            with ThreadPoolExecutor(args.workers) as pool, open(dest, "a", encoding="utf-8") as f:
                for row in pool.map(run_one, todo):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{args.slug} {method} {dataset}: {len(todo)} new items in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
