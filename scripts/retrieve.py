"""Retrieve the in-context examples / references for the retrieval-based methods.

- demographic_axes: top-10 examples of data.build_axes_pool() by cosine similarity of BAAI/bge-m3
  dense embeddings (CLS token, normalised), as in Majumdar et al.
- biasalert: top-10 entries of BiasAlert's bias_doc.tsv by inner product of
  facebook/contriever-msmarco embeddings (mean pooling), as in BiasAlert's retrieval code.

Writes results/retrieval/<method>.jsonl with one line per item: dataset, id, neighbours (ids), scores.
Needs torch + transformers and preferably a GPU.

Usage: python scripts/retrieve.py [--data data] [--assets data/methods] [--out results/retrieval]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fairblabla.data import build_axes_pool, load_all  # noqa: E402

TOP = 10


@torch.no_grad()
def embed(model_id, texts, pooling, batch=256, max_length=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        if pooling == "cls":
            e = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
        else:  # mean over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            e = (hidden * mask).sum(1) / mask.sum(1)
        out.append(e.float().cpu())
        print(f"{model_id}: {min(i + batch, len(texts))}/{len(texts)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return torch.cat(out)


def search(queries, keys, ids, dest, items):
    scores, idx = (queries @ keys.T).topk(TOP, dim=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for it, s, j in zip(items, scores.tolist(), idx.tolist()):
            row = {"dataset": it["dataset"], "id": it["id"], "neighbours": [ids[k] for k in j], "scores": [round(x, 4) for x in s]}
            f.write(json.dumps(row) + "\n")
    print(f"saved {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--assets", default="data/methods")
    parser.add_argument("--out", default="results/retrieval")
    args = parser.parse_args()
    out = Path(args.out)

    items = load_all(args.data)
    queries = [it["text"] for it in items]

    pool = build_axes_pool(args.data)
    print(f"demographic_axes pool: {len(pool)} examples", file=sys.stderr)
    q = embed("BAAI/bge-m3", queries, "cls")
    k = embed("BAAI/bge-m3", [p["text"] for p in pool], "cls")
    search(q, k, [p["pool_id"] for p in pool], out / "demographic_axes.jsonl", items)

    with open(Path(args.assets) / "biasalert/bias_doc.tsv", encoding="utf-8", newline="") as f:
        docs = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    q = embed("facebook/contriever-msmarco", queries, "mean")
    # contriever's generate_passage_embeddings.py embeds "<title> <text>" by default
    k = embed("facebook/contriever-msmarco", [f"{d['title']} {d['text']}" for d in docs], "mean")
    search(q, k, [d["id"] for d in docs], out / "biasalert.jsonl", items)


if __name__ == "__main__":
    main()
