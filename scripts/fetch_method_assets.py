"""Download the prompts and databases of the bias detection methods into data/methods/.

Files come from the authors' repositories at fixed commits:
- Görge et al. (Apache-2.0): prompt P_F_01 -> data/methods/goerge/
- BiasAlert (no license stated): bias database and instruction template -> data/methods/biasalert/

Usage: python scripts/fetch_method_assets.py [--out data/methods]
"""
import argparse
import json
import urllib.request
from pathlib import Path

GOERGE = (
    "https://raw.githubusercontent.com/r-goerge/Detecting-Linguistic-Indicators-for-Stereotype-Assessment-with-LLMs/"
    "ee3b8b9bd2ec22bb83699a29c9f741b78df86832/"
)
BIASALERT = "https://raw.githubusercontent.com/FanZT6/BiasAlert/65cb15c2b63301095fb7c7952c169d9dec7767a4/"


def fetch(url, dest):
    if dest.exists():
        print(f"skip {dest} (exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())
    print(f"saved {dest} ({dest.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/methods", help="output directory (default: data/methods)")
    out = Path(parser.parse_args().out)

    fetch(GOERGE + "src/prompts/Prompts_to_detect_linguistic_indicators.csv",
          out / "goerge/Prompts_to_detect_linguistic_indicators.csv")
    fetch(BIASALERT + "data/retrieval/bias_doc.tsv", out / "biasalert/bias_doc.tsv")

    # The instruction is the same in every processed RedditBias example; keep it as plain text.
    instruction = out / "biasalert/instruction.txt"
    if not instruction.exists():
        with urllib.request.urlopen(BIASALERT + "data/precessed_redditbias/test_data/race.json", timeout=120) as r:
            examples = json.load(r)
        texts = {e["instruction"] for e in examples}
        assert len(texts) == 1, "expected a single instruction template"
        instruction.write_text(texts.pop(), encoding="utf-8")
        print(f"saved {instruction}")


if __name__ == "__main__":
    main()
