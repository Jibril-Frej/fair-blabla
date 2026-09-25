"""EMGSD: Expanded Multi-Grain Stereotype Dataset (King et al., HEARTS, IJCNLP-AACL 2025).

Source: huggingface.co/datasets/holistic-ai/EMGSD — license MIT.
"""
from _common import describe_csv, fetch, out_dir

BASE = "https://huggingface.co/datasets/holistic-ai/EMGSD/resolve/main/"

out = out_dir("data/emgsd")
for name in ["README.md", "train.csv", "test.csv"]:
    path = fetch(BASE + name, out / name)
    if name.endswith(".csv"):
        describe_csv(path, label_col="label")
