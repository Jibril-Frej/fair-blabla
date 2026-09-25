"""Fifty Shades of Bias (Hada et al., EMNLP 2023).

Source: github.com/microsoft/fifty-shades-of-bias — license MIT.
fsb_final_scores.csv holds the graded human gender-bias score per text.
"""
from _common import describe_csv, fetch, out_dir

BASE = "https://raw.githubusercontent.com/microsoft/fifty-shades-of-bias/main/"
FILES = [
    "README.md",
    "LICENSE",
    "data/FSB/FSB_text.csv",
    "data/FSB/fsb_final_scores.csv",
    "data/FSB/fsb-tuples-annotations.csv",
]

out = out_dir("data/fifty_shades_of_bias")
for name in FILES:
    path = fetch(BASE + name, out / name.split("/")[-1])
    if name.endswith(".csv"):
        describe_csv(path)
