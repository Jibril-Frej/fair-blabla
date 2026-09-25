"""CrowS-Pairs (Nangia et al., EMNLP 2020).

Source: github.com/nyu-mll/crows-pairs — license CC BY-SA 4.0.
Each row is a pair: sent_more (more stereotypical) / sent_less.
"""
from _common import describe_csv, fetch, out_dir

BASE = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"

out = out_dir("data/crows_pairs")
fetch(BASE + "README.md", out / "README.md")
path = fetch(BASE + "data/crows_pairs_anonymized.csv", out / "crows_pairs_anonymized.csv")
describe_csv(path, label_col="bias_type")
