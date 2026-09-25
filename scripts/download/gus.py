"""GUS: Generalizations, Unfairness, Stereotypes (Powers et al., arXiv 2410.08388).

arXiv-only (not peer-reviewed). Span-level labels.
Source: huggingface.co/datasets/ethical-spectacle/gus-dataset-v1 — license MIT.
"""
from _common import describe_csv, fetch, out_dir

BASE = "https://huggingface.co/datasets/ethical-spectacle/gus-dataset-v1/resolve/main/"

out = out_dir("data/gus")
fetch(BASE + "README.md", out / "README.md")
describe_csv(fetch(BASE + "gus-dataset-v1.csv", out / "gus-dataset-v1.csv"))
