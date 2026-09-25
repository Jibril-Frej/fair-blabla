"""ToxiGen, human-annotated subset (Hartvigsen et al., ACL 2022).

Source: huggingface.co/datasets/toxigen/toxigen-data — data under
CDLA-Permissive-2.0 (see github.com/microsoft/TOXIGEN). Only the annotated
files are downloaded, not the 274k machine-generated prompts.
"""
from _common import describe_csv, fetch, out_dir

BASE = "https://huggingface.co/datasets/toxigen/toxigen-data/resolve/main/"

out = out_dir("data/toxigen")
fetch(BASE + "README.md", out / "README.md")
for name in ["annotated_train.csv", "annotated_test.csv"]:
    path = fetch(BASE + name, out / name)
    describe_csv(path, label_col="target_group")
