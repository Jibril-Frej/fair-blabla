"""StereoDetect (Shejole & Bhattacharyya, Findings of EMNLP 2025).

Source: github.com/KaustubhShejole/StereoDetect — no license stated in the
repo: check with the authors before redistributing.
"""
from _common import describe_csv, fetch, out_dir

BASE = "https://raw.githubusercontent.com/KaustubhShejole/StereoDetect/main/"

out = out_dir("data/stereodetect")
fetch(BASE + "README.md", out / "README.md")
for split in ["train", "val", "test"]:
    path = fetch(BASE + f"dataset/{split}.csv", out / f"{split}.csv")
    describe_csv(path, label_col="labels")
