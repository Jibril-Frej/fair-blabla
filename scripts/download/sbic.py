"""SBIC: Social Bias Inference Corpus (Sap et al., ACL 2020).

Source: maartensap.com/social-bias-frames — license CC BY 4.0.
"""
import tarfile

from _common import describe_csv, fetch, out_dir

URL = "https://maartensap.com/social-bias-frames/SBIC.v2.tgz"

out = out_dir("data/sbic")
archive = fetch(URL, out / "SBIC.v2.tgz")
with tarfile.open(archive) as tar:
    tar.extractall(out, filter="data")
for path in sorted(out.glob("*.csv")):
    print(path.name)
    describe_csv(path, label_col="offensiveYN")
