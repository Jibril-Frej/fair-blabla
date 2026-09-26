"""Draw a fixed random sample of rows from each dataset's test set.

Run the download scripts first. Samples are written to data/samples/<dataset>.csv
with all original columns kept.

Usage: python scripts/sample_test_sets.py [--n 50] [--seed 42] [--data data]
"""
import argparse
import csv
import random
from pathlib import Path

# dataset name -> test file (relative to the data dir), as listed in the README
TEST_FILES = {
    "emgsd": "emgsd/test.csv",
    "stereodetect": "stereodetect/test.csv",
    "sbic": "sbic/SBIC.v2.agg.tst.csv",
    "crows_pairs": "crows_pairs/crows_pairs_anonymized.csv",
    "toxigen": "toxigen/annotated_test.csv",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="rows per dataset (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--data", default="data", help="data directory (default: data)")
    args = parser.parse_args()

    csv.field_size_limit(10**8)
    data = Path(args.data)
    out = data / "samples"
    out.mkdir(parents=True, exist_ok=True)

    for name, rel in TEST_FILES.items():
        fieldnames, rows = read_csv(data / rel)
        # One RNG per dataset so each sample doesn't depend on the others.
        sample = random.Random(f"{args.seed}-{name}").sample(rows, args.n)
        dest = out / f"{name}.csv"
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample)
        print(f"{name}: {len(sample)}/{len(rows)} rows -> {dest}")


if __name__ == "__main__":
    main()
