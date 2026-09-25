"""Shared helpers for the download scripts (standard library only)."""
import argparse
import csv
import urllib.request
from pathlib import Path


def out_dir(default):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=default, help=f"output directory (default: {default})")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch(url, dest):
    """Download url to dest unless it already exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    print(f"OK {dest}")
    return dest


def describe_csv(path, label_col=None):
    """Print row count, columns and (optionally) label distribution."""
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} rows; columns: {reader.fieldnames}")
    if label_col and label_col in reader.fieldnames:
        counts = {}
        for r in rows:
            counts[r[label_col]] = counts.get(r[label_col], 0) + 1
        print(f"  {label_col}: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
