"""Run every download script in this folder, one after the other.

Usage: python scripts/download/all.py
A failing dataset is reported and skipped; the others still run.
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
scripts = sorted(p for p in here.glob("*.py") if p.name not in {"all.py", "_common.py"})

failed = []
for script in scripts:
    print(f"\n=== {script.stem} ===", flush=True)
    if subprocess.run([sys.executable, str(script)]).returncode != 0:
        failed.append(script.stem)

print(f"\nDone: {len(scripts) - len(failed)}/{len(scripts)} datasets downloaded.")
if failed:
    sys.exit(f"Failed: {', '.join(failed)}")
