"""Check the results files independently of the pipeline code.

1. Gold labels: re-derive text, binary label and CrowS-Pairs direction from the raw sample CSVs
   (data/samples/*.csv) and compare with every results file.
2. Predictions: recompute each method's decision (and types) from the raw model output and compare
   with pred_biased / types.

Usage: python scripts/audit_results.py   (prints the mismatches; "none" means everything agrees)
"""
import csv, json, glob, re, collections
S = "data/samples/"
def rd(n): return list(csv.DictReader(open(S + n + ".csv", newline="", encoding="utf-8")))
exp = {}  # (dataset, id) -> (text, gold_biased, stereo)
for i, r in enumerate(rd("stereodetect")): exp[("stereodetect", i)] = (r["Sentence"], 1 if r["labels"] in ("1", "4") else 0, None)
for i, r in enumerate(rd("sbic")): exp[("sbic", i)] = (r["post"], 1 if float(r["hasBiasedImplication"]) == 0 else 0, None)
for i, r in enumerate(rd("toxigen")): exp[("toxigen", i)] = (r["text"], 1 if float(r["toxicity_human"]) >= 3 else 0, None)
for i, r in enumerate(rd("crows_pairs")):
    st = r["stereo_antistereo"] == "stereo"
    exp[("crows_pairs", f"{i}-more")] = (r["sent_more"], None, st)
    exp[("crows_pairs", f"{i}-less")] = (r["sent_less"], None, not st)
bad = collections.Counter(); files = 0; seen = collections.Counter()
for p in glob.glob("results/*/*/*.jsonl"):
    if "retrieval" in p: continue
    files += 1; d = p.split("/")[-1][:-6]
    ids = set()
    for l in open(p):
        r = json.loads(l); k = (d, r["id"] if d == "crows_pairs" else int(r["id"]))
        ids.add(k)
        t, g, st = exp[k]
        if r["text"] != t: bad["text", d] += 1
        if r["gold_biased"] != g: bad["gold_biased", d] += 1
        if d == "crows_pairs" and r["stereo"] != st: bad["stereo", d] += 1
    if ids != {k for k in exp if k[0] == d}: bad["missing ids", p] += 1
    seen[d] += 1
print("gold: files", files, dict(seen)); print("mismatches:", dict(bad) or "none")
# label distribution and cross-tab of raw category columns
for n, col in [("stereodetect", "labels"), ("sbic", "hasBiasedImplication"), ("crows_pairs", "stereo_antistereo")]:
    print(n, collections.Counter(r[col] for r in rd(n)))
print("toxigen >=3:", sum(float(r["toxicity_human"]) >= 3 for r in rd("toxigen")), "/ 50")

AX = {"S1": "gender", "S2": "sexual_orientation", "S3": "disability", "S4": "age", "S5": "race_ethnicity",
      "S6": "nationality", "S7": "religion", "S8": "socioeconomic", "S9": "appearance"}
bad = collections.Counter(); n = collections.Counter(); ex = {}
for p in sorted(glob.glob("results/*/*/*.jsonl")):
    if "retrieval" in p: continue
    run, m, d = p.split("/")[1:4]
    rows = {}
    for l in open(p):
        r = json.loads(l); rows[r["id"]] = r
    for r in rows.values():
        n[m] += 1; pred = r["pred_biased"]; types = set(r.get("types") or [])
        if not r["parse_ok"]:
            bad[m, "parse_fail"] += 1; continue
        if m == "demographic_axes":
            codes = re.findall(r"S10|S[1-9]", r["raw"])
            exp_t = {AX[c] for c in codes if c != "S10"}
            ok = pred == int(bool(exp_t)) and types == exp_t
        elif m == "guardian_criteria":
            exp_t = {t for t, v in r["p_yes"].items() if v > 0.5}
            ok = pred == int(bool(exp_t)) and types == exp_t and abs(r["score"] - max(r["p_yes"].values())) < 1e-9
        elif m == "guardian_social_bias":
            ok = pred == int(r["score"] > 0.5)
        elif m == "linguistic_indicators":
            ok = pred == int(r["score"] > 0.5)
        elif m == "biasalert_rag":
            if run.endswith("structured"):
                v = json.loads(r["raw"])["biased"]
            else:
                v = re.findall(r"(Yes|No), the following SENTENCE", r["raw"])[-1]
            ok = pred == int(v == "Yes")
        if not ok:
            bad[m, "mismatch"] += 1; ex.setdefault(m, (run, d, r["id"], r.get("raw") if isinstance(r.get("raw"), str) else "", pred, types))
print("predictions: answers", dict(n)); print("predictions: problems", dict(bad) or "none"); [print(k, v) for k, v in ex.items()]
