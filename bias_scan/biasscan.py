#!/usr/bin/env python3
"""biasscan -- measure how a text corpus portrays people.

Reports gender representation, gender x occupation, gender x activity, and
coverage of ethnicity / class / appearance terms, with significance tests.

This measures the CORPUS, not a model. A skew here is what a model trained on
this data would absorb, not what any model does.

Usage:
    python biasscan.py --file corpus.csv --lang en --audit     # check lexicon first
    python biasscan.py --file corpus.csv --lang en
    python biasscan.py --file corpus.csv --lang fr --names names.csv

Language knowledge lives entirely in lang_*.py. Adding a language means
writing a word list, not touching this file.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, mannwhitneyu

GENDERS = ("M", "F")


# ===========================================================================
#  Language pack
# ===========================================================================
@dataclass
class Lang:
    name: str
    # "word" wraps terms in \b...\b (English, French, most scripts).
    # "char" matches as plain substrings (Chinese, Japanese, Thai -- no spaces).
    segmentation: str = "word"
    # derive "teachers"/"families"/"policemen" from singulars. English only;
    # leave off for languages with real inflection.
    auto_plural: bool = False
    # proximity window in CHARACTERS, not words. ~6-char words -> 40.
    window: int = 40
    # True where "," is the decimal separator (fr, de, es, ru). Getting this
    # wrong is a silent 100x error: "12,50" would parse as 1250.
    decimal_comma: bool = False

    pronouns: dict = field(default_factory=dict)      # {"M": [...], "F": [...]}
    kinship: dict = field(default_factory=dict)       # kinship + gendered roles
    occupations: dict = field(default_factory=dict)   # {category: [words]}
    activities: dict = field(default_factory=dict)
    dimensions: dict = field(default_factory=dict)    # ethnicity, class, looks...
    # Terms that are social descriptors in prose but measurements in technical
    # text ("height" is geometry in a maths problem, body size in a novel).
    # Counted only with --include-ambiguous. A domain choice, not a language one.
    ambiguous: dict = field(default_factory=dict)
    # one or more capture groups yielding a number; None skips the money section
    money: str | None = None
    money_unit: str = ""

    # names are supplied by the user at run time, never shipped
    names: dict = field(default_factory=lambda: {"M": [], "F": [], "A": []})


def _plural(t: str) -> list[str]:
    if " " in t:
        head, _, last = t.rpartition(" ")
        return [t] + [f"{head} {p}" for p in _plural(last)[1:]]
    if re.search(r"(s|sh|ch|x|z)$", t):
        return [t, t + "es"]
    if re.search(r"[^aeiou]y$", t):
        return [t, t[:-1] + "ies"]
    if t.endswith("man"):
        return [t, t[:-3] + "men"]
    return [t, t + "s"]


class Matchers:
    """Compiled regexes for one Lang. Build once, reuse."""

    def __init__(self, L: Lang, include_ambiguous: bool = False):
        self.L = L
        self.dims = dict(L.dimensions)
        if include_ambiguous:
            for k, extra in L.ambiguous.items():
                self.dims[k] = sorted(set(self.dims.get(k, [])) | set(extra))

        self.pron = {g: self._c(L.pronouns.get(g, [])) for g in GENDERS}
        self.kin = {g: self._c(L.kinship.get(g, [])) for g in GENDERS}
        self.name = {g: self._c(L.names.get(g, [])) for g in ("M", "F", "A")}
        self.act = {k: self._c(v) for k, v in L.activities.items()}
        self.dim = {k: self._c(v) for k, v in self.dims.items()}
        self.money = re.compile(L.money) if L.money else None

        # One combined regex for every occupation form, plus a map back to
        # (canonical word, category). Matching each word separately was 35x
        # slower on a 200k-document corpus.
        self.occ_of: dict[str, tuple[str, str]] = {}
        forms = []
        for cat, words in L.occupations.items():
            for w in words:
                for f in self._expand([w]):
                    self.occ_of.setdefault(f.lower(), (w, cat))
                    forms.append(f)
        self.occ = self._c(forms)

    def _expand(self, terms):
        out = []
        for t in terms:
            out.extend(_plural(t) if self.L.auto_plural else [t])
        # longest first: "mother" must win over "other", 妈妈 over 妈
        return sorted(set(out), key=lambda x: (-len(x), x))

    def _c(self, terms):
        terms = self._expand(terms)
        if not terms:
            return None
        body = "|".join(re.escape(t) for t in terms)
        pat = f"(?:{body})"
        if self.L.segmentation == "word":
            pat = rf"\b{pat}\b"
        return re.compile(pat, re.IGNORECASE)

    def gender_of(self, text: str, mask: tuple[int, int] | None = None):
        """Is a man / a woman referred to here?

        `mask` blanks a span before looking. Needed because the span being
        measured must not supply its own gender: French "infirmiere" IS
        feminine and English "saleswoman" contains "woman", so without the
        mask, occupation-vs-gender would just rediscover grammar.
        """
        if mask:
            a, b = mask
            text = text[:a] + ("\x00" * (b - a)) + text[b:]
        out = []
        for g in GENDERS:
            out.append(any(m is not None and m.search(text)
                           for m in (self.name[g], self.kin[g], self.pron[g])))
        return out[0], out[1]


# ===========================================================================
#  Extraction
# ===========================================================================
def extract(texts, M: Matchers) -> pd.DataFrame:
    """One row per document: which signals it contains."""
    rows = []
    for t in texts:
        t = t or ""
        r = {"text": t}
        for g in ("M", "F", "A"):
            r[f"name_{g}"] = M.name[g].findall(t) if M.name[g] else []
        for g in GENDERS:
            r[f"kin_{g}"] = sorted(set(M.kin[g].findall(t))) if M.kin[g] else []
            r[f"pron_{g}"] = len(M.pron[g].findall(t)) if M.pron[g] else 0
        r["has_M"] = bool(r["name_M"] or r["kin_M"] or r["pron_M"])
        r["has_F"] = bool(r["name_F"] or r["kin_F"] or r["pron_F"])
        words, cats = set(), set()
        if M.occ:
            for mo in M.occ.finditer(t):
                hit = M.occ_of.get(mo.group(0).lower())
                if hit:
                    words.add(hit[0])
                    cats.add(hit[1])
        r["occ"], r["occ_words"] = sorted(cats), sorted(words)
        r["act"] = sorted(k for k, m in M.act.items() if m and m.search(t))
        for k, m in M.dim.items():
            r[f"dim_{k}"] = sorted(set(m.findall(t))) if m else []
        rows.append(r)
    return pd.DataFrame(rows)


# ===========================================================================
#  Statistics
# ===========================================================================
def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _odds(a, b, c, d):
    """Haldane-corrected odds ratio, so a zero cell gives a finite number."""
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a / c) / (b / d)


def _crosstab(m_docs, f_docs, col, keys, min_n=15):
    nm, nf = len(m_docs), len(f_docs)
    rows = []
    for k in keys:
        a = int(m_docs[col].map(lambda l: k in l).sum())
        b = int(f_docs[col].map(lambda l: k in l).sum())
        if a + b < min_n:
            continue
        c, d = nm - a, nf - b
        try:
            p = float(chi2_contingency([[a, b], [c, d]])[1])
        except Exception:
            p = float("nan")
        rows.append(dict(key=k, m=a, f=b, m_rate=a / max(nm, 1),
                         f_rate=b / max(nf, 1), odds=_odds(a, b, c, d), p=p))
    return sorted(rows, key=lambda r: -r["odds"])


def _number(raw, decimal_comma):
    s = raw.strip().replace(" ", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".") if decimal_comma else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def analyze(sig, M: Matchers, label="corpus"):
    """Return the report as a list of lines, plus a stats dict."""
    L = M.L
    out, data = [], {}
    P = out.append
    N = len(sig)
    P(f"\n{'=' * 72}\n{label}   lang: {L.name}   documents: {N:,}\n{'=' * 72}")

    # ---- 1. representation ------------------------------------------------
    P("\n1. GENDER REPRESENTATION")
    for lbl, m, f in (("names", sig.name_M.map(len).sum(), sig.name_F.map(len).sum()),
                      ("kinship/roles", sig.kin_M.map(len).sum(), sig.kin_F.map(len).sum()),
                      ("pronouns", sig.pron_M.sum(), sig.pron_F.sum())):
        m, f = int(m), int(f)
        p = binomtest(m, m + f, 0.5).pvalue if m + f else float("nan")
        P(f"   {lbl:14s} M {m:8,}  F {f:8,}   M:F = {m / max(f, 1):6.2f}   p={p:.1e}")
        data[lbl] = dict(M=m, F=f)
    qm, qf = int(sig.has_M.sum()), int(sig.has_F.sum())
    P(f"   docs with a man {qm:,} ({qm / max(N, 1):.1%})   "
      f"a woman {qf:,} ({qf / max(N, 1):.1%})")
    data["docs"] = dict(N=N, M=qm, F=qf)

    only_m = sig[sig.has_M & ~sig.has_F]
    only_f = sig[sig.has_F & ~sig.has_M]
    if not len(only_m) or not len(only_f):
        P("\n   too few single-gender documents for the cross-tabs")
        return out, data
    P(f"   single-gender docs used below: M {len(only_m):,} / F {len(only_f):,}")

    def table(title, rows, c1="M docs", c2="F docs"):
        P(f"\n{title}")
        P(f"   {'':22s} {c1:>8s} {c2:>8s} {'M %':>8s} {'F %':>8s} {'odds':>7s} {'p':>9s}")
        for r in rows:
            P(f"   {r['key']:22s} {r['m']:8d} {r['f']:8d} {r['m_rate']:8.2%} "
              f"{r['f_rate']:8.2%} {r['odds']:7.2f} {r['p']:9.1e}{_stars(r['p'])}")

    # ---- 2. occupation ----------------------------------------------------
    rows = _crosstab(only_m, only_f, "occ", L.occupations)
    table("2. OCCUPATION x GENDER  (odds > 1 = male-skewed)", rows)
    data["occupation"] = rows

    # ---- 3. occupation word, masked proximity -----------------------------
    P(f"\n3. OCCUPATION WORD x GENDER  (gender marker within {L.window} chars,")
    P("   with the occupation word itself masked -- the honest version of 2)")
    cnt = collections.Counter()
    if M.occ:
        W = L.window
        for t in sig.text:
            for mo in M.occ.finditer(t):
                hit = M.occ_of.get(mo.group(0).lower())
                if not hit:
                    continue
                lo = max(0, mo.start() - W)
                seg = t[lo: mo.end() + W]
                hm, hf = M.gender_of(seg, mask=(mo.start() - lo, mo.end() - lo))
                if hm != hf:
                    cnt[(hit[0], "M" if hm else "F")] += 1
    tot = collections.Counter()
    for (w, g), n in cnt.items():
        tot[w] += n
    wrows = []
    for w, _ in tot.most_common(40):
        m, f = cnt[(w, "M")], cnt[(w, "F")]
        if m + f >= 8:
            wrows.append(dict(word=w, m=m, f=f, ratio=m / max(f, 1),
                              p=float(binomtest(m, m + f, 0.5).pvalue)))
    wrows.sort(key=lambda r: -r["ratio"])
    P(f"   {'':22s} {'M':>8s} {'F':>8s} {'M:F':>8s} {'p':>9s}")
    for r in wrows:
        P(f"   {r['word']:22s} {r['m']:8d} {r['f']:8d} {r['ratio']:8.2f} "
          f"{r['p']:9.1e}{_stars(r['p'])}")
    data["occupation_word"] = wrows

    # ---- 4. activity ------------------------------------------------------
    rows = _crosstab(only_m, only_f, "act", L.activities)
    table("4. ACTIVITY x GENDER", rows)
    data["activity"] = rows

    # ---- 5. mention order -------------------------------------------------
    both = sig[sig.has_M & sig.has_F]
    fm = ff = 0
    for t in both.text:
        pos = {}
        for g in GENDERS:
            c = [mo.start() for mo in
                 (mm.search(t) for mm in (M.name[g], M.kin[g], M.pron[g]) if mm)
                 if mo]
            pos[g] = min(c) if c else 10 ** 9
        if pos["M"] < pos["F"]:
            fm += 1
        elif pos["F"] < pos["M"]:
            ff += 1
    if fm + ff:
        p = binomtest(fm, fm + ff, 0.5).pvalue
        P(f"\n5. MENTION ORDER  (when both appear, who comes first)")
        P(f"   man first {fm:,} ({fm / (fm + ff):.1%})   "
          f"woman first {ff:,} ({ff / (fm + ff):.1%})   p={p:.1e}{_stars(p)}")
        data["order"] = dict(M_first=fm, F_first=ff, p=float(p))

    # ---- 6. money ---------------------------------------------------------
    if M.money:
        def amounts(df):
            v = []
            for t in df.text:
                for x in M.money.findall(t):
                    groups = (x,) if isinstance(x, str) else x
                    raw = next((g for g in groups if g), None)
                    if raw is not None:
                        n = _number(raw, L.decimal_comma)
                        if n is not None:
                            v.append(n)
            return np.array(v)

        am, af = amounts(only_m), amounts(only_f)
        if len(am) > 30 and len(af) > 30:
            u = mannwhitneyu(am, af, alternative="two-sided")
            P(f"\n6. MONEY AMOUNTS x GENDER  ({L.money_unit})")
            for lbl, v in (("men", am), ("women", af)):
                P(f"   {lbl:7s} n={len(v):7,}  median {np.median(v):12,.1f}  "
                  f"mean {v.mean():12,.1f}")
            P(f"   Mann-Whitney p={u.pvalue:.1e}   "
              f"median ratio M/F = {np.median(am) / max(np.median(af), 1e-9):.2f}")
            data["money"] = dict(M_median=float(np.median(am)),
                                 F_median=float(np.median(af)), p=float(u.pvalue))

    # ---- 7. dimension coverage -------------------------------------------
    P("\n7. OTHER DIMENSIONS  (how often the corpus mentions them at all)")
    for k in M.dims:
        col = f"dim_{k}"
        n = int(sig[col].map(len).gt(0).sum())
        top = collections.Counter(x for l in sig[col] for x in l).most_common(5)
        P(f"   {k:16s} {n:8,} docs ({n / max(N, 1):7.3%})  {top}")
        data.setdefault("dimensions", {})[k] = dict(n=n, rate=n / max(N, 1), top=top)
    return out, data


# ===========================================================================
#  Lexicon audit -- run this before trusting any number
# ===========================================================================
def audit(texts, M: Matchers, top=8):
    """Per-term hit counts with example contexts, to expose false positives.

    Corpora reuse everyday words for other things. "height" is geometry in a
    maths problem; "white" is a marble colour; "pretty" is an adverb in
    "pretty hard"; "rent" is a verb in "rent the video". Every one of those was
    a real false positive found with this function. If a category's top hit
    shows a colour, a unit or an adverb in context, drop the term.
    """
    out = []
    P = out.append
    P(f"\n{'=' * 72}\nLEXICON AUDIT   {M.L.name}   {len(texts):,} documents\n{'=' * 72}")
    P("A term taking >60% of its category is flagged. Read the contexts: if a")
    P("term is not actually about people here, remove it from the lang file.")

    groups = [("pronouns", M.L.pronouns), ("kinship", M.L.kinship),
              ("names", M.L.names), ("occupations", M.L.occupations),
              ("activities", M.L.activities), ("dimensions", M.dims)]
    for gname, cats in groups:
        cats = {k: v for k, v in cats.items() if v}
        if not cats:
            continue
        P(f"\n--- {gname} " + "-" * (66 - len(gname)))
        for cat, terms in cats.items():
            counts, example = collections.Counter(), {}
            pats = [(t, M._c([t])) for t in terms]
            pats = [(t, p) for t, p in pats if p]
            for text in texts:
                for t, p in pats:
                    mo = p.search(text)
                    if mo:
                        counts[t] += len(p.findall(text))
                        if t not in example:
                            a, b = max(0, mo.start() - 32), mo.end() + 32
                            example[t] = re.sub(r"\s+", " ", text[a:b])
            total = sum(counts.values())
            live = sum(1 for v in counts.values() if v)
            P(f"\n  [{cat}]  {total:,} hits, {live}/{len(terms)} terms match")
            if not total:
                P("    no hits -- absent from this corpus, or the wrong words")
                continue
            for t, n in counts.most_common(top):
                flag = "   <-- dominates category" if n / total > 0.6 and live > 1 else ""
                P(f"    {n:7,} {n / total:6.1%}  {t!r}{flag}")
                P(f"            {example.get(t, '')}")
            dead = [t for t in terms if not counts[t]]
            if dead:
                P(f"    never matched ({len(dead)}): {', '.join(map(repr, dead[:10]))}")
    return out


# ===========================================================================
#  Names -- supplied by the user, never shipped
# ===========================================================================
ALIASES = {"M": {"m", "male", "man", "homme", "h", "boy", "1"},
           "F": {"f", "female", "woman", "femme", "girl", "2"},
           "A": {"a", "ambiguous", "unisex", "neutral", "both", "x", "0"}}
_ALIAS = {a: g for g, s in ALIASES.items() for a in s}


def read_names(path):
    """Load a name list. Two formats:

    CSV  with a header:          name,gender
                                 Jean,M
                                 Camille,unisex
    JSON keyed by gender:        {"M": ["Jean"], "F": ["Marie"], "A": ["Camille"]}

    Gender labels are case-insensitive and accept aliases. An unrecognised
    label raises rather than being skipped, so a typo cannot quietly shrink
    the list. Put unisex names in "A": they count as a person but give no
    gender signal.
    """
    def canon(raw):
        k = str(raw).strip().lower()
        if k in _ALIAS:
            return _ALIAS[k]
        raise SystemExit(f"unknown gender label {raw!r}; use M / F / A")

    out = {"M": [], "F": [], "A": []}
    text = open(path, encoding="utf8").read()
    if text.lstrip().startswith("{"):
        for k, v in json.loads(text).items():
            if not k.startswith("_"):            # "_comment" keys are ignored
                out[canon(k)].extend(str(x).strip() for x in v if str(x).strip())
    else:
        for row in csv.DictReader(text.splitlines()):
            low = {k.lower(): v for k, v in row.items() if k}
            nm = low.get("name") or low.get("first_name") or low.get("prenom")
            gd = low.get("gender") or low.get("sex") or low.get("genre")
            if not nm or not gd:
                raise SystemExit("CSV needs a 'name' and a 'gender' column")
            out[canon(gd)].append(nm.strip())
    if not any(out.values()):
        raise SystemExit(f"{path}: no names found")
    return out


def suggest_names(texts, M: Matchers, top=120):
    """Rank name candidates for YOU to label. Nothing is labelled for you.

    Lists mid-sentence capitalised words not already covered by the lang file.
    Sentence-initial capitals are skipped, which removes most of the noise.
    Place names and brands will still appear -- drop them yourself.
    """
    if M.L.segmentation == "char":
        raise SystemExit("--suggest-names needs a language with capitalisation")
    known = M._c([w for d in (M.L.occupations, M.L.activities, M.dims)
                  for ws in d.values() for w in ws]
                 + M.L.kinship.get("M", []) + M.L.kinship.get("F", [])
                 + M.L.pronouns.get("M", []) + M.L.pronouns.get("F", []))
    tok = re.compile(r"[^\W\d_][\w'’-]+", re.UNICODE)
    counts = collections.Counter()
    for t in texts:
        for mo in tok.finditer(t):
            w = mo.group(0)
            if len(w) < 3 or not w[:1].isupper() or w.isupper():
                continue
            before = t[:mo.start()].rstrip()
            if not before or before[-1] in ".!?:;…\"'()[]":
                continue                          # sentence-initial: skip
            if known and known.search(w):
                continue
            counts[w] += 1
    return counts.most_common(top)


# ===========================================================================
#  Loading and CLI
# ===========================================================================
TEXT_COLS = ["text", "question", "problem", "review", "content", "sentence",
             "query", "prompt", "body", "input"]


def load(path, column=None):
    paths = sorted(glob.glob(path)) if any(c in path for c in "*?[") else [path]
    if not paths:
        raise SystemExit(f"no files matched: {path}")
    frames = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".jsonl", ".ndjson"):
            frames.append(pd.read_json(p, lines=True))
        elif ext == ".json":
            frames.append(pd.read_json(p))
        elif ext == ".parquet":
            frames.append(pd.read_parquet(p))
        elif ext == ".tsv":
            frames.append(pd.read_csv(p, sep="\t"))
        else:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if column:
        if column not in df.columns:
            raise SystemExit(f"no column {column!r}; have {list(df.columns)}")
        return df[column]
    for c in TEXT_COLS:
        if c in df.columns:
            return df[c]
    strs = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    if not strs:
        raise SystemExit(f"no text column found in {list(df.columns)}")
    return df[max(strs, key=lambda c: df[c].astype(str).str.len().mean())]


def get_lang(code):
    try:
        mod = __import__(f"lang_{code}")
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        have = sorted(f[5:-3] for f in os.listdir(here)
                      if f.startswith("lang_") and f.endswith(".py")
                      and f != "lang_template.py")
        raise SystemExit(f"no lang_{code}.py. available: {', '.join(have)}\n"
                         f"to add one: cp lang_template.py lang_{code}.py")
    return mod.LANG


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", required=True,
                    help="csv/tsv/jsonl/json/parquet, or a glob")
    ap.add_argument("--lang", default="en", help="language code (en, fr, ...)")
    ap.add_argument("--column", help="text column (auto-detected otherwise)")
    ap.add_argument("--names", help="your gendered name list (CSV or JSON)")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="count domain-ambiguous terms (height/white/pretty). "
                         "Right for prose, wrong for technical text")
    ap.add_argument("--audit", action="store_true",
                    help="check the lexicon for false positives, then exit")
    ap.add_argument("--suggest-names", action="store_true",
                    help="list name candidates for you to label, then exit")
    ap.add_argument("--limit", type=int, help="only the first N documents")
    ap.add_argument("--out", help="write the report and stats here (basename)")
    a = ap.parse_args(argv)

    series = load(a.file, a.column)
    if a.limit:
        series = series.head(a.limit)
    texts = series.fillna("").astype(str).tolist()

    L = get_lang(a.lang)
    if a.names:
        L.names = read_names(a.names)
        print(f"[names] M {len(L.names['M'])}  F {len(L.names['F'])}  "
              f"A {len(L.names['A'])}", file=sys.stderr)
    elif not a.suggest_names:
        print("[names] none given; gender comes from pronouns and kinship only. "
              "Use --suggest-names then --names to raise coverage.", file=sys.stderr)
    M = Matchers(L, include_ambiguous=a.include_ambiguous)
    print(f"[scan] {len(texts):,} documents, lang {a.lang}", file=sys.stderr)

    if a.suggest_names:
        print("name,count,gender")                 # gender left empty for you
        for w, n in suggest_names(texts, M):
            print(f"{w},{n},")
        return 0

    if a.audit:
        lines = audit(texts, M)
        print("\n".join(lines))
        if a.out:
            open(f"{a.out}_audit.txt", "w", encoding="utf8").write("\n".join(lines))
            print(f"\n-> {a.out}_audit.txt", file=sys.stderr)
        return 0

    sig = extract(texts, M)
    lines, data = analyze(sig, M, label=os.path.basename(a.file))
    print("\n".join(lines))

    if a.out:
        open(f"{a.out}.txt", "w", encoding="utf8").write("\n".join(lines))
        json.dump(data, open(f"{a.out}.json", "w", encoding="utf8"),
                  ensure_ascii=False, indent=1, default=float)
        # documents labelled by gender, for building counterfactual pairs
        sig["gender"] = np.where(sig.has_M & ~sig.has_F, "M",
                                 np.where(sig.has_F & ~sig.has_M, "F", "both"))
        sig[["gender", "occ_words", "act", "text"]].to_csv(
            f"{a.out}_labelled.csv", index=False)
        print(f"\n-> {a.out}.txt  {a.out}.json  {a.out}_labelled.csv",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
