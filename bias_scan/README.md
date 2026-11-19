# biasscan

Measure how a text corpus portrays people: gender representation, gender ×
occupation, gender × activity, money amounts, and how often it mentions
ethnicity, class or appearance at all. With significance tests.

Four files. English and French included.

```
biasscan.py        the whole tool: matching, stats, audit, CLI
lang_en.py         English word lists
lang_fr.py         French word lists
lang_template.py   copy this to add a language
```

This measures **the corpus, not a model**. A skew here is what a model trained
on this data would absorb, not what any model does.

---

## Install

```bash
pip install pandas numpy scipy pyarrow
```

---

## Use

Three commands, in this order.

**1. Check the word lists against your corpus.** Do this first, always.

```bash
python biasscan.py --file corpus.csv --lang en --audit
```

**2. Optional: get name candidates to label.**

```bash
python biasscan.py --file corpus.csv --lang en --suggest-names > names.csv
```

**3. Scan.**

```bash
python biasscan.py --file corpus.csv --lang en --names names.csv --out result
```

Reads `csv`, `tsv`, `jsonl`, `json`, `parquet`, or a glob. The text column is
auto-detected (`text`, `question`, `review`, `content`, …) or set with
`--column`.

| flag | |
|---|---|
| `--lang` | `en` or `fr`, or any `lang_*.py` you add |
| `--names` | your gendered name list (CSV or JSON) |
| `--include-ambiguous` | count domain-risky terms — see below |
| `--audit` | check the word lists, then exit |
| `--suggest-names` | print name candidates, then exit |
| `--limit N` | only the first N documents |
| `--out NAME` | write `NAME.txt`, `NAME.json`, `NAME_labelled.csv` |

---

## Names are yours to supply

The language files ship **no names**. Which names read as gendered depends on
your corpus's era, region and community, so no shipped list is right in
general. You never have to edit Python.

```bash
python biasscan.py --file corpus.csv --lang fr --suggest-names > names.csv
```

This ranks **mid-sentence** capitalised words not already covered by the
language file — sentence-initial capitals are skipped, which removes most of
the noise. The `gender` column comes out **empty**:

```csv
name,count,gender
Marc,23,
Jean,21,
Marie,15,
France,26,
```

Nothing is labelled for you. Place names and brands (`France`) survive the
filter — delete them. Fill in `M`, `F` or `A`, then pass it back with
`--names`.

JSON works too:

```json
{"M": ["Jean"], "F": ["Marie"], "A": ["Camille"]}
```

Labels accept aliases (`male`/`homme`/`h`/`1`, `female`/`femme`/`2`,
`unisex`/`neutral`/`x`). An unrecognised label **raises** instead of being
skipped, so a typo cannot quietly shrink your list.

Put unisex names in **`A`**: they count as a person but give no gender signal.
Declining to guess beats guessing wrong.

Without names the tool still works — pronouns and kinship carry most of the
signal. Names matter most where people are named rather than described by role.

---

## Why `--audit` exists

Corpora reuse everyday words for other things. Every one of these was a real
false positive, found with `--audit`:

| word | what it actually was | damage |
|---|---|---|
| `height` | geometric height | 86% of appearance hits in maths problems |
| `white`, `black` | marble colours | 1,320 + 607 false ethnicity hits |
| `race` | "a 5,000-metre race" | 87% of ethnicity hits |
| `pretty` | "pretty hard", an adverb | 22% of appearance hits in film reviews |
| `poor` | "poor editing", a quality judgement | made `rural` read as 12.6% of IMDB |
| `rent` | "rent the video", a verb | 12% of socioeconomic |
| `interest` | "the interest of a narcoleptic" | 13% of socioeconomic |

`--audit` prints each word's hit count, its share of its category, and a real
example in context. It flags anything taking over 60% of its category. If the
context shows a colour, a unit or an adverb, delete the word from the language
file.

Run it on every new corpus. This is not a one-time setting — the shipped lists
were fixed twice each this way, and your corpus will break differently.

### The `--include-ambiguous` switch

Some words are genuine social descriptors in prose and measurements in
technical text. Those live in a separate `ambiguous` bucket, counted only when
you ask. Same word list, opposite verdict:

| | maths word problems | English CrowS-Pairs |
|---|---|---|
| `white` | 1,320 hits — marble colour | 340 hits — genuine ethnicity |
| ethnicity coverage | 0.002% | 23.4% |

Use `--include-ambiguous` for prose (reviews, news, dialogue, fiction). Leave
it off for technical text (maths, code, logs).

---

## Reading the output

Seven sections. Read them in this order of trust:

1. **Gender representation** — the headline. Three signal sources reported
   separately; if they disagree wildly, suspect the word lists.
2. **Occupation word × gender** (§3) — the honest occupational result. The
   occupation word is **masked out** before gender is read, so the finding is
   about the surrounding text, not the word itself.
3. **Activity × gender** (§4) — housework vs repair is usually the sharpest
   contrast.
4. **Occupation × gender** (§2) — document-level, so weaker than §3.
5. **Money, dimensions** (§6, §7) — only if coverage is non-trivial. A
   dimension at 0.00x% is an absence, not a measurement.

Odds ratios are Haldane-corrected (`+0.5` per cell), so a zero cell gives a
finite number. `*` p<0.05, `**` p<0.01, `***` p<0.001.

Things that will mislead you:

- **A `student` category near M:F 1.0 in school corpora** is a genre artifact
  ("30 boys and 28 girls"), not balance.
- **Low coverage is normal.** Only ~10% of maths word problems mention a
  person's gender at all. Unlabelled ≠ neutral.
- **Near-zero ethnicity coverage is itself the finding.** A bias the corpus
  never expresses cannot be mined, only synthesised.
- **Do not compare M:F ratios across languages.** French inflates the
  masculine for grammatical reasons. Compare directions.

---

## Adding a language

```bash
cp lang_template.py lang_de.py
# fill in the word lists
python biasscan.py --file corpus.csv --lang de --audit
```

Fill in, in this order — stop when it is good enough:

| field | effort | gives you |
|---|---|---|
| `pronouns` | ~5 words | §1, the biggest signal |
| `kinship` | ~25/gender | §1, §5 |
| `occupations` | ~100 words | §2, §3 — the headline result |
| `activities` | ~80 words | §4 |
| `dimensions` | ~70 words | §7 |
| `ambiguous` | ~20 words | the opt-in bucket |
| `money` | 1 regex | §6 — must match **your corpus's** currency |

`pronouns` + `kinship` alone already give a usable §1 and §5.

Settings to check, not translate:

| | |
|---|---|
| `segmentation` | `"char"` for Chinese/Japanese/Thai — scripts without spaces. `"word"` otherwise. |
| `auto_plural` | `True` for English-style `-s` only. Off for German, Russian, Arabic. |
| `window` | §3 proximity, in **characters**. ~2-char words → 10; ~6-char words → 40. |
| `decimal_comma` | `True` where `,` is the decimal separator. Wrong value is a **silent 100× error**: `12,50` becomes 1250. |
| dict keys | **Keep in English, unchanged** — that is what lets two languages be compared. |

### If your language marks gender on nouns

French, Spanish, German, Russian and others inflect the job title itself:
`infirmier`/`infirmière`.

That is an **opportunity**: `une infirmière` tells you the nurse is a woman
with no pronoun and no name. List both forms in `kinship` as well as
`occupations` and coverage rises a lot. `lang_fr.py` does this.

It is also a **trap**: if the job word supplies the gender *and* is the thing
being measured, §2 just rediscovers your grammar. biasscan masks the
occupation word before reading gender, which is why **§3 is the claim to
trust** and §2 is weaker in these languages.

Two details `lang_fr.py` handles, which yours probably needs too:

- **Epicene nouns** have one form for both genders (`le`/`la secrétaire`).
  They must **never** be gender signals — listing one marks every man in that
  job as a woman. French recovers the signal from the article instead, by
  listing `le secrétaire` / `la secrétaire` as terms. Elision
  (`l'architecte`) stays correctly unmarked.
- **Generic masculine plurals** (`les instituteurs` = a mixed group) are left
  **out** of `pronouns`, or every mixed group counts as male. French keeps
  `elles` and omits `ils`.

---

## Validation

Checked on a parallel corpus:
[CrowS-Pairs multilingual](https://huggingface.co/datasets/BigScienceBiasEval/crows_pairs_multilingual)
([Névéol et al. 2022](https://aclanthology.org/2022.acl-long.583/)) has 1,677
English and 1,677 French stereotype pairs, the French translated from the
English. Running each language pack on its own half tests whether two
independently written word lists agree.

| | en | fr |
|---|---|---|
| pronoun M:F | **1.81** | **1.82** |
| kinship M:F | 1.69 | 1.83 |
| docs with a man | 38.6% | 35.1% |
| ethnicity coverage | 23.4% | 21.4% |
| appearance coverage | 8.1% | 9.2% |
| rural coverage | 0.8% | 0.9% |

Also run on unrelated prose — 50,000 English IMDB reviews and 200,000 French
Allociné reviews. The result that holds everywhere:

**`nurse` is female-skewed in every corpus measured** — odds 0.57 (IMDB), 0.07
(Allociné, p=4e-43), and 0.38 for the equivalent care/service category in
Chinese maths problems. Three languages, three genres, one direction.
`agriculture` is male-skewed at 2.46 and 2.79 in the two review corpora.

---

## Building an eval set from this

`--out` writes `NAME_labelled.csv`: every document with its gender label, the
occupation words found, and the activities. Suitable for pairing.

If you go on to swap genders and test a model, **include a random-name
baseline** — also swap to a *different same-gender* name. Without it you are
measuring prompt sensitivity, not bias. In one reported comparison,
demographic and neutral perturbation gave flip rates of 14.9% vs 14.1%.
