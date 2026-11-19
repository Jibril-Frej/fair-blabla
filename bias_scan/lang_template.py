# -*- coding: utf-8 -*-
"""Template. Copy to lang_<code>.py and fill in the word lists.

    cp lang_template.py lang_de.py
    python biasscan.py --file corpus.csv --lang de --audit    # always audit first
    python biasscan.py --file corpus.csv --lang de

Fill these in, in this order. Stop whenever it is good enough:

    pronouns      ~5 words     -> section 1. Do this first, biggest signal.
    kinship       ~25/gender   -> sections 1 and 5.
    occupations   ~100 words   -> sections 2 and 3, the headline result.
    activities    ~80 words    -> section 4. housework vs repair is the
                                  sharpest contrast in both shipped packs.
    dimensions    ~70 words    -> section 7.
    ambiguous     ~20 words    -> the opt-in bucket, see below.
    money         1 regex      -> section 6. Match YOUR corpus's currency.

Leave `names` alone -- the user supplies names with --names.
Keep the dict KEYS in English and unchanged, so two languages can be compared.
An empty list is fine; that category is simply skipped.

Three things that are easy to get wrong:

* Everyday words double as measurements. "height" is geometry in a maths
  problem, body size in a novel. Put such words in `ambiguous`, not
  `dimensions`, and they count only with --include-ambiguous. Prefer multiword
  social phrasings ("white people", not "white") in `dimensions`.

* If your language inflects job titles by gender (Romance, Slavic, Semitic,
  German), list both forms in `kinship` as well as `occupations` -- that buys
  real coverage. See lang_fr.py. Watch out for epicene nouns, which have one
  form for both genders; those must NOT be gender signals.

* If the masculine plural is also the generic for mixed groups, leave it OUT
  of `pronouns`, or every mixed group counts as male.

Then run --audit and read the example contexts. That is the only way to know
your lists are measuring what you think.
"""
from biasscan import Lang

LANG = Lang(
    name="CHANGE ME",

    # "word" wraps terms in \b...\b. Use "char" for Chinese, Japanese, Thai --
    # any script written without spaces between words.
    segmentation="word",
    # derive "teachers"/"policemen" from singulars. Right for English only;
    # for German, Russian, Arabic leave it off and list the forms you want.
    auto_plural=False,
    # section 3's proximity window, in CHARACTERS not words.
    # ~2-char words (Chinese) -> 10.  ~6-char words (European) -> 40.
    window=40,
    # True where "," is the decimal separator and "." or space groups
    # thousands: French, German, Spanish, Italian, Russian. Wrong value is a
    # silent 100x error -- "12,50" would be read as 1250.
    decimal_comma=False,

    # Subject, object and possessive forms. EXCLUDE plurals that cover mixed
    # groups ("they", French "ils") -- they blur the signal.
    pronouns={"M": [], "F": []},

    # Kinship terms plus any noun that marks gender by itself: parent,
    # grandparent, sibling, child, aunt/uncle, spouse, "boy"/"girl",
    # "man"/"woman", honorifics (Mr/Mrs), and inflected job titles if your
    # language has them.
    kinship={"M": [], "F": []},

    occupations={
        "teacher": [],
        "doctor": [],
        "nurse": [],
        "agriculture": [],
        "manual_worker": [],
        "driver": [],
        "retail_service": [],      # shop, food, cleaning, childcare
        "business_owner": [],
        "professional": [],        # engineer, scientist, lawyer, accountant
        "public_military": [],
        "arts_sports": [],
        "clerical": [],
        # In school corpora this lands at M:F ~ 1 because of class-headcount
        # sentences ("30 boys and 28 girls"). That is a genre artifact, not
        # balance. Do not report it as a finding.
        "student": [],
    },

    activities={
        "housework": [],
        "repair_build": [],
        "shopping": [],
        "sports": [],
        "study": [],
        "labor_production": [],
    },

    dimensions={
        # Expect near-zero in technical corpora. That absence IS the finding:
        # a bias the corpus never expresses cannot be mined, only synthesised.
        "ethnicity": [],
        "rural": [],
        "urban": [],
        "socioeconomic": [],
        "appearance": [],
    },

    # Opt-in with --include-ambiguous. Single common words that double as
    # units, colours, sizes or adverbs go here, not in `dimensions`.
    # Real examples caught by --audit:
    #   "height" = geometry (86% of appearance hits in maths problems)
    #   "white"  = marble colour there, real ethnicity in news
    #   "pretty" = "pretty hard", an adverb (22% of appearance hits in reviews)
    #   "poor"   = "poor editing", a quality judgement
    ambiguous={
        "ethnicity": [],
        "appearance": [],
        "socioeconomic": [],
    },

    # One or more capture groups yielding a number. Set to None to skip
    # section 6. Mind the decimal separator:
    #   en: r"(?:\$\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*dollars?)"
    #   fr: r"([\d  .]+(?:,\d+)?)\s*(?:€|euros?)"
    money=None,
    money_unit="",
)
