"""Shared list of bias types, and mapping of free-text group names to it.

Every gold label and every method output is mapped to these types, so that the
type predictions can be compared across datasets and methods.
"""
import re

TYPES = [
    "gender",
    "sexual_orientation",
    "disability",
    "age",
    "race_ethnicity",
    "nationality",
    "religion",
    "socioeconomic",
    "appearance",
    "profession",
    "other",
]

# Keyword patterns, matched on lowercased text with "_" and "-" replaced by spaces.
_KEYWORDS = {
    "gender": r"gender\w*|sexis\w*|wom[ae]n|m[ae]n|females?|males?|girls?|boys?|femini\w*|misogyn\w*"
    r"|mothers?|fathers?|wi(?:fe|ves)|husbands?|lad(?:y|ies)|trans|transgender\w*|non binary|nonbinary"
    r"|sisters?|brothers?|daughters?|sons?",
    "sexual_orientation": r"gays?|lesbians?|lgbt\w*|homosexual\w*|bisexual\w*|queer|orientation|homophob\w*",
    "disability": r"disab\w*|disorders?|mental\w*|illness\w*|autis\w*|blind|deaf|wheelchair\w*|handicap\w*"
    r"|retard\w*|down syndrome|ableis\w*",
    "age": r"age|ageis\w*|old|older|elderly|young|youth|child|children|kids?|teen\w*|boomers?|millennials?|seniors?",
    "race_ethnicity": r"races?|racial\w*|racis\w*|black\w*|white\w*|asians?|latin[oax]\w*|hispanic\w*"
    r"|native americans?|indigenous|first nations?|african\w*|arabs?|arabic|ethnic\w*|mexicans?|chinese"
    r"|middle east\w*|colou?r",
    "nationality": r"nationalit\w*|national|immigra\w*|foreign\w*|countr\w*|xenophob\w*|mexicans?|chinese"
    r"|americans?|germans?|french|british|english|russians?|italians?|japanese|refugees?|migrants?",
    "religion": r"relig\w*|jews?|jewish|judai\w*|muslims?|islam\w*|christian\w*|catholic\w*|hindu\w*"
    r"|buddhis\w*|atheis\w*|sikh\w*|mormon\w*|antisemit\w*",
    "socioeconomic": r"socio\w*|poor|poverty|rich|wealth\w*|homeless\w*|class|classes|classis\w*|income"
    r"|welfare",
    "appearance": r"fat|obes\w*|overweight|ugly|body|bodies|appearance|beauty|beautiful|attractive\w*"
    r"|skinny|thin|short|tall|weight",
    "profession": r"profession\w*|occupation\w*|jobs?|careers?",
}
_PATTERNS = {t: re.compile(rf"\b(?:{p})\b") for t, p in _KEYWORDS.items()}


def types_from_text(text):
    """Map a free-text group or bias name (e.g. "racial bias", "folks with physical disabilities")
    to the shared types. Returns ["other"] when nothing matches."""
    s = re.sub(r"[_\-]", " ", (text or "").lower())
    found = [t for t, p in _PATTERNS.items() if p.search(s)]
    return found or ["other"]


def types_from_list(text):
    """Pick the types named in a model answer such as "gender, race_ethnicity"."""
    s = (text or "").lower()
    return [t for t in TYPES if re.search(rf"\b{t}\b", s)]
