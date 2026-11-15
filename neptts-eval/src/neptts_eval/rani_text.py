"""Rani text normalization, identical to the official NepTel 0.2 scorer.

Ported from github.com/Ampixa/rani-asr@583dbef2 eval/score_v02.py (MIT); only the
resource paths changed. Both reference and hypothesis pass through `normalize`:
digits become spoken Nepali words (vendored `rani_numbers`), English in either
script maps to one Devanagari form (`rani_latin_map.json`), punctuation is removed.

Known upstream quirk, kept on purpose so scores match Rani's published numbers:
the number speller writes short-i forms (२९ -> उनन्तिस) while Rani's training
normalizer writes long-i forms (उनन्तीस). A prompt containing digits therefore
costs every system the same constant penalty; rankings are unaffected.
"""

import html
import json
import re
import unicodedata
from pathlib import Path

from . import rani_numbers

MAP_PATH = Path(__file__).with_name("rani_latin_map.json")
_MAP = json.loads(MAP_PATH.read_text(encoding="utf-8"))
MAPPING, MAX_WORDS = _MAP["mapping"], _MAP["max_words"]

LATIN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*(?:[ \t]+[A-Za-z]+(?:['-][A-Za-z]+)*)*")
AT = re.compile(r"\s*@\s*")
DOT = re.compile(r"(?<=[A-Za-zऀ-ॣ॰-ॿ])\.(?=[A-Za-zऀ-ॣ॰-ॿ])")
EVENT = re.compile(r"[\[(][^\])]*[\])]")
PUNCT = re.compile(r"[।॥,.?!:;\"“”‘’…–—_/\\|*~^`<>{}=+#%&@$]")
LAT_WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


def _replace_run(match):
    words, out, i = match.group().split(), [], 0
    while i < len(words):
        for length in range(min(MAX_WORDS, len(words) - i), 0, -1):
            key = " ".join(words[i:i + length]).lower()
            if key in MAPPING:
                out.append(MAPPING[key])
                i += length
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _clean(text):
    s = (unicodedata.normalize("NFC", html.unescape(text))
         .replace("‌", "").replace("‍", "").replace(">>", " "))
    s = PUNCT.sub(" ", EVENT.sub(" ", s))
    out = []
    for tok in s.split():
        tok = tok.strip("-'")
        if not tok:
            continue
        if LAT_WORD.match(tok):
            letters = tok.replace("-", "").replace("'", "")
            tok = tok if (len(letters) >= 2 and letters.isupper()) else tok.lower()
        out.append(tok)
    return " ".join(out)


def normalize(text: str) -> str:
    text = DOT.sub(" dot ", AT.sub(" at the rate ", text))
    if re.search("[0-9०-९]", text):
        text = rani_numbers.normalize(text, english=False)
    return _clean(LATIN.sub(_replace_run, text))
