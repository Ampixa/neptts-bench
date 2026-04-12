"""Nepali text normalization for CER/WER computation."""

import re
import unicodedata


def normalize_nepali_text(text: str) -> str:
    """Normalize Nepali text for comparison.

    1. Unicode NFC normalization
    2. Strip punctuation (Devanagari + Latin)
    3. Normalize Devanagari numerals to Arabic
    4. Lowercase Latin characters
    5. Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥,;:!?\.\-\"'\(\)\[\]{}<>…–—/\\@#\$%\^&\*\+\=\|~`]", "", text)

    dev_digits = "०१२३४५६७८९"
    for i, d in enumerate(dev_digits):
        text = text.replace(d, str(i))

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
