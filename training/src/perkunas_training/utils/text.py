from __future__ import annotations

import html
import re
import unicodedata


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE = re.compile(r"[ \t\r\f\v]+")
MANY_NEWLINES = re.compile(r"\n{4,}")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str, *, normalize_unicode: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\ufeff", "")
    text = html.unescape(text)
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub("", text)
    lines = [WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = MANY_NEWLINES.sub("\n\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def looks_low_value(text: str) -> bool:
    if not text:
        return True
    printable = sum(ch.isprintable() or ch == "\n" for ch in text)
    if printable / max(1, len(text)) < 0.98:
        return True
    unique = len(set(text))
    if len(text) > 80 and unique <= 4:
        return True
    alpha = sum(ch.isalpha() for ch in text)
    if len(text) > 200 and alpha / max(1, len(text)) < 0.05:
        return True
    return False
