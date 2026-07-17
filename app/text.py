"""Small deterministic text helpers: slugs, entity-name and value normalization."""
from __future__ import annotations

import re

_COMPANY_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited", "llc",
    "gmbh", "co", "company", "plc", "sa", "ag", "bv", "oy", "ab",
}

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_entity_name(name: str, entity_type: str | None = None) -> str:
    """Lowercase, strip punctuation, collapse whitespace; drop company suffixes."""
    s = _PUNCT.sub(" ", name.lower())
    s = _WS.sub(" ", s).strip()
    if entity_type in (None, "company", "organization", "org"):
        tokens = [t for t in s.split() if t not in _COMPANY_SUFFIXES]
        if tokens:
            s = " ".join(tokens)
    return s


def slugify(text: str) -> str:
    s = _PUNCT.sub(" ", text.lower())
    s = _WS.sub("_", s).strip("_")
    return s or "entity"


def normalize_value(value: str) -> str:
    """Normalize a property value for equality comparison in the gate."""
    s = _PUNCT.sub(" ", value.lower())
    return _WS.sub(" ", s).strip()
