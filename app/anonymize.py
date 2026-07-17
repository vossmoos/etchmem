"""
Claims anonymization (enabled via ETCHMEM_CLAIMS_ANONYMIZATION=true).

Personal data is removed as signals are folded into claims/etches — the
retrieval surface. Raw signals keep the original text for provenance but are
excluded from recall while anonymization is on.

Three cooperating layers:

  1. Entity pseudonyms (deterministic, consistent). After entity resolution,
     person/company subject names are replaced by numbered tokens assigned
     once per canonical entity ([PERSON_1], [COMPANY_2]) and persisted in the
     right DB. Consistency preserves corroboration and conflict detection —
     the same person always folds onto the same etch.

  2. LLM instruction blocks appended to the Stage-2 extractor and Stage-3
     resolver prompts, so free-form personal data (addresses in particular)
     never enters claim values or narratives.

  3. `scrub()` — a deterministic regex safety net for machine-recognizable
     personal data (bank cards, IBANs, emails, phone numbers) applied to
     claim values and narratives regardless of what the LLM returned.
"""
from __future__ import annotations

import re

# Entity types that receive numbered pseudonym tokens. Other types (product,
# project, ...) are not personal data and keep their names.
ANON_ENTITY_LABELS: dict[str, str] = {
    "person": "PERSON",
    "company": "COMPANY",
}

# ── Regex safety net ─────────────────────────────────────────────────────────
# Order matters: IBAN before card (an IBAN body would otherwise match the
# card pattern), email before phone (digits in domains stay untouched).

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){10,30}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PHONE = re.compile(r"(?<![\w.])\+\d[\d\s().-]{6,}\d\b")

_RULES = (
    (_EMAIL, "[EMAIL]"),
    (_IBAN, "[IBAN]"),
    (_CARD, "[BANK_CARD]"),
    (_PHONE, "[PHONE]"),
)


def scrub(text: str) -> str:
    """Replace machine-recognizable personal data with tokens."""
    for pattern, token in _RULES:
        text = pattern.sub(token, text)
    return text


# ── LLM prompt blocks ────────────────────────────────────────────────────────

EXTRACT_ANON_BLOCK = """

ANONYMIZATION MODE is ON (privacy):
- `entity_name` must stay the REAL surface name — the server replaces it with
  a consistent pseudonym after entity resolution.
- `value` fields must NEVER contain personal data. Replace:
  street/postal addresses -> [ADDRESS]; bank card / account numbers ->
  [BANK_CARD]; IBANs -> [IBAN]; emails -> [EMAIL]; phone numbers -> [PHONE].
  A person's name used as a value -> [PERSON]; a company name used as a
  value -> [COMPANY].
- Prefer dropping a claim over leaking personal data through its value.
"""

RESOLVE_ANON_BLOCK = """
- ANONYMIZATION MODE is ON: the entity is identified by a pseudonym token
  (e.g. [PERSON_2]). Use that token in the narrative and never introduce real
  names, addresses, card numbers, emails or phone numbers — use [ADDRESS],
  [BANK_CARD], [IBAN], [EMAIL], [PHONE] instead.
"""
