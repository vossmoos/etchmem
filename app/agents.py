"""
The LLM stages of the cascade, as Pydantic-AI agents.

Stage 1 (signal dedup) is embedding-based and lives in app/dedup.py — no LLM.

Stage 2  ClaimExtractor  (mini model)  : signal text → structured claims.
Stage 3  ConflictResolver (top model)  : competing claims → resolved belief.

Each stage is an ABC so tests can inject deterministic stubs with no network.
Model strings come from settings and are swapped with one env var each.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from app.anonymize import EXTRACT_ANON_BLOCK, RESOLVE_ANON_BLOCK
from app.config import settings

if TYPE_CHECKING:
    from app.ext import ExtRegistry


# ── Stage 2: claim extraction ────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    entity_name: str = Field(..., description="Surface name of the subject, e.g. 'Acme Corp'.")
    entity_type: str = Field("company", description="company | person | product | project | other")
    property: str = Field(..., description="snake_case property within a domain, e.g. 'contract_status'.")
    value: str = Field(..., description="Short canonical value, e.g. 'signed', 'broken'. Prefer 1-3 words.")
    polarity: Literal["asserted", "negated"] = "asserted"
    event_time: str | None = Field(None, description="ISO-8601 time the fact became true, if stated.")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="Extraction confidence.")


class ExtractionResult(BaseModel):
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="Zero or more claims. Empty = the signal carries no durable fact.",
    )


_CLAIM_SYSTEM = """\
You are etchmem's claim extractor. Convert a raw signal into structured claims.

A claim is one atomic fact: (entity, property, value). Rules:
- Identify the real-world SUBJECT entity and give it a stable surface name.
- Use a snake_case `property` describing WHICH attribute changed
  (e.g. contract_status, pricing_tier, decision_maker, relationship_health).
- Use a SHORT canonical `value` (1-3 words, lowercase where natural) so that
  equal facts get equal values (e.g. always "signed", not "signed the deal").
- One signal may yield several claims, or zero. If the signal states no durable
  fact (chatter, questions, greetings), return an empty list.
- Set polarity = "negated" for explicit negations ("did NOT sign").
- ONE SUBJECT PER CLAIM. If a sentence states something about several
  subjects, emit one claim per subject, each named on its own. Never put two
  subjects in one `entity_name`: "A and B both fail" is two claims. When a
  sentence names one subject and mentions another as context ("part X for
  machine Y"), decide which one the fact is about and name only that one.
- Do NOT invent facts. Do NOT merge different entities.
- If known entities are provided, reuse the exact name when it's the same one.
"""


_SUBJECT_CORRECTION = """

CORRECTION. Your previous answer named SEVERAL subjects in one claim's
`entity_name`: {names}. That is not answerable as one claim — a fact belongs to
one subject.

Re-read the signal and decide, from the sentence itself:
- If it states the fact about EACH of them ("A and B both leak"), emit one
  claim per subject, each `entity_name` carrying exactly one identifier.
- If it names one subject and mentions the other as context ("part A for
  machine B", "A replaces B"), emit one claim for the subject the fact is
  actually about, and drop the other.
- If the sentence does not make that clear, emit no claim for it.
{ceiling}
Never repeat a compound `entity_name`.
"""


class SignalClaims(BaseModel):
    """Claims extracted from ONE signal of a batch, keyed by its index."""
    signal_index: int = Field(..., description="0-based index of the '### Signal <i>' block these claims came from.")
    claims: list[ExtractedClaim] = Field(default_factory=list)


class BatchExtractionResult(BaseModel):
    signals: list[SignalClaims] = Field(
        default_factory=list,
        description="One entry per input signal (entries with zero claims may be omitted).",
    )


_CLAIM_BATCH_SUFFIX = """

BATCH MODE: you will receive SEVERAL independent signals, each under a
'### Signal <i>' heading. Apply all rules to EVERY signal separately and
return one entry per signal carrying its `signal_index`. Never mix facts
from different signals into one claim. A signal with no durable fact gets
an empty claims list (or may be omitted).
"""


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, signal_text: str, known_entities: list[str] | None = None) -> ExtractionResult: ...

    def extract_batch(
        self, signal_texts: list[str], known_entities: list[str] | None = None,
    ) -> list[ExtractionResult]:
        """Extract claims for several signals. Default: sequential fallback."""
        return [self.extract(t, known_entities) for t in signal_texts]

    def re_extract_subjects(
        self, signal_text: str, compound_names: list[str],
        multi_subject_properties: list[str] | None = None,
    ) -> ExtractionResult:
        """Re-read ONE signal that produced a claim naming several subjects.

        The model gets the original sentence back, plus what it got wrong. That
        is the only place the question can be answered: whether "A und B" is a
        list and "A für B" is a relation is a reading of the sentence, and by
        the time a claim exists the sentence is gone. A word list inspecting the
        mangled name would be guessing from strictly less information than the
        model already had.

        `multi_subject_properties` are the properties declared as legitimately
        applying to several subjects at once. They are passed to the model as
        guidance, never applied afterwards as a rule: a declaration cannot know
        whether this particular sentence is a list or a relation.

        Default implementation re-runs plain extraction, so stubs and simple
        backends need no changes.
        """
        return self.extract(signal_text)


class PydanticAIClaimExtractor(ClaimExtractor):
    def __init__(self, model: str | None = None, registry: "ExtRegistry | None" = None) -> None:
        from pydantic_ai import Agent

        from app.ext import load_extensions

        self._registry = registry if registry is not None else load_extensions()
        anon = EXTRACT_ANON_BLOCK if settings.claims_anonymization else ""
        self._agent = Agent(
            model or settings.claim_model,
            output_type=ExtractionResult,
            system_prompt=_CLAIM_SYSTEM + self._registry.prompt_block() + anon,
        )
        self._batch_agent = Agent(
            model or settings.claim_model,
            output_type=BatchExtractionResult,
            system_prompt=_CLAIM_SYSTEM + self._registry.prompt_block()
                          + _CLAIM_BATCH_SUFFIX + anon,
        )

    def extract(self, signal_text: str, known_entities: list[str] | None = None) -> ExtractionResult:
        hint = ""
        if known_entities:
            hint = "\n\nKnown entities (reuse exact names if same): " + ", ".join(known_entities[:50])
        result = self._agent.run_sync(f"## Signal\n\n{signal_text}{hint}").output
        # Drop claims that violate a declared extension's enum / entity_types.
        # Unknown (core, open-vocabulary) properties always pass.
        kept = [
            c for c in result.claims
            if self._registry.accept(property=c.property, value=c.value, entity_type=c.entity_type)
        ]
        return ExtractionResult(claims=kept)

    def extract_batch(
        self, signal_texts: list[str], known_entities: list[str] | None = None,
    ) -> list[ExtractionResult]:
        if len(signal_texts) == 1:
            return [self.extract(signal_texts[0], known_entities)]
        hint = ""
        if known_entities:
            hint = "\n\nKnown entities (reuse exact names if same): " + ", ".join(known_entities[:50])
        prompt = "\n\n".join(
            f"### Signal {i}\n\n{text}" for i, text in enumerate(signal_texts)
        ) + hint
        result = self._batch_agent.run_sync(prompt).output

        out = [ExtractionResult() for _ in signal_texts]
        for sc in result.signals:
            if not 0 <= sc.signal_index < len(out):
                continue  # hallucinated index — drop rather than mis-attribute
            kept = [
                c for c in sc.claims
                if self._registry.accept(property=c.property, value=c.value, entity_type=c.entity_type)
            ]
            out[sc.signal_index].claims.extend(kept)
        return out

    def re_extract_subjects(
        self, signal_text: str, compound_names: list[str],
        multi_subject_properties: list[str] | None = None,
    ) -> ExtractionResult:
        """Ask the model again, telling it exactly what it got wrong."""
        ceiling = ""
        if multi_subject_properties:
            ceiling = ("\nThese properties may legitimately hold for several "
                       "subjects at once: " + ", ".join(multi_subject_properties)
                       + ". Any other property belongs to ONE subject.\n")
        prompt = (f"## Signal\n\n{signal_text}"
                  + _SUBJECT_CORRECTION.format(names="; ".join(compound_names),
                                               ceiling=ceiling))
        result = self._agent.run_sync(prompt).output
        kept = [
            c for c in result.claims
            if self._registry.accept(property=c.property, value=c.value,
                                     entity_type=c.entity_type)
        ]
        return ExtractionResult(claims=kept)


# ── Stage 3: conflict resolution + narrative ─────────────────────────────────

class CompetingClaim(BaseModel):
    value: str
    polarity: str
    sources: list[str]
    corroboration_count: int
    event_time: float | None = None


class ConflictResolution(BaseModel):
    current_value: str = Field(..., description="The value that best represents current truth.")
    status: Literal["settled", "contested"] = Field(
        ..., description="'settled' if confidently resolved; 'contested' if genuine disagreement remains.")
    narrative: str = Field(..., description="One clear sentence stating the current belief, noting conflict if any.")
    confidence: float = Field(..., ge=0.0, le=1.0)


_RESOLVE_SYSTEM = """\
You are etchmem's conflict resolution engine. Several sources disagree about one
property of one entity. Decide the current belief.

Rules:
- Prefer the most recent claim when the property is a state that changes over
  time (e.g. contract_status: a later "broken" supersedes an earlier "signed").
- Weigh corroboration (more independent sources = stronger) and source trust.
- If genuine, unresolved disagreement remains, set status = "contested" and say
  so plainly in the narrative; otherwise "settled".
- The narrative is ONE sentence. Do not invent facts beyond the claims.
- Attribute opinions; never state them as fact.
"""


class ConflictResolver(ABC):
    @abstractmethod
    def resolve(self, entity_name: str, prop: str, claims: list[CompetingClaim]) -> ConflictResolution: ...


class PydanticAIConflictResolver(ConflictResolver):
    def __init__(self, model: str | None = None) -> None:
        from pydantic_ai import Agent

        anon = RESOLVE_ANON_BLOCK if settings.claims_anonymization else ""
        self._agent = Agent(
            model or settings.etch_model,
            output_type=ConflictResolution,
            system_prompt=_RESOLVE_SYSTEM + anon,
        )

    def resolve(self, entity_name: str, prop: str, claims: list[CompetingClaim]) -> ConflictResolution:
        lines = [
            f"- value={c.value!r} polarity={c.polarity} sources={c.sources} "
            f"corroboration={c.corroboration_count} event_time={c.event_time}"
            for c in claims
        ]
        prompt = (
            f"## Entity\n{entity_name}\n\n## Property\n{prop}\n\n"
            f"## Competing claims\n" + "\n".join(lines) + "\n\nResolve."
        )
        return self._agent.run_sync(prompt).output


# ── Factories ─────────────────────────────────────────────────────────────────

def build_claim_extractor() -> ClaimExtractor:
    return PydanticAIClaimExtractor()


def build_conflict_resolver() -> ConflictResolver:
    return PydanticAIConflictResolver()
