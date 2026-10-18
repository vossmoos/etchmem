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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from app.anonymize import EXTRACT_ANON_BLOCK, RESOLVE_ANON_BLOCK
from app.config import settings

if TYPE_CHECKING:
    from app.ext import ExtRegistry


def model_settings() -> dict | None:
    """Per-request settings for every model call: a request timeout, so a hung call
    fails fast and is retried instead of waiting for the HTTP client's default."""
    timeout = settings.llm_timeout_seconds
    return {"timeout": float(timeout)} if timeout and timeout > 0 else None


# ── Stage 2: claim extraction ────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    entity_name: str = Field(..., description="Surface name of the subject, e.g. 'Acme Corp'.")
    entity_type: str = Field("company", description="company | person | product | project | other")
    property: str = Field(..., description="snake_case property within a domain, e.g. 'contract_status'.")
    value: str = Field(..., description="Short canonical value, e.g. 'signed', 'broken'. Prefer 1-3 words.")
    polarity: Literal["asserted", "negated"] = "asserted"
    event_time: str | None = Field(None, description="ISO-8601 time the fact became true, if stated.")
    detail: str | None = Field(
        None,
        description="Only for properties marked WITH DETAIL: the full wording of the fact "
                    "as the author stated it. Leave empty for every other property.")
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
  Exception: a property marked WITH DETAIL takes a 2-5 word LABEL as `value`
  and the author's full wording in `detail`. Never fill `detail` otherwise.
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
- If known labels are provided, reuse the exact label when a signal states the
  same fact again or retracts it ("we dropped X" negates the label X exactly).
- A signal may carry a line `[meaning in its conversation: ...]`: what the signal
  says once its references to earlier messages ("that", "yes", "no, new task")
  are spelled out. Treat it as the signal's content, and extract from it.
  Agreeing to work ("let us implement that") makes it decision_state "agreed",
  and every concrete thing it agrees should be done is a requirement of its own.
- A signal (or meaning) that only asks a question states nothing: no claims.
- A subject marked `[wrongly recorded]` had facts recorded on it that belong to
  another subject. Emit exactly those facts negated for it, with the same labels,
  and asserted for the right subject. Never state anything else about it: not its
  decision_state, not its work_state.
- A signal may end with a line `[subjects: ...]`: the subjects it was found to
  be about. Emit claims only about those subjects, using those exact names and
  types. `[subjects: none]` means the signal is about no subject: no claims.
- A signal may start with a header line `[sent <date> · source <name>]`. It is
  metadata, not content: never extract claims from it. Resolve relative dates in
  the text ("Friday", "yesterday", "in two weeks") against the sent date, and
  write every date value and `event_time` as an absolute ISO-8601 date.
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


@dataclass(frozen=True)
class SignalInput:
    """One signal as the extractor sees it: the text plus when and from where.

    `subjects` are the subjects the resolve step found this signal to be about
    ("name (type)"); None when no resolve step ran. An empty tuple means the
    signal is about no identifiable subject.
    """
    text: str
    sent_at: float = 0.0            # occurred_at, else ingest time (epoch seconds)
    source: str = ""
    subjects: tuple[str, ...] | None = None
    # What the signal means in its conversation, written by the resolve step when it
    # refers to earlier messages ("let us implement that"). Extraction reads it next
    # to the signal; the raw conversation itself never reaches extraction.
    meaning: str = ""


def meaning_footer(sig: SignalInput) -> str:
    return f"\n[meaning in its conversation: {sig.meaning}]" if sig.meaning else ""


def subjects_footer(sig: SignalInput) -> str:
    if sig.subjects is None:
        return ""
    listed = "; ".join(sig.subjects) if sig.subjects else "none"
    return f"\n[subjects: {listed}]"


@dataclass(frozen=True)
class SubjectCandidate:
    """An existing subject offered to the resolve step.

    role: "focus" (the conversation has been about it), "linked" (tied to a focus
    subject by a declared relation), or "similar" (found by embedding search).
    """
    name: str
    entity_type: str
    facts: tuple[str, ...] = ()
    role: str = "similar"


@dataclass(frozen=True)
class ConversationView:
    """Where a signal was said: what that conversation has been about so far
    (focus subjects, most recent first) and its last messages before this one."""
    focus: tuple[str, ...] = ()
    recent: tuple[str, ...] = ()


class ResolvedSubject(BaseModel):
    name: str = Field(..., description="The subject's name: a candidate's exact name, or the "
                                       "name the signal uses for a new subject.")
    entity_type: str = Field(..., description="One of the declared subject types.")
    existing: bool = Field(..., description="True when this is one of the candidates.")
    quote: str = Field(..., description="The words of the signal that refer to this subject.")
    reason: str = Field("", description="One short sentence: why this subject.")
    in_focus: bool = Field(False, description="True when this is a subject the "
                                              "conversation was already about.")
    wrongly_recorded: bool = Field(
        False, description="True when the signal corrects earlier facts that were "
                           "recorded on this subject but do not belong to it.")


class AssociatedSubject(BaseModel):
    """Related knowledge recalled while reading a signal: worth knowing, but not
    what the signal is about. Recorded; never a claim."""
    name: str
    entity_type: str
    reason: str = Field(..., description="Why it is related, in one short sentence.")


class SignalSubjects(BaseModel):
    signal_index: int
    subjects: list[ResolvedSubject] = Field(default_factory=list)
    associations: list[AssociatedSubject] = Field(default_factory=list)
    meaning: str = Field(
        "", description="Only when the signal refers to earlier messages: what it says "
                        "with those references spelled out, as one self-contained "
                        "statement in the author's terms. Empty when it stands on its own.")


@dataclass
class Resolution:
    """The resolve step's answer for one signal."""
    subjects: list[ResolvedSubject] = field(default_factory=list)
    associations: list[AssociatedSubject] = field(default_factory=list)
    meaning: str = ""


class BatchResolution(BaseModel):
    signals: list[SignalSubjects] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim_id: str = Field(..., description="The claim's id as given, e.g. 'c3'.")
    supported: bool
    same_as: str | None = Field(
        None, description="An existing label, copied exactly, when the claim states that "
                          "same fact (or retracts it). Otherwise null.")
    reason: str = ""


@dataclass(frozen=True)
class ClaimCheck:
    """The verify step's answer for one claim."""
    supported: bool
    same_as: str | None = None


class BatchVerification(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


_RESOLVE_SYSTEM_SUBJECTS = """\
You decide what each signal is ABOUT, before any facts are extracted from it.

For every '### Signal <i>' return the subjects it states facts about.

Read a signal in its conversation first. It may come with "Conversation focus"
(what the conversation it was said in has been about so far) and "Earlier in the
conversation" (its last messages, and the last ones around it, each with how long
before it was said). Unless the signal clearly turns to something else, it is about
the conversation's focus: "this ticket", "it", "the link" refer to what the
conversation is discussing. A reply like "No, ..." continues an earlier message;
judge from the content and the time gap whether that earlier message still frames
it. Mark subjects the conversation was already about in_focus=true.

Each signal also comes with candidates: existing subjects, labelled [focus],
[linked] (tied to a focus subject, e.g. the ticket that tracks it) or [similar].
They are options, not content.
- Choose a candidate only when the signal itself refers to it: by its name, its
  key, or a description a reader would match to that same item, in any language or
  wording — or by a reference ("this ticket", "it") while the conversation is
  about it.
- A signal that names an existing item is about it: "a plan for followup-3",
  "the link in followup-3" are about followup-3.
- Same area is not the same item. A signal that does NOT name an existing item
  and asks to add, build or change something that item does not already describe
  is about NEW work, even when the item covers the same component: name the new
  work by what it is (existing=false) and put the existing item in
  `associations`.
- Similarity alone is not a reference. A candidate the signal does not refer to
  is not a subject of the signal, however close it looks.
- Words like "the link", "this ticket", "it", "the plan" are references, not
  names. Resolve them to the conversation focus, or to the candidate that what the
  signal says points to (its facts: an idea about links, a ticket about the plan).
  Never make a reference word the name of a new subject; if nothing fits, give no
  subject.
- Your meaning and your subjects must agree: every piece of work the meaning
  talks about is among the subjects. Give no subjects only when the meaning names
  no identifiable work.
- A subject the signal names that is not among the candidates is new: name it the
  way the signal does and set existing=false.
- A signal that states facts about no identifiable subject gets no subjects.
- Recalling a related subject is not the same as the signal being about it. An
  earlier iteration of the same work, a ticket that already covers it, a dependency
  goes into `associations` with a short reason, never into subjects, unless the
  signal itself states something about it. At most 3 associations, and only ones
  that matter for what the signal says. A candidate that merely looks similar is
  neither a subject nor an association.
- A correction ("no, new task", "not this ticket", "that was about X, not Y") says
  the work is not the other subject. Mark that other subject wrongly_recorded=true
  ONLY when the earlier messages show facts recorded on it ("recorded: X · ...");
  otherwise it is just an association.
- `quote` is the signal's own words that refer to the subject.
- `meaning`: when the signal refers to earlier messages ("let us implement that",
  "yes, good idea, but ...", "no, new task"), write what it says with those
  references spelled out, as one self-contained statement, using the earlier
  messages' own terms. Say what is agreed, requested or corrected; add nothing the
  conversation does not say. A question stays a question ("Andrey asks whether
  ...?"). Leave it empty when the signal stands on its own.
- A header line `[sent ... · source ...]` is metadata, never a subject.
"""

_VERIFY_SYSTEM = """\
You check extracted claims against the signal they were extracted from.

For every claim answer `supported`: true only when the signal itself states this
fact about this subject. False when the subject is not the one the signal is
about, when the fact is not stated, or when it was inferred from something other
than the signal. Judge meaning, not wording; any language. A signal may come with
"Meaning in its conversation": what it says once its references to earlier
messages are spelled out. A claim supported by that meaning is supported. A
correction ("not this ticket, a new task") supports both the facts on the right
subject and the same facts negated on the subject it says they were wrongly
recorded on.

Some claims list the labels memory already holds for that subject and property.
When the claim states the same fact as one of them, or retracts it, set `same_as`
to that label, copied exactly. A different fact gets null.

Give one verdict for every claim id.
"""


def signal_header(sig: SignalInput) -> str:
    """`[sent 2026-10-16 (Fri) 14:05 UTC · source slack]` — or '' without a time."""
    if not sig.sent_at:
        return ""
    t = datetime.fromtimestamp(sig.sent_at, tz=timezone.utc)
    when = t.strftime("%Y-%m-%d (%a)") + ("" if (t.hour, t.minute) == (0, 0)
                                          else t.strftime(" %H:%M UTC"))
    return f"[sent {when}" + (f" · source {sig.source}" if sig.source else "") + "]\n"


def context_hint(known_entities: list[str] | None,
                 known_labels: list[str] | None) -> str:
    hint = ""
    if known_entities or known_labels:
        hint += ("\n\nREFERENCE ONLY — the lists below are spellings to reuse, "
                 "not content. Never create a claim about a subject because it "
                 "is listed here: a claim's subject must be named or clearly "
                 "referred to in the signal itself.")
    if known_entities:
        hint += ("\n\nKnown entities (reuse exact names if same): "
                 + ", ".join(known_entities[:50]))
    if known_labels:
        hint += ("\n\nLabels already in memory for these subjects. They are NOT facts of "
                 "this signal and say nothing about what it states. Use one only to spell "
                 "a label identically when the signal states that same fact again or "
                 "retracts it. Extract every fact the signal states, whether or not a "
                 "label exists for it:\n"
                 + "\n".join(f"- {line}" for line in known_labels))
    return hint


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, signal_text: str, known_entities: list[str] | None = None) -> ExtractionResult: ...

    def extract_batch(
        self, signal_texts: list[str], known_entities: list[str] | None = None,
    ) -> list[ExtractionResult]:
        """Extract claims for several signals. Default: sequential fallback."""
        return [self.extract(t, known_entities) for t in signal_texts]

    def extract_signals(
        self, signals: list[SignalInput], known_entities: list[str] | None = None,
        known_labels: list[str] | None = None,
    ) -> list[ExtractionResult]:
        """What the pipeline calls: signals with their send time and source.

        Default: plain texts through `extract_batch`, so stubs and simple
        backends need no changes. Model-backed extractors override this to show
        the header and the known labels to the model.
        """
        return self.extract_batch([s.text for s in signals], known_entities)

    def resolve_subjects(
        self, signals: list[SignalInput], candidates: list[list[SubjectCandidate]],
        conversations: list[ConversationView | None] | None = None,
    ) -> list["Resolution | list[ResolvedSubject]"] | None:
        """Which subjects each signal is about, read in its conversation, plus
        related subjects recalled on the way (associations). None: this backend has
        no resolve step, and the pipeline offers the candidates as hints instead.
        A plain list per signal is taken as subjects without associations."""
        return None

    def verify_claims(
        self, signals: list[SignalInput], results: list[ExtractionResult],
        labels: list[list[list[str]]] | None = None,
    ) -> list[list[ClaimCheck]] | None:
        """Per claim: does the signal support it, and is it an existing label's
        fact? `labels[i][j]` are the labels memory holds for claim j's subject and
        property. None: this backend does not verify; keep every claim as is."""
        return None

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
            model_settings=model_settings(),
        )
        self._batch_agent = Agent(
            model or settings.claim_model,
            output_type=BatchExtractionResult,
            system_prompt=_CLAIM_SYSTEM + self._registry.prompt_block()
                          + _CLAIM_BATCH_SUFFIX + anon,
            model_settings=model_settings(),
        )
        self._resolve_agent = Agent(
            settings.resolve_model or model or settings.claim_model,
            output_type=BatchResolution,
            system_prompt=_RESOLVE_SYSTEM_SUBJECTS + self._registry.prompt_block(),
            model_settings=model_settings(),
        )
        self._verify_agent = Agent(
            settings.verify_model or model or settings.claim_model,
            output_type=BatchVerification,
            system_prompt=_VERIFY_SYSTEM,
            model_settings=model_settings(),
        )

    def resolve_subjects(
        self, signals: list[SignalInput], candidates: list[list[SubjectCandidate]],
        conversations: list[ConversationView | None] | None = None,
    ) -> list[Resolution] | None:
        blocks = []
        for i, (sig, cands) in enumerate(zip(signals, candidates)):
            lines = [f"### Signal {i}", ""]
            conv = conversations[i] if conversations else None
            if conv is not None and (conv.focus or conv.recent):
                lines.append("Conversation focus: " + (", ".join(conv.focus) or "none yet"))
                if conv.recent:
                    lines.append("Earlier in the conversation:")
                    lines += [f"- {m}" for m in conv.recent]
                lines.append("")
            lines += ["Signal:", signal_header(sig) + sig.text, "", "Candidates:"]
            for c in cands:
                facts = "; ".join(c.facts)
                lines.append(f"- [{c.role}] {c.name} ({c.entity_type})" + (f": {facts}" if facts else ""))
            if not cands:
                lines.append("- none")
            blocks.append("\n".join(lines))
        result = self._resolve_agent.run_sync("\n\n".join(blocks)).output
        out = [Resolution() for _ in signals]
        for entry in result.signals:
            if 0 <= entry.signal_index < len(out):
                out[entry.signal_index].subjects.extend(entry.subjects)
                out[entry.signal_index].associations.extend(entry.associations)
                out[entry.signal_index].meaning = entry.meaning or out[entry.signal_index].meaning
        return out

    def verify_claims(
        self, signals: list[SignalInput], results: list[ExtractionResult],
        labels: list[list[list[str]]] | None = None,
    ) -> list[list[ClaimCheck]] | None:
        ids = {f"c{n}": (i, j) for n, (i, j) in enumerate(
            (i, j) for i, r in enumerate(results) for j in range(len(r.claims)))}
        checks: list[list[ClaimCheck | None]] = [[None] * len(r.claims) for r in results]

        def ask(wanted: dict[str, tuple[int, int]]) -> None:
            blocks = []
            for i, sig in enumerate(signals):
                mine = [(cid, j) for cid, (si, j) in wanted.items() if si == i]
                if not mine:
                    continue
                lines = ["### Signal", "", signal_header(sig) + sig.text]
                if sig.meaning:
                    lines += [f"Meaning in its conversation: {sig.meaning}"]
                lines += ["", "Claims:"]
                for cid, j in mine:
                    c = results[i].claims[j]
                    pol = " (NOT)" if c.polarity == "negated" else ""
                    lines.append(f"- {cid}: {c.entity_name} ({c.entity_type}) · {c.property}"
                                 f"{pol} = {c.value}" + (f" — {c.detail}" if c.detail else ""))
                    known = labels[i][j] if labels else []
                    if known:
                        lines.append("  existing labels: " + "; ".join(known))
                blocks.append("\n".join(lines))
            out = self._verify_agent.run_sync("\n\n".join(blocks)).output
            for v in out.verdicts:
                if v.claim_id in wanted:
                    i, j = wanted[v.claim_id]
                    checks[i][j] = ClaimCheck(v.supported, v.same_as or None)

        if ids:
            ask(ids)
            missing = {cid: ij for cid, ij in ids.items() if checks[ij[0]][ij[1]] is None}
            if missing:
                ask(missing)        # once more, only for the claims left without a verdict
        # Still unanswered after two asks: kept as extracted, not judged.
        return [[c if c is not None else ClaimCheck(True) for c in row] for row in checks]

    def extract(self, signal_text: str, known_entities: list[str] | None = None,
                known_labels: list[str] | None = None) -> ExtractionResult:
        hint = context_hint(known_entities, known_labels)
        result = self._agent.run_sync(f"## Signal\n\n{signal_text}{hint}").output
        # Drop claims that violate a declared extension's enum / entity_types.
        # Unknown (core, open-vocabulary) properties always pass.
        kept = [
            c for c in result.claims
            if self._registry.accept(property=c.property, value=c.value, entity_type=c.entity_type)
        ]
        return ExtractionResult(claims=kept)

    def extract_signals(
        self, signals: list[SignalInput], known_entities: list[str] | None = None,
        known_labels: list[str] | None = None,
    ) -> list[ExtractionResult]:
        texts = [signal_header(s) + s.text + meaning_footer(s) + subjects_footer(s)
                 for s in signals]
        return self.extract_batch(texts, known_entities, known_labels=known_labels)

    def extract_batch(
        self, signal_texts: list[str], known_entities: list[str] | None = None,
        known_labels: list[str] | None = None,
    ) -> list[ExtractionResult]:
        if len(signal_texts) == 1:
            return [self.extract(signal_texts[0], known_entities, known_labels)]
        hint = context_hint(known_entities, known_labels)
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
            model_settings=model_settings(),
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
