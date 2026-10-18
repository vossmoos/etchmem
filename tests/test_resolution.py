"""
Subject resolution and claim verification in the pipeline, with stub models.
No network, no LLM. Run:  pytest -q   (from etchmem-server/)

The resolve step decides what a signal is about before claims are extracted; the
verify step drops claims the signal does not state. Both are model judgements;
these tests pin only how the pipeline wires them: what each step is given, what
is kept, and what is recorded.
"""
from __future__ import annotations

import textwrap

import pytest

from app import config as _cfg
from app.agents import (AssociatedSubject, ClaimCheck, ClaimExtractor, ConflictResolution,
                        ConflictResolver, ExtractedClaim, ExtractionResult, Resolution,
                        ResolvedSubject)
from app.embeddings import FakeEmbedding
from app.service import MemoryService

S = _cfg.settings


class Res(ConflictResolver):
    def resolve(self, e, p, c):
        return ConflictResolution(current_value=c[0].value, status="settled",
                                  narrative=f"{e} {p} {c[0].value}", confidence=0.9)


class Scripted(ClaimExtractor):
    """Signal text 'entity|property|value'. Resolution and verdicts are scripted
    per text; every call records what it was given."""

    def __init__(self):
        self.subjects: dict[str, list[ResolvedSubject]] = {}
        self.unsupported: set[tuple[str, str]] = set()      # (text, entity)
        self.same_as: dict[tuple[str, str], str] = {}       # (text, value) -> label
        self.associations: dict[str, list[AssociatedSubject]] = {}
        self.meanings: dict[str, str] = {}
        self.resolve_calls, self.extract_calls, self.verify_calls = [], [], []

    def extract(self, text, known_entities=None):
        raise AssertionError("pipeline must call extract_signals")

    def resolve_subjects(self, signals, candidates, conversations=None):
        self.resolve_calls.append((signals, candidates, conversations))
        return [Resolution(self.subjects.get(s.text, []), self.associations.get(s.text, []),
                           self.meanings.get(s.text, "")) for s in signals]

    def extract_signals(self, signals, known_entities=None, known_labels=None):
        self.extract_calls.append((signals, known_entities, known_labels))
        out = []
        for s in signals:
            claims = []
            for part in s.text.split(";"):
                ent, prop, val = part.split("|")
                claims.append(ExtractedClaim(entity_name=ent, entity_type="feature",
                                             property=prop, value=val))
            out.append(ExtractionResult(claims=claims))
        return out

    def verify_claims(self, signals, results, labels=None):
        self.verify_calls.append((signals, results, labels))
        return [[ClaimCheck((s.text, c.entity_name) not in self.unsupported,
                            self.same_as.get((s.text, c.value)))
                 for c in r.claims] for s, r in zip(signals, results)]


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "t.yaml").write_text(textwrap.dedent("""
        domain: t
        entities:
          - type: feature
          - type: ticket
        properties:
          - name: tracked_in
            relation: ticket
          - name: work_state
            values: [todo, done]
          - name: idea
            detail: true
          - name: decision_state
            values: [idea, proposed]
    """), encoding="utf-8")
    for name, value in {"data_dir": str(tmp_path / "data"), "ext_dir": str(ext),
                        "multi_value_properties": "idea", "extract_min_batch": 1,
                        "subject_candidates": 8, "extract_known_labels": 10,
                        "verify_claims": True, "claims_anonymization": False,
                        "context_window_seconds": 7200.0}.items():
        monkeypatch.setattr(S, name, value)
    ext_ = Scripted()
    service = MemoryService(embedder=FakeEmbedding(), extractor=ext_, resolver=Res())
    service.ext = ext_
    return service


def subject(name, existing):
    return ResolvedSubject(name=name, entity_type="feature", existing=existing,
                           quote=name, reason="named")


def test_extraction_sees_only_the_resolved_subjects_verify_sees_labels(svc):
    first = "followup-3|idea|links in emails"
    svc.ext.subjects[first] = [subject("followup-3", False)]
    svc.remember(first, "slack", "t")
    svc.sleep()
    signals, names, labels = svc.ext.extract_calls[0]
    assert signals[0].subjects == ("followup-3 (feature)",)
    assert names is None and labels is None              # nothing known yet

    second = "followup-3|idea|radar registration links"
    svc.ext.subjects[second] = [subject("followup-3", True)]
    svc.remember(second, "slack", "t")
    svc.sleep()
    signals, cands, _ = svc.ext.resolve_calls[1]
    assert [c.name for c in cands[0]] == ["followup-3"]   # offered as a candidate
    assert any("idea: links in emails" in f for f in cands[0][0].facts)
    signals, names, labels = svc.ext.extract_calls[1]
    assert names is None and labels is None              # message and subjects only
    _, _, claim_labels = svc.ext.verify_calls[1]
    assert claim_labels == [[["links in emails"]]]       # labels go to verify


def test_verify_reuses_an_existing_label_for_the_same_fact(svc):
    first = "followup-3|idea|links in emails"
    svc.ext.subjects[first] = [subject("followup-3", False)]
    svc.remember(first, "slack", "t")
    svc.sleep()
    again = "followup-3|idea|email links"
    svc.ext.subjects[again] = [subject("followup-3", True)]
    svc.ext.same_as[(again, "email links")] = "links in emails"
    svc.remember(again, "slack", "t")
    svc.sleep()
    idea = next(e for e in svc.know("followup-3").etches if e.property == "idea")
    assert [v.value for v in idea.values] == ["links in emails"]   # one value, corroborated

    invented = "followup-3|idea|parquet export"
    svc.ext.subjects[invented] = [subject("followup-3", True)]
    svc.ext.same_as[(invented, "parquet export")] = "no such label"
    svc.remember(invented, "slack", "t")
    svc.sleep()
    idea = next(e for e in svc.know("followup-3").etches if e.property == "idea")
    assert {v.value for v in idea.values} == {"links in emails", "parquet export"}


def test_a_signal_about_nothing_gets_an_empty_subject_list(svc):
    text = "anything|idea|x"
    svc.ext.subjects[text] = []
    svc.remember(text, "slack", "t")
    svc.sleep()
    signals, _, _ = svc.ext.extract_calls[0]
    assert signals[0].subjects == ()


def test_unsupported_claims_are_dropped_and_counted(svc):
    text = "followup-3|idea|links;etchmem development|decision_state|idea"
    svc.ext.subjects[text] = [subject("followup-3", False)]
    svc.ext.unsupported.add((text, "etchmem development"))
    sig_id, _, _ = svc.remember(text, "slack", "t")
    summary = svc.sleep()
    assert summary["claims_unsupported"] == 1
    assert svc.know("etchmem development") is None or not svc.know("etchmem development").etches
    record = svc.signal(sig_id)
    assert record.status == "extracted"
    assert [(c.entity_name, c.property) for c in record.claims] == [("followup-3", "idea")]
    assert record.metadata["subjects"][0]["name"] == "followup-3"


def test_verification_can_be_switched_off(svc, monkeypatch):
    monkeypatch.setattr(S, "verify_claims", False)
    text = "followup-3|idea|links"
    svc.ext.subjects[text] = [subject("followup-3", False)]
    svc.remember(text, "slack", "t")
    svc.sleep()
    assert svc.ext.verify_calls == []


def test_signal_record_before_extraction_and_unknown_id(svc):
    sig_id, _, _ = svc.remember("followup-3|idea|links", "slack", "t",
                                metadata={"channel": "C1", "ts": "1.0"})
    record = svc.signal(sig_id)
    assert record.status == "new" and record.claims == []
    assert record.metadata == {"channel": "C1", "ts": "1.0"}
    assert svc.signal("nope") is None


# ── conversations ──────────────────────────────────────────────────────────

T = 1_800_000_000.0


def said(svc, text, at, thread, channel="slack:C1"):
    sig_id, stored, _ = svc.remember(text, "slack", "t", occurred_at=at,
                                     context=f"{channel}:{thread}", context_parent=channel)
    assert stored
    return sig_id


def test_a_reply_is_read_in_its_thread_with_the_ticket_its_focus_is_tracked_in(svc):
    # memory already knows: followup-3 is tracked in LIFT-38
    link = "followup-3|tracked_in|LIFT-38"
    svc.ext.subjects[link] = [subject("followup-3", False)]
    svc.remember(link, "mossy", "t")
    svc.sleep()

    first = "followup-3|requirement|landing link"
    svc.ext.subjects[first] = [subject("followup-3", True)]
    said(svc, first, T, "100.0")
    svc.sleep()

    reply = "LIFT-38|work_state|done"
    svc.ext.subjects[reply] = [ResolvedSubject(name="LIFT-38", entity_type="ticket", existing=True,
                                               quote="this ticket", reason="the thread's ticket",
                                               in_focus=True)]
    svc.ext.associations[reply] = [AssociatedSubject(name="LIFT-22", entity_type="ticket",
                                                     reason="earlier followup work, done")]
    svc.ext.meanings[reply] = "LIFT-38 is done"
    reply_id = said(svc, reply, T + 60, "100.0")
    svc.sleep()

    signals, cands, convs = svc.ext.resolve_calls[-1]
    roles = {c.name: c.role for c in cands[0]}
    assert roles["followup-3"] == "focus" and roles["LIFT-38"] == "linked"
    assert convs[0].focus == ("followup-3 (feature)",)
    assert convs[0].recent == ("[1 min earlier] followup-3|requirement|landing link"
                               " (recorded: followup-3 · requirement = landing link)",)
    record = svc.signal(reply_id)
    # extraction never sees the raw conversation; it gets the resolve step's meaning
    extracted = svc.ext.extract_calls[-1][0][0]
    assert extracted.meaning == "LIFT-38 is done"
    assert svc.ext.verify_calls[-1][0][0].meaning == "LIFT-38 is done"
    assert record.metadata["meaning"] == "LIFT-38 is done"

    assert record.context == "slack:C1:100.0" and record.context_parent == "slack:C1"
    assert record.metadata["focus"] == ["followup-3 (feature)"]
    assert record.metadata["associations"][0]["name"] == "LIFT-22"
    assert [(c.entity_name, c.property) for c in record.claims] == [("LIFT-38", "work_state")]
    assert svc.know("LIFT-22") is None                     # recalled, never a claim

    ctx = svc.context("slack:C1:100.0")
    assert [f.name for f in ctx.focus] == ["LIFT-38", "followup-3"]
    assert [x.id for x in ctx.signals][-1] == reply_id


def test_a_new_message_in_the_channel_sees_its_recent_focus_within_the_window(svc):
    first = "followup-3|idea|links"
    svc.ext.subjects[first] = [subject("followup-3", False)]
    said(svc, first, T, "200.0")
    svc.sleep()
    near = "followup-3|idea|more links"
    svc.ext.subjects[near] = [subject("followup-3", True)]
    said(svc, near, T + 600, "201.0")                   # a new thread, 10 minutes later
    far = "followup-3|idea|even more"
    svc.ext.subjects[far] = []
    said(svc, far, T + 6 * 3600, "202.0")               # hours later
    svc.sleep()
    by_text = {sigs[0].text: convs[0] for sigs, _, convs in svc.ext.resolve_calls if convs}
    assert by_text[near].focus == ("followup-3 (feature)",)
    assert by_text[far].focus == ()                     # focus carries over only nearby
    assert by_text[far].recent[-1].startswith("[5 h earlier] ")   # but it is still shown


def test_signals_of_one_conversation_are_read_in_order_in_one_tick(svc):
    a, b = "followup-3|idea|a", "followup-3|idea|b"
    svc.ext.subjects[a] = [subject("followup-3", False)]
    svc.ext.subjects[b] = [subject("followup-3", True)]
    said(svc, b, T + 30, "300.0")                       # deposited first, said second
    said(svc, a, T, "300.0")
    svc.sleep()
    order = [sigs[0].text for sigs, _, _ in svc.ext.resolve_calls]
    assert order == [a, b]
    conv_b = svc.ext.resolve_calls[1][2][0]
    assert conv_b.focus == ("followup-3 (feature)",)   # seen in this tick, not yet written


def test_same_words_in_another_conversation_are_another_signal(svc):
    one = said(svc, "anything|idea|done", T, "400.0")
    two = said(svc, "anything|idea|done", T, "401.0")
    assert one != two
    sig_id, stored, _ = svc.remember("anything|idea|done", "slack", "t")
    assert stored and sig_id not in (one, two)           # no context: the old identity


def test_messages_read_in_one_tick_see_what_the_earlier_ones_recorded(svc):
    a, b = "followup-3|idea|a", "followup-3|decision_state|idea"
    svc.ext.subjects[a] = [subject("followup-3", False)]
    svc.ext.subjects[b] = [subject("followup-3", True)]
    said(svc, a, T, "500.0")
    said(svc, b, T + 30, "500.0")
    svc.sleep()                                   # both in one tick, not yet written
    conv = svc.ext.resolve_calls[-1][2][0]
    assert conv.recent == ("[1 min earlier] followup-3|idea|a (recorded: followup-3 · idea = a)",)


def test_every_model_request_carries_the_timeout(monkeypatch):
    from app import agents
    monkeypatch.setattr(S, "llm_timeout_seconds", 60.0)
    assert agents.model_settings() == {"timeout": 60.0}
    monkeypatch.setattr(S, "llm_timeout_seconds", 0)
    assert agents.model_settings() is None              # 0 = client default
