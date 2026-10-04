"""Offline tests for evidence-based confidence and declared source trust.

Confidence rests on who said it (trust declared in ext `sources:`) and on how many
independent witnesses there are, not on how many source labels exist.
No network, no LLM. Run:  pytest -q   (from etchmem-server/)
"""
from __future__ import annotations

import os
import textwrap

os.environ["EMBEDDING_PROVIDER"] = "fake"

import pytest                                          # noqa: E402

from app import config as _cfg                         # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolver, ConflictResolution,
    ExtractedClaim, ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.ext import load_extensions                    # noqa: E402
from app.gate import _strength                         # noqa: E402
from app.service import MemoryService                  # noqa: E402

EXT = """
    domain: test
    properties:
      - name: avoid_pattern
        description: Something to avoid.
    sources:
      slack.reject:
        trust: 0.9
        description: human reviewer rejected a draft
      reply.ingest: 0.7
      bad.one: 1.5          # out of range: ignored
      worse: not-a-number   # ignored
"""


class Scripted(ClaimExtractor):
    def extract(self, text, known_entities=None):
        return ExtractionResult(claims=[ExtractedClaim(
            entity_name="Acme", entity_type="company", property="avoid_pattern",
            value=text.split("|")[0], confidence=0.7)])

    def extract_batch(self, texts, known_entities=None):
        return [self.extract(t) for t in texts]


class Res(ConflictResolver):
    def resolve(self, e, p, c):
        return ConflictResolution(current_value=c[0].value, status="settled",
                                  narrative="n", confidence=0.9)


def _svc(tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir(parents=True)
    (ext / "t.yaml").write_text(textwrap.dedent(EXT), encoding="utf-8")
    _cfg.settings.data_dir = str(tmp_path / "data")
    _cfg.settings.ext_dir = str(ext)
    _cfg.settings.extract_min_batch = 1
    _cfg.settings.multi_value_properties = "avoid_pattern"
    _cfg.settings.source_trust_json = "{}"
    return MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())


def _etch(svc):
    return next(e for e in svc.know("Acme", entity_type="company").etches
                if e.property == "avoid_pattern")


def test_sources_block_parses_and_rejects_bad_values(tmp_path):
    (tmp_path / "t.yaml").write_text(textwrap.dedent(EXT), encoding="utf-8")
    reg = load_extensions(str(tmp_path))
    assert reg.source_trust == {"slack.reject": 0.9, "reply.ingest": 0.7}
    assert reg.sources[0].description.startswith("human reviewer")


def test_strength_math():
    t = {"human": 0.9, "a": 0.5, "b": 0.5}
    assert _strength({"human": 1}, t) == pytest.approx(0.9)
    assert _strength({"unknown": 1}, t) == pytest.approx(0.5)          # default trust
    assert _strength({"a": 1, "b": 1}, t) == pytest.approx(0.75)       # independent witnesses
    # repeats from ONE source help, but less than a second source, and saturate
    one, three, many = (_strength({"a": n}, t) for n in (1, 3, 50))
    assert one < three < many < _strength({"a": 1, "b": 1}, t) + 0.2
    assert three < _strength({"a": 1, "b": 1, "x": 1}, t)


def test_one_trusted_human_clears_a_point_six_floor(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core framing|x", "slack.reject", "t")
    svc.sleep()
    e = _etch(svc)
    assert e.confidence == pytest.approx(0.9)
    assert e.evidence.max_trust == 0.9
    assert [s.model_dump() for s in e.evidence.sources] == [
        {"source": "slack.reject", "trust": 0.9, "signals": 1}]


def test_unknown_source_stays_at_the_old_single_source_value(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core framing|x", "some.new.channel", "t")
    svc.sleep()
    assert _etch(svc).confidence == pytest.approx(0.5)
    assert svc.stats()["sources_without_trust"] == ["some.new.channel"]


def test_repeats_from_one_source_do_not_count_as_independent(tmp_path):
    svc = _svc(tmp_path)
    for i in range(3):
        svc.remember(f"core framing|variant {i}", "some.new.channel", "t")
        svc.sleep()
    repeated = _etch(svc).confidence
    assert 0.5 < repeated < 0.75          # up from 0.5, below two independent sources

    svc2 = _svc(tmp_path / "two")
    svc2.remember("core framing|a", "some.new.channel", "t")
    svc2.remember("core framing|b", "other.channel", "t")
    svc2.sleep()
    assert _etch(svc2).confidence == pytest.approx(0.75)


def test_one_deposit_with_two_labels_is_one_witness(tmp_path):
    class TwoLabels(Scripted):
        def extract(self, text, known_entities=None):
            return ExtractionResult(claims=[ExtractedClaim(
                entity_name="Acme", entity_type="company", property="avoid_pattern",
                value=v) for v in ("label one", "label two")])
    svc = _svc(tmp_path)
    svc._extractor = TwoLabels()
    svc._pipeline = None
    svc.remember("one rejection", "slack.reject", "t")
    svc.sleep()
    e = _etch(svc)
    assert e.confidence == pytest.approx(0.9) and e.evidence.signals == 1


def test_env_override_beats_the_yaml(tmp_path):
    svc = _svc(tmp_path)
    _cfg.settings.source_trust_json = '{"slack.reject": 0.6}'
    try:
        svc.remember("core framing|x", "slack.reject", "t")
        svc.sleep()
        assert _etch(svc).confidence == pytest.approx(0.6)
        assert svc.stats()["sources_without_trust"] == []
    finally:
        _cfg.settings.source_trust_json = "{}"


def test_evidence_in_recall_history_and_time_travel(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core framing|x", "slack.reject", "t")
    svc.sleep()
    hit = svc.recall("core framing", scope="t", include_signals=False)[0]
    assert hit.evidence and hit.evidence.max_trust == 0.9
    assert svc.history(_etch(svc).id)[0]["evidence"]["max_trust"] == 0.9
