"""Offline tests for label + detail claims (`detail: true` properties).

The label is the claim's identity; the detail is the author's full wording and
rides next to it. No network, no LLM. Run:  pytest -q   (from etchmem-server/)
"""
from __future__ import annotations

import os
import textwrap

os.environ["EMBEDDING_PROVIDER"] = "fake"

from app import config as _cfg                         # noqa: E402
from app.agents import (                               # noqa: E402
    ClaimExtractor, ConflictResolver, ConflictResolution,
    ExtractedClaim, ExtractionResult,
)
from app.embeddings import FakeEmbedding               # noqa: E402
from app.ext import load_extensions                    # noqa: E402
from app.service import MemoryService                  # noqa: E402

EXT = """
    domain: test
    properties:
      - name: avoid_pattern
        description: Something to avoid.
        detail: true
        label_hint: "e.g. 'core-product framing'"
        detail_hint: "the reviewer's full rule"
      - name: status_note
        description: A plain property without detail.
"""


class Scripted(ClaimExtractor):
    """Signal text is 'label|detail|when' → one avoid_pattern claim."""

    def extract(self, text, known_entities=None):
        parts = text.split("|")
        label, detail = parts[0], parts[1]
        when = parts[2] if len(parts) > 2 else None
        return ExtractionResult(claims=[
            ExtractedClaim(entity_name="Acme", entity_type="company",
                           property="avoid_pattern", value=label, detail=detail,
                           event_time=when),
            # a detail on a property that did not opt in must be dropped
            ExtractedClaim(entity_name="Acme", entity_type="company",
                           property="status_note", value="noted", detail="ignored"),
        ])

    def extract_batch(self, texts, known_entities=None):
        return [self.extract(t) for t in texts]


class Res(ConflictResolver):
    def resolve(self, e, p, c):
        return ConflictResolution(current_value=c[0].value, status="settled",
                                  narrative="n", confidence=0.9)


def _svc(tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "t.yaml").write_text(textwrap.dedent(EXT), encoding="utf-8")
    _cfg.settings.data_dir = str(tmp_path / "data")
    _cfg.settings.ext_dir = str(ext)
    _cfg.settings.extract_min_batch = 1
    _cfg.settings.multi_value_properties = "avoid_pattern"
    return MemoryService(embedder=FakeEmbedding(), extractor=Scripted(), resolver=Res())


def _etch(svc, prop):
    facts = svc.know("Acme", entity_type="company")
    return next(e for e in facts.etches if e.property == prop)


def test_registry_parses_detail_and_prompts_for_it(tmp_path):
    (tmp_path / "t.yaml").write_text(textwrap.dedent(EXT), encoding="utf-8")
    reg = load_extensions(str(tmp_path))
    spec = reg.by_name["avoid_pattern"]
    assert spec.detail and spec.label_hint and spec.detail_hint
    assert reg.has_detail("avoid_pattern") and not reg.has_detail("status_note")
    block = reg.prompt_block()
    assert "WITH DETAIL" in block and "core-product framing" in block
    assert block.count("WITH DETAIL") == 1       # only the opted-in property


def test_label_and_detail_are_stored_and_recalled(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core-product framing|This campaign is about CAST Radar, not core.|2026-07-01",
                 "slack.reject", "t")
    svc.sleep()
    e = _etch(svc, "avoid_pattern")
    assert e.current_value == "core-product framing"           # the label
    assert [d.model_dump() for d in e.details] == [
        {"value": "core-product framing",
         "detail": "This campaign is about CAST Radar, not core."}]
    assert "This campaign is about CAST Radar" in e.narrative   # embedded + returned
    hit = svc.recall("CAST Radar", scope="t", include_signals=False)[0]
    assert hit.details and hit.details[0].detail.startswith("This campaign")


def test_property_without_detail_drops_it(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("x|wording|2026-07-01", "s", "t")
    svc.sleep()
    e = _etch(svc, "status_note")
    assert e.details == []


def test_same_label_new_wording_latest_wins_and_corroborates(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core framing|old wording|2026-07-01", "slack.reject", "t")
    svc.sleep()
    svc.remember("core framing|newer, fuller wording|2026-08-01", "reply.ingest", "t")
    svc.sleep()
    e = _etch(svc, "avoid_pattern")
    assert e.version == 2                                       # wording change is a new version
    assert e.details[0].detail == "newer, fuller wording"
    assert e.confidence > 0.5                                   # two sources, one label
    assert len(e.claim_ids) == 1                                # identity stayed on the label
    hist = svc.history(e.id)
    assert hist[0]["details"][0]["detail"] == "old wording"     # nothing lost


def test_late_older_wording_does_not_overwrite(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core framing|newest wording|2026-09-01", "a", "t")
    svc.sleep()
    svc.remember("core framing|stale wording|2026-01-01", "b", "t")
    svc.sleep()
    assert _etch(svc, "avoid_pattern").details[0].detail == "newest wording"


def test_multiple_labels_each_keep_their_own_wording(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("2100 teams claim|The stat describes core, not Radar.|2026-07-01", "a", "t")
    svc.remember("core framing|Do not frame it as the core product.|2026-07-02", "a", "t")
    svc.sleep()
    e = _etch(svc, "avoid_pattern")
    got = {d.value: d.detail for d in e.details}
    assert got == {"2100 teams claim": "The stat describes core, not Radar.",
                   "core framing": "Do not frame it as the core product."}
    assert set(e.current_value.split(", ")) == set(got)         # labels stay short


def test_time_travel_returns_details_of_that_time(tmp_path):
    svc = _svc(tmp_path)
    svc.remember("core framing|old wording|2026-07-01", "a", "t")
    svc.sleep()
    e1 = _etch(svc, "avoid_pattern")
    svc.remember("core framing|new wording|2026-08-01", "b", "t")
    svc.sleep()
    snap = svc.stores.right.version_as_of(e1.id, e1.updated_at, "event")
    assert snap is not None and "details" in snap
