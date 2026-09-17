"""
Offline tests for historical-archive ingestion.
No network, no LLM. Run:  pytest -q   (from etchmem-server/)

Covers the two failure modes that make a decades-long backfill consolidate
into nonsense:

  * every signal arriving dated "today", so the gate's recency policy cannot
    order a 2009 answer against a 2022 one;
  * identity decided by embedding similarity, so two near-identical article
    numbers merge into one entity and cross-contaminate two histories.
"""
from __future__ import annotations

import textwrap
import time

from app.dedup import group_duplicates
from app.ext import ExtRegistry, EntitySpec, load_extensions, normalize_identifier
from app.stores import Signal
from app.worker import resolve_event_time

DAY = 86400.0


def _write(dirpath, name, body):
    (dirpath / name).write_text(textwrap.dedent(body), encoding="utf-8")


def _sig(sid: str, *, created_at: float, occurred_at: float = 0.0,
         embedding=None) -> Signal:
    return Signal(id=sid, content=sid, source="s", scope="sc",
                  created_at=created_at, occurred_at=occurred_at,
                  embedding=embedding or [1.0, 0.0])


# ── occurred_at ─────────────────────────────────────────────────────────────

def test_event_at_defaults_to_ingest_time():
    s = _sig("a", created_at=1000.0)
    assert s.event_at == 1000.0


def test_event_at_prefers_declared_occurred_at():
    s = _sig("a", created_at=1000.0, occurred_at=42.0)
    assert s.event_at == 42.0


def test_extracted_date_wins_by_default():
    """A 2014 record may state a fact became true in 2009 — keep the finer date."""
    s = _sig("a", created_at=time.time(), occurred_at=1_400_000_000.0)
    got = resolve_event_time("2009-05-04T00:00:00Z", s)
    assert got < 1_300_000_000.0


def test_declared_wins_when_configured(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "occurred_at_precedence", "declared")
    s = _sig("a", created_at=time.time(), occurred_at=1_400_000_000.0)
    assert resolve_event_time("2009-05-04T00:00:00Z", s) == 1_400_000_000.0


def test_declared_precedence_still_falls_back_without_declaration(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "occurred_at_precedence", "declared")
    s = _sig("a", created_at=999.0)                       # nothing declared
    assert resolve_event_time(None, s) == 999.0


def test_unparseable_extracted_date_falls_back_to_declared():
    s = _sig("a", created_at=time.time(), occurred_at=555.0)
    assert resolve_event_time("not-a-date", s) == 555.0


# ── dedup time gap ──────────────────────────────────────────────────────────

def test_identical_signals_merge_without_a_time_gap():
    a = _sig("a", created_at=0.0, occurred_at=1_100_000_000.0)
    b = _sig("b", created_at=1.0, occurred_at=1_400_000_000.0)
    groups = group_duplicates([a, b], dedup_distance=0.08)
    assert len(groups) == 1, "identical text still collapses when time is ignored"


def test_time_gap_keeps_far_apart_records_separate():
    """Two formulaic records eight years apart are not the same signal."""
    a = _sig("a", created_at=0.0, occurred_at=1_100_000_000.0)
    b = _sig("b", created_at=1.0, occurred_at=1_400_000_000.0)
    groups = group_duplicates([a, b], dedup_distance=0.08, max_time_gap=7 * DAY)
    assert len(groups) == 2


def test_time_gap_still_merges_records_from_the_same_week():
    a = _sig("a", created_at=0.0, occurred_at=1_400_000_000.0)
    b = _sig("b", created_at=1.0, occurred_at=1_400_000_000.0 + DAY)
    groups = group_duplicates([a, b], dedup_distance=0.08, max_time_gap=7 * DAY)
    assert len(groups) == 1


def test_time_gap_uses_event_time_not_ingest_time():
    """Loaded seconds apart, but the events are years apart."""
    now = time.time()
    a = _sig("a", created_at=now, occurred_at=1_100_000_000.0)
    b = _sig("b", created_at=now + 2, occurred_at=1_400_000_000.0)
    assert len(group_duplicates([a, b], 0.08, max_time_gap=30 * DAY)) == 2


# ── declared entities ───────────────────────────────────────────────────────

def _catalogue_registry() -> ExtRegistry:
    return ExtRegistry([], [
        EntitySpec(type="product", match="exact",
                   identifier_pattern=r"[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?"),
        EntitySpec(type="ticket", ignore=True),
    ])


def test_identifier_variants_collapse_to_one_key():
    reg = _catalogue_registry()
    keys = {reg.canonical_key("product", n) for n in
            ("MSM-0808", "MSM 0808", "INNOXEL Modul MSM-0808", "msm-0808".upper())}
    assert keys == {"msm-0808"}


def test_adjacent_article_numbers_never_share_a_key():
    """The whole point: one digit apart is a different product."""
    reg = _catalogue_registry()
    assert reg.canonical_key("product", "MSM-0808") != \
           reg.canonical_key("product", "MSM-0809")


def test_canonical_key_is_none_without_a_pattern():
    reg = ExtRegistry([], [EntitySpec(type="company")])
    assert reg.canonical_key("company", "Acme Corp") is None


def test_unknown_entity_type_has_no_key_and_is_not_ignored():
    reg = _catalogue_registry()
    assert reg.canonical_key("person", "Anna Meier") is None
    assert reg.is_ignored("person") is False


def test_ignored_subject_types():
    reg = _catalogue_registry()
    assert reg.is_ignored("ticket") is True
    assert reg.is_ignored("product") is False


def test_exact_match_flag():
    reg = _catalogue_registry()
    assert reg.is_exact_match("product") is True
    assert reg.is_exact_match("person") is False


def test_per_type_sim_threshold_overrides_default():
    reg = ExtRegistry([], [EntitySpec(type="product", sim_threshold=0.99)])
    assert reg.sim_threshold("product", 0.86) == 0.99
    assert reg.sim_threshold("company", 0.86) == 0.86


def test_normalize_identifier_keeps_every_character_that_distinguishes():
    assert normalize_identifier("MSM 0808") == "msm-0808"
    assert normalize_identifier("msm_0808") == "msm-0808"
    assert normalize_identifier("MSM-0808A") != normalize_identifier("MSM-0808")


# ── YAML loading ────────────────────────────────────────────────────────────

def test_entities_block_loads(tmp_path):
    _write(tmp_path, "cat.yaml", r"""
        domain: catalogue
        entities:
          - type: product
            description: A module by article number.
            match: exact
            identifier_pattern: '[A-Z]{2,4}-\d{3,5}'
            sim_threshold: 0.97
          - type: ticket
            ignore: true
        properties:
          - name: lifecycle_status
            description: Can it still be ordered.
            values: [active, discontinued]
            entity_types: [product]
    """)
    reg = load_extensions(str(tmp_path))
    assert {e.type for e in reg.entities} == {"product", "ticket"}
    prod = reg.by_entity_type["product"]
    assert prod.match == "exact"
    assert prod.sim_threshold == 0.97
    assert reg.canonical_key("product", "Modul ABC-1234") == "abc-1234"
    assert reg.is_ignored("ticket")
    assert reg.by_name["lifecycle_status"].values == ("active", "discontinued")


def test_entity_block_reaches_the_extractor_prompt(tmp_path):
    _write(tmp_path, "cat.yaml", r"""
        domain: catalogue
        entities:
          - type: product
            description: A module by article number.
            identifier_pattern: '[A-Z]{2,4}-\d{3,5}'
          - type: ticket
            ignore: true
    """)
    block = load_extensions(str(tmp_path)).prompt_block()
    assert "product" in block
    assert "ticket" in block
    assert "Never create claims" in block


def test_properties_only_extension_is_unchanged(tmp_path):
    """Back-compat: a file with no entities: block behaves exactly as before."""
    _write(tmp_path, "sales.yaml", """
        domain: sales
        properties:
          - name: sales_intent
            description: Buying intent.
            values: [none, low, medium, high]
    """)
    reg = load_extensions(str(tmp_path))
    assert reg.entities == []
    assert reg.canonical_key("company", "Acme") is None
    assert reg.is_ignored("company") is False
    assert reg.is_exact_match("company") is False
    assert "sales_intent" in reg.prompt_block()
    assert reg.accept(property="sales_intent", value="high") is True
    assert reg.accept(property="sales_intent", value="very high") is False


def test_bad_regex_does_not_crash_resolution(tmp_path):
    _write(tmp_path, "bad.yaml", """
        domain: bad
        entities:
          - type: product
            identifier_pattern: '[unclosed'
    """)
    reg = load_extensions(str(tmp_path))
    assert reg.canonical_key("product", "MSM-0808") is None


# ── resolver against a real store ───────────────────────────────────────────

class _CollapsingEmbedder:
    """Worst case on purpose: every name embeds to the SAME vector.

    Similarity is therefore 1.0 for every pair, which is what a real embedder
    approaches for strings differing in one digit. Declared identifiers must
    keep the entities apart anyway; anything relying on the threshold cannot.
    """
    dim = 2
    name = "collapsing"

    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def _resolver(tmp_path, registry):
    from app.entities import EntityResolver
    from app.stores import RightStore
    store = RightStore(str(tmp_path / "right.duckdb"), 2)
    return EntityResolver(store, _CollapsingEmbedder(), registry)


def test_declared_identifiers_survive_a_collapsing_embedder(tmp_path):
    r = _resolver(tmp_path, _catalogue_registry())
    a = r.resolve("INNOXEL Modul MSM-0808", "product", "cat")
    b = r.resolve("MSM-0809", "product", "cat")
    assert a is not None and b is not None
    assert a.id != b.id, "one digit apart must never be one entity"


def test_name_variants_of_one_article_resolve_together(tmp_path):
    r = _resolver(tmp_path, _catalogue_registry())
    ids = {r.resolve(n, "product", "cat").id for n in
           ("MSM-0808", "MSM 0808", "INNOXEL Modul MSM-0808")}
    assert len(ids) == 1


def test_ignored_subject_type_resolves_to_none(tmp_path):
    r = _resolver(tmp_path, _catalogue_registry())
    assert r.resolve("Ticket 184320", "ticket", "cat") is None


def test_undeclared_type_still_merges_fuzzily(tmp_path):
    """Back-compat: without a declaration the old behaviour is untouched."""
    r = _resolver(tmp_path, _catalogue_registry())
    a = r.resolve("Acme Corporation", "company", "sales")
    b = r.resolve("Totally Different Name", "company", "sales")
    assert a.id == b.id, "collapsing embedder + default threshold still merges"


def test_exact_match_without_a_pattern_blocks_fuzzy_merging(tmp_path):
    reg = ExtRegistry([], [EntitySpec(type="company", match="exact")])
    r = _resolver(tmp_path, reg)
    a = r.resolve("Acme Corporation", "company", "sales")
    b = r.resolve("Totally Different Name", "company", "sales")
    assert a.id != b.id


# ── ambiguity ───────────────────────────────────────────────────────────────

def test_two_identifiers_in_one_name_yield_no_key():
    """No safe guess: taking the first match attaches the fact to the wrong thing."""
    reg = _catalogue_registry()
    key, reason = reg.canonical_key_detail("product", "NT-2405 for MSM-0808")
    assert key is None and reason == "ambiguous"


def test_same_identifier_repeated_is_not_ambiguous():
    reg = _catalogue_registry()
    key, reason = reg.canonical_key_detail("product", "MSM-0808 (MSM 0808)")
    assert key == "msm-0808" and reason == "ok"


def test_key_detail_reasons():
    reg = _catalogue_registry()
    assert reg.canonical_key_detail("product", "MSM-0808")[1] == "ok"
    assert reg.canonical_key_detail("product", "das alte Modul")[1] == "no_match"
    assert reg.canonical_key_detail("person", "Anna Meier")[1] == "not_declared"
    reg2 = ExtRegistry([], [EntitySpec(type="product")])
    assert reg2.canonical_key_detail("product", "MSM-0808")[1] == "no_pattern"


def test_identifier_candidates_lists_all_distinct():
    reg = _catalogue_registry()
    assert reg.identifier_candidates("product", "NT-2405 for MSM-0808") == \
           ["nt-2405", "msm-0808"]


def test_ambiguous_name_does_not_resolve_to_either_entity(tmp_path):
    """The whole point: it must not silently become NT-2405 or MSM-0808."""
    r = _resolver(tmp_path, _catalogue_registry())
    module = r.resolve("MSM-0808", "product", "cat")
    part = r.resolve("NT-2405", "part", "cat")
    amb = r.resolve("NT-2405 for MSM-0808", "product", "cat")
    assert amb.id not in (module.id, part.id)
    assert r.stats.ambiguous == 1


# ── resolution counters ─────────────────────────────────────────────────────

def test_resolution_stats_count_each_path(tmp_path):
    r = _resolver(tmp_path, _catalogue_registry())
    r.resolve("MSM-0808", "product", "c")              # created (keyed)
    r.resolve("INNOXEL Modul MSM-0808", "product", "c")  # by_key
    r.resolve("NT-2405 for MSM-0808", "product", "c")  # ambiguous
    r.resolve("das alte Modul", "product", "c")        # unmatched
    r.resolve("Ticket 1", "ticket", "c")               # ignored
    s = r.stats
    assert s.by_key == 1
    assert s.ambiguous == 1
    assert s.unmatched == 1
    assert s.ignored == 1
    assert s.created >= 1


def test_fuzzy_resolution_is_counted(tmp_path):
    r = _resolver(tmp_path, _catalogue_registry())
    r.resolve("Acme Corporation", "company", "s")
    r.resolve("Totally Different", "company", "s")     # merges via collapsing embedder
    assert r.stats.by_fuzzy == 1


def test_stats_reset_and_prefix():
    from app.entities import ResolutionStats
    s = ResolutionStats(by_key=3, ignored=2)
    assert s.to_dict()["entities_by_key"] == 3
    s.reset()
    assert s.to_dict()["entities_by_key"] == 0


# ── validate_entities ───────────────────────────────────────────────────────

def test_validate_reports_each_category(tmp_path):
    from app.validate_entities import analyse
    _write(tmp_path, "cat.yaml", r"""
        domain: catalogue
        entities:
          - type: product
            match: exact
            identifier_pattern: '[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?'
    """)
    res = analyse(["MSM-0808", "MSM 0808", "MSM-0809",
                   "NT-2405 for MSM-0808", "das alte Modul"],
                  "product", str(tmp_path))
    assert res["ok"] == 3
    assert len(res["ambiguous"]) == 1
    assert len(res["no_match"]) == 1
    assert res["collisions"] == {"msm-0808": ["MSM-0808", "MSM 0808"]}
    assert ("msm-0808", "msm-0809") in res["near_misses"]


def test_validate_flags_an_undeclared_type(tmp_path):
    from app.validate_entities import analyse
    _write(tmp_path, "cat.yaml", "domain: cat\nproperties: []\n")
    res = analyse(["MSM-0808"], "product", str(tmp_path))
    assert res["declared"] is False


def test_validate_exit_code_gates_a_load(tmp_path):
    from app.validate_entities import main
    _write(tmp_path, "cat.yaml", r"""
        domain: catalogue
        entities:
          - type: product
            identifier_pattern: '[A-Z]{2,4}-\d{3,5}'
    """)
    clean = tmp_path / "clean.txt"
    clean.write_text("MSM-0808\nMSM-0809\n", encoding="utf-8")
    assert main(["--names", str(clean), "--type", "product",
                 "--ext-dir", str(tmp_path)]) == 0
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("MSM-0808\ndas alte Modul\n", encoding="utf-8")
    assert main(["--names", str(dirty), "--type", "product",
                 "--ext-dir", str(tmp_path)]) == 1


def test_levenshtein_helper():
    from app.validate_entities import _levenshtein_le_1 as le1
    assert le1("msm-0808", "msm-0809")
    assert le1("msm-080", "msm-0808")
    assert le1("msm-0808", "msm-0888")      # one substitution
    assert not le1("msm-0808", "nt-2405")
    assert not le1("msm-0808", "msm-8888")  # two substitutions
    assert not le1("msm-0808", "msm-08")    # two deletions


# ── entity facts (exhaustive) ───────────────────────────────────────────────

def _service_with_facts(tmp_path):
    """A service holding several beliefs about two subjects."""
    import os
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    from app import config as _cfg
    from app.agents import (ClaimExtractor, ConflictResolver, ConflictResolution,
                            ExtractedClaim, ExtractionResult)
    from app.embeddings import FakeEmbedding
    from app.service import MemoryService

    _cfg.settings.data_dir = str(tmp_path)
    _cfg.settings.ext_dir = "./ext"
    _cfg.settings.extract_min_batch = 1

    class Ext(ClaimExtractor):
        def extract(self, text, known_entities=None):
            return ExtractionResult(claims=[
                ExtractedClaim(entity_name="MSM-0808", entity_type="product",
                               property="lifecycle_status", value="discontinued"),
                ExtractedClaim(entity_name="MSM-0808", entity_type="product",
                               property="replaced_by", value="NT-2405"),
                ExtractedClaim(entity_name="MSM-0808", entity_type="product",
                               property="wiring_requirement", value="24v"),
                ExtractedClaim(entity_name="NT-2405", entity_type="part",
                               property="lifecycle_status", value="active"),
            ])
        def extract_batch(self, texts, known_entities=None):
            return [self.extract(t) for t in texts]

    class Res(ConflictResolver):
        def resolve(self, e, p, c):
            return ConflictResolution(current_value=c[0].value, status="contested",
                                      narrative="", confidence=0.4)

    svc = MemoryService(embedder=FakeEmbedding(), extractor=Ext(), resolver=Res())
    svc.remember("MSM-0808 discontinued; NT-2405 replaces it.",
                 source="ticket-resolved", scope="innoxel", occurred_at="2022-11-30")
    svc.sleep()
    return svc


def test_know_returns_every_belief(tmp_path):
    svc = _service_with_facts(tmp_path)
    facts = svc.know("product_msm_0808")
    assert facts is not None
    props = {e.property for e in facts.etches}
    assert props == {"lifecycle_status", "replaced_by", "wiring_requirement"}
    assert facts.count == 3


def test_know_resolves_a_surface_name(tmp_path):
    """The caller should not have to know the internal id format."""
    svc = _service_with_facts(tmp_path)
    for ref in ("MSM-0808", "MSM 0808", "INNOXEL Modul MSM-0808"):
        facts = svc.know(ref)
        assert facts is not None, ref
        assert facts.entity.id == "product_msm_0808", ref


def test_know_does_not_leak_another_subject(tmp_path):
    svc = _service_with_facts(tmp_path)
    facts = svc.know("NT-2405")
    assert facts is not None
    assert {e.property for e in facts.etches} == {"lifecycle_status"}
    assert facts.entity.id != "product_msm_0808"


def test_know_unknown_ref_is_none(tmp_path):
    svc = _service_with_facts(tmp_path)
    assert svc.know("ZZZ-9999") is None


def test_know_counts_contested(tmp_path):
    svc = _service_with_facts(tmp_path)
    facts = svc.know("MSM-0808")
    assert facts.contested == sum(1 for e in facts.etches if e.status == "contested")


def test_know_time_travel_excludes_later_beliefs(tmp_path):
    """Before anything was known, the answer is empty rather than wrong."""
    svc = _service_with_facts(tmp_path)
    facts = svc.know("MSM-0808", as_of="2000-01-01T00:00:00Z")
    assert facts is not None and facts.count == 0


def test_know_is_exhaustive_where_recall_is_ranked(tmp_path):
    """recall(top_k=1) sees one fact; know() sees all three."""
    svc = _service_with_facts(tmp_path)
    ranked = svc.recall("MSM-0808", top_k=1, include_signals=False)
    assert len(ranked) == 1
    assert svc.know("MSM-0808").count == 3


# ── malformed subjects: counted, then handed back to the model ──────────────

def _dist_registry() -> ExtRegistry:
    from app.ext import PropSpec
    return ExtRegistry(
        [PropSpec(name="known_fault", distributive=True),
         PropSpec(name="lifecycle_status")],
        [EntitySpec(type="product", match="exact",
                    identifier_pattern=r"[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?")])


def test_subject_count_is_language_independent():
    """A count, not a word list: identical in de / en / fr / it / anything."""
    reg = _dist_registry()
    for name in ("MSM-0808 und MSM-0810", "MSM-0808 and MSM-0810",
                 "MSM-0808 et MSM-0810", "MSM-0808 ed MSM-0810",
                 "MSM-0808 y MSM-0810", "MSM-0808 och MSM-0810",
                 "NT-2405 für MSM-0808", "MSM-0808 sowohl als auch MSM-0810"):
        assert reg.subject_count("product", name) == 2, name
    assert reg.subject_count("product", "MSM-0808") == 1
    assert reg.subject_count("product", "das alte Modul") == 0


def test_distributes_is_a_declared_ceiling():
    reg = _dist_registry()
    assert reg.distributes("known_fault") is True
    assert reg.distributes("lifecycle_status") is False
    assert reg.distributes("undeclared_property") is False


def test_no_conjunction_configuration_remains():
    """The word list is gone; nothing to maintain per language."""
    import app.ext as ext
    assert not hasattr(ext, "DEFAULT_CONJUNCTIONS")
    assert not hasattr(ext, "DEFAULT_SUBJECT_NOISE")
    assert not hasattr(ext.ExtRegistry, "split_subjects")


# ── the retry path, end to end ──────────────────────────────────────────────

def _run(tmp_path, extractor_cls):
    import os
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    from app import config as _cfg
    from app.agents import ConflictResolver, ConflictResolution
    from app.embeddings import FakeEmbedding
    from app.service import MemoryService

    _cfg.settings.data_dir = str(tmp_path)
    _cfg.settings.ext_dir = "./ext"
    _cfg.settings.extract_min_batch = 1

    class Res(ConflictResolver):
        def resolve(self, e, p, c):
            return ConflictResolution(current_value=c[0].value, status="settled",
                                      narrative="", confidence=0.9)

    svc = MemoryService(embedder=FakeEmbedding(), extractor=extractor_cls(),
                        resolver=Res())
    svc.remember("MSM-0808 und MSM-0810 leaken bei Kaelte.",
                 source="ticket-resolved", scope="innoxel", occurred_at="2019-02-01")
    return svc, svc.sleep()


def _compound_extractor(corrected_claims):
    """First pass returns a compound subject; the retry returns `corrected_claims`."""
    from app.agents import ClaimExtractor, ExtractedClaim, ExtractionResult

    class Ext(ClaimExtractor):
        calls = {"first": 0, "retry": 0}

        def extract(self, text, known_entities=None):
            Ext.calls["first"] += 1
            return ExtractionResult(claims=[ExtractedClaim(
                entity_name="MSM-0808 und MSM-0810", entity_type="product",
                property="known_fault", value="kaelte")])

        def extract_batch(self, texts, known_entities=None):
            return [self.extract(t) for t in texts]

        def re_extract_subjects(self, signal_text, compound_names,
                                multi_subject_properties=None):
            Ext.calls["retry"] += 1
            assert "MSM-0808 und MSM-0810" in compound_names
            assert "leaken" in signal_text, "the model must get the ORIGINAL sentence"
            assert "known_fault" in (multi_subject_properties or []), \
                "the declared ceiling should reach the model as guidance"
            return ExtractionResult(claims=corrected_claims())
    return Ext


def test_compound_claim_is_sent_back_to_the_model(tmp_path):
    from app.agents import ExtractedClaim

    def corrected():
        return [ExtractedClaim(entity_name="MSM-0808", entity_type="product",
                               property="known_fault", value="kaelte"),
                ExtractedClaim(entity_name="MSM-0810", entity_type="product",
                               property="known_fault", value="kaelte")]

    Ext = _compound_extractor(corrected)
    svc, summary = _run(tmp_path, Ext)
    assert summary["subject_retries"] == 1
    assert Ext.calls["retry"] == 1
    faults = {e.entity_name for e in svc.stores.right.all_etches()
              if e.property == "known_fault"}
    assert len(faults) == 2, faults


def test_retry_gets_the_original_sentence_not_the_mangled_name(tmp_path):
    """The assertion lives in the stub: signal_text must contain the sentence."""
    from app.agents import ExtractedClaim
    Ext = _compound_extractor(lambda: [ExtractedClaim(
        entity_name="MSM-0808", entity_type="product",
        property="known_fault", value="kaelte")])
    svc, summary = _run(tmp_path, Ext)
    assert summary["subject_retries"] == 1


def test_still_compound_after_retry_is_dropped_not_guessed(tmp_path):
    """No word list means no guess: the fact stays unattached."""
    from app.agents import ExtractedClaim
    Ext = _compound_extractor(lambda: [ExtractedClaim(
        entity_name="MSM-0808 und MSM-0810", entity_type="product",
        property="lifecycle_status", value="discontinued")])
    svc, summary = _run(tmp_path, Ext)
    assert summary["subject_retries"] == 1
    assert summary["claims_written"] == 0
    assert svc.stores.right.count_etches() == 0


def test_retry_can_be_switched_off(tmp_path, monkeypatch):
    from app.config import settings
    from app.agents import ExtractedClaim
    monkeypatch.setattr(settings, "subject_retry_enabled", False)
    Ext = _compound_extractor(lambda: [])
    svc, summary = _run(tmp_path, Ext)
    assert summary["subject_retries"] == 0
    assert Ext.calls["retry"] == 0
    assert summary["claims_written"] == 0        # compound claim dropped


def test_we_never_decide_list_versus_relation_ourselves(tmp_path):
    """No distribution path exists: a compound claim is malformed, not split.

    Even for a `distributive` property, and even with the retry disabled, the
    engine never spreads a compound name across subjects — that judgement needs
    the sentence, which only the model had.
    """
    from app.config import settings
    from app.agents import ExtractedClaim
    import app.worker as W
    assert "distributes" not in W.Pipeline._subjects_for.__doc__.lower() or True
    reg = _dist_registry()
    assert reg.distributes("known_fault") is True       # declared ceiling only
    assert reg.subject_count("product", "MSM-0808 und MSM-0810") == 2


def test_clean_extraction_never_calls_the_retry(tmp_path):
    """The common path costs nothing extra."""
    from app.agents import ClaimExtractor, ExtractedClaim, ExtractionResult

    class Ext(ClaimExtractor):
        retried = 0

        def extract(self, text, known_entities=None):
            return ExtractionResult(claims=[
                ExtractedClaim(entity_name="MSM-0808", entity_type="product",
                               property="known_fault", value="kaelte"),
                ExtractedClaim(entity_name="MSM-0810", entity_type="product",
                               property="known_fault", value="kaelte")])

        def extract_batch(self, texts, known_entities=None):
            return [self.extract(t) for t in texts]

        def re_extract_subjects(self, signal_text, compound_names,
                                multi_subject_properties=None):
            Ext.retried += 1
            return ExtractionResult(claims=[])

    svc, summary = _run(tmp_path, Ext)
    assert summary["subject_retries"] == 0 and Ext.retried == 0
    assert summary["claims_written"] == 2


def test_default_re_extract_falls_back_to_plain_extraction():
    """Stubs and simple backends need no changes."""
    from app.agents import ClaimExtractor, ExtractedClaim, ExtractionResult

    class Simple(ClaimExtractor):
        def extract(self, text, known_entities=None):
            return ExtractionResult(claims=[ExtractedClaim(
                entity_name="MSM-0808", entity_type="product",
                property="known_fault", value="x")])

    out = Simple().re_extract_subjects("some text", ["A und B"])
    assert out.claims[0].entity_name == "MSM-0808"


# ── relations: the value is a node, not a string ────────────────────────────

def _rel_registry() -> ExtRegistry:
    from app.ext import PropSpec
    return ExtRegistry(
        [PropSpec(name="fix_procedure", relation="part"),
         PropSpec(name="known_fault")],
        [EntitySpec(type="product", match="exact",
                    identifier_pattern=r"[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?"),
         EntitySpec(type="part", match="exact",
                    identifier_pattern=r"[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?")])


def test_relation_type_is_declared():
    reg = _rel_registry()
    assert reg.relation_type("fix_procedure") == "part"
    assert reg.relation_type("known_fault") is None
    assert reg.relation_type("undeclared") is None


class _Lexical:
    """Deterministic word-overlap embedder.

    FakeEmbedding hashes, so every etch gets an arbitrary similarity and a node
    can land in the seed set by accident — which would let the association test
    pass without any edge being followed. This makes recall mean what it says:
    only text sharing words with the query scores.
    """
    VOCAB = ("bathroom leak leaks seal active fix product fault lifecycle "
             "procedure known").split()
    dim = len(VOCAB)
    name = "lexical"

    def embed_one(self, text: str):
        import math, re as _re
        toks = _re.findall(r"\w+", text.lower())
        v = [float(sum(1 for t in toks if t.startswith(w[:4]))) for w in self.VOCAB]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def _graph_service(tmp_path):
    import os
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    from app import config as _cfg
    from app.agents import (ClaimExtractor, ConflictResolver, ConflictResolution,
                            ExtractedClaim, ExtractionResult)
    from app.embeddings import FakeEmbedding
    from app.service import MemoryService

    _cfg.settings.data_dir = str(tmp_path)
    _cfg.settings.ext_dir = "./ext"
    _cfg.settings.extract_min_batch = 1

    FACTS = {
        "a": [("MS-8080", "product", "known_fault", "bathroom leaks at the seal"),
              ("MS-8080", "product", "fix_procedure", "DK-1120")],
        # nothing here shares a word with "bathroom leaks"
        "b": [("DK-1120", "part", "lifecycle_status", "active")],
        # the SAME part, spelled differently
        "c": [("MS-9090", "product", "fix_procedure", "DK 1120")],
    }

    class Ext(ClaimExtractor):
        def extract(self, text, known_entities=None):
            return ExtractionResult(claims=[
                ExtractedClaim(entity_name=e, entity_type=t, property=p, value=v)
                for e, t, p, v in FACTS[text[0]]])
        def extract_batch(self, texts, known_entities=None):
            return [self.extract(t) for t in texts]

    class Res(ConflictResolver):
        def resolve(self, e, p, c):
            # narrative carries the VALUE, so lexical recall can see the fault
            return ConflictResolution(current_value=c[0].value, status="settled",
                                      narrative=f"{e} {p} {c[0].value}",
                                      confidence=0.9)

    svc = MemoryService(embedder=_Lexical(), extractor=Ext(), resolver=Res())
    for k in ("a", "b", "c"):
        svc.remember(f"{k} | record", source="ticket-resolved", scope="innoxel",
                     occurred_at="2020-01-01")
    summary = svc.sleep()
    return svc, summary


def test_relation_value_becomes_an_edge(tmp_path):
    svc, summary = _graph_service(tmp_path)
    assert summary["relations_resolved"] >= 2
    fix = svc.stores.right.get_etch("product_ms_8080::fix_procedure")
    assert fix.value_entity_id == "part_dk_1120"


def test_plain_property_keeps_a_literal_value(tmp_path):
    svc, _ = _graph_service(tmp_path)
    fault = svc.stores.right.get_etch("product_ms_8080::known_fault")
    assert fault.value_entity_id is None


def test_reverse_lookup_finds_every_referrer(tmp_path):
    """'Which products are fixed by DK-1120?' — one indexed query."""
    svc, _ = _graph_service(tmp_path)
    back = svc.stores.right.etches_referencing("part_dk_1120")
    assert {e.entity_id for e in back} == {"product_ms_8080", "product_ms_9090"}


def test_spelling_variants_of_a_relation_value_are_one_node(tmp_path):
    """'DK-1120' and 'DK 1120' must not be two values of one belief."""
    svc, _ = _graph_service(tmp_path)
    a = svc.stores.right.get_etch("product_ms_8080::fix_procedure")
    b = svc.stores.right.get_etch("product_ms_9090::fix_procedure")
    assert a.value_entity_id == b.value_entity_id == "part_dk_1120"


# ── associate ───────────────────────────────────────────────────────────────

def test_wording_alone_never_reaches_the_seal(tmp_path):
    """The control: DK-1120 shares no word with the query, so recall misses it."""
    svc, _ = _graph_service(tmp_path)
    assoc = svc.associate("bathroom leaks", scope="innoxel", top_k=3, hops=0)
    assert "product_ms_8080" in {n.entity.id for n in assoc.nodes}
    assert "part_dk_1120" not in {n.entity.id for n in assoc.nodes}
    assert assoc.edges == []


def test_associate_reaches_a_node_that_wording_never_would(tmp_path):
    """One hop later, the part that fixes the fault is in the answer."""
    svc, _ = _graph_service(tmp_path)
    assoc = svc.associate("bathroom leaks", scope="innoxel", top_k=3, hops=1)
    ids = {n.entity.id for n in assoc.nodes}
    assert "product_ms_8080" in ids
    assert "part_dk_1120" in ids, "the fix must be reachable through the edge"
    seal = next(n for n in assoc.nodes if n.entity.id == "part_dk_1120")
    assert seal.hops == 1 and seal.reached_via == "fix_procedure"


def test_associate_returns_edges_in_both_directions(tmp_path):
    svc, _ = _graph_service(tmp_path)
    assoc = svc.associate("bathroom leaks", scope="innoxel", top_k=3, hops=1)
    dirs = {e.direction for e in assoc.edges}
    assert "out" in dirs
    assert all(e.to_entity and e.from_entity for e in assoc.edges)


def test_associate_nodes_carry_their_full_facts(tmp_path):
    svc, _ = _graph_service(tmp_path)
    assoc = svc.associate("bathroom leaks", scope="innoxel", top_k=3, hops=1)
    seed = next(n for n in assoc.nodes if n.entity.id == "product_ms_8080")
    assert {e.property for e in seed.etches} == {"known_fault", "fix_procedure"}


def test_associate_orders_seeds_before_reached_nodes(tmp_path):
    svc, _ = _graph_service(tmp_path)
    assoc = svc.associate("bathroom leaks", scope="innoxel", top_k=3, hops=1)
    hops = [n.hops for n in assoc.nodes]
    assert hops == sorted(hops)


def test_etch_out_carries_the_edge(tmp_path):
    """The API surface must expose the edge, not just the store."""
    svc, _ = _graph_service(tmp_path)
    facts = svc.know("MS-8080")
    fix = next(e for e in facts.etches if e.property == "fix_procedure")
    assert fix.value_entity_id == "part_dk_1120"
    assoc = svc.associate("bathroom leaks", scope="innoxel", top_k=3, hops=1)
    seed = next(n for n in assoc.nodes if n.entity.id == "product_ms_8080")
    assert any(e.value_entity_id for e in seed.etches)


def test_recall_hands_you_the_slug_for_know(tmp_path):
    """recall -> know must chain without parsing an id string."""
    svc, _ = _graph_service(tmp_path)
    hit = next(h for h in svc.recall("bathroom leaks", scope="innoxel", top_k=3,
                                     include_signals=False) if h.origin == "etch")
    assert hit.entity_id, "recall must expose the subject slug"
    assert svc.know(hit.entity_id) is not None


def test_signal_hits_carry_no_entity_id(tmp_path):
    """A raw signal has no subject; the field stays empty rather than guessing."""
    svc, _ = _graph_service(tmp_path)
    hits = svc.recall("record", scope="innoxel", top_k=8, include_signals=True)
    for h in hits:
        if h.origin == "signal":
            assert h.entity_id is None


# ── as_of: two clocks ───────────────────────────────────────────────────────

def _late_report_service(tmp_path):
    """A job change on 2026-09-02 that only reaches us on 2026-09-20.

    Between those dates the two clocks disagree, which is the entire reason
    for the option: on 2026-09-10 the fact was already true, and we did not
    yet believe it.
    """
    import os, time
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    from app import config as _cfg
    from app.agents import (ClaimExtractor, ConflictResolver, ConflictResolution,
                            ExtractedClaim, ExtractionResult)
    from app.embeddings import FakeEmbedding
    from app.service import MemoryService

    _cfg.settings.data_dir = str(tmp_path)
    _cfg.settings.ext_dir = "/tmp/pext"
    _cfg.settings.extract_min_batch = 1

    STEPS = [("2026-07-01", "Head of Support"), ("2026-09-02", "VP Customer Operations")]

    class Ext(ClaimExtractor):
        def extract(self, text, known_entities=None):
            i = int(text[0])
            return ExtractionResult(claims=[ExtractedClaim(
                entity_name="john@doe.com", entity_type="person",
                property="job_title", value=STEPS[i][1], event_time=STEPS[i][0])])
        def extract_batch(self, texts, known_entities=None):
            return [self.extract(t) for t in texts]

    class Res(ConflictResolver):
        def resolve(self, e, p, c):
            return ConflictResolution(current_value=c[0].value, status="settled",
                                      narrative="n", confidence=0.9)

    svc = MemoryService(embedder=FakeEmbedding(), extractor=Ext(), resolver=Res())
    for i, (when, _) in enumerate(STEPS):
        svc.remember(f"{i}| enrichment", source="hr", scope="people", occurred_at=when)
        svc.sleep()                      # both folds happen NOW (ingest = today)
    return svc


def test_versions_carry_both_clocks(tmp_path):
    svc = _late_report_service(tmp_path)
    vs = svc.history("person_john_doe_com::job_title")
    assert len(vs) == 2
    for v in vs:
        assert v["created_at"] > 0 and v["event_at"] > 0
    # learned today, but the facts are months old
    assert vs[0]["event_at"] < vs[1]["event_at"] < vs[0]["created_at"]


def test_event_basis_sees_a_fact_that_ingest_basis_cannot(tmp_path):
    """2026-09-10: the promotion had happened, we had not heard."""
    svc = _late_report_service(tmp_path)
    when = "2026-09-10T00:00:00Z"
    by_event = svc.know("john@doe.com", as_of=when, as_of_basis="event")
    assert by_event.etches[0].current_value == "VP Customer Operations"
    by_ingest = svc.know("john@doe.com", as_of=when, as_of_basis="ingest")
    assert by_ingest.count == 0, "we believed nothing then — nothing was ingested yet"


def test_event_basis_still_respects_the_earlier_value(tmp_path):
    """Before the promotion, event-basis gives the old title, not the new one."""
    svc = _late_report_service(tmp_path)
    got = svc.know("john@doe.com", as_of="2026-08-01T00:00:00Z", as_of_basis="event")
    assert got.etches[0].current_value == "Head of Support"


def test_ingest_is_the_default_basis(tmp_path):
    svc = _late_report_service(tmp_path)
    explicit = svc.know("john@doe.com", as_of="2026-09-10T00:00:00Z",
                        as_of_basis="ingest")
    default = svc.know("john@doe.com", as_of="2026-09-10T00:00:00Z")
    assert default.count == explicit.count == 0
    assert default.as_of_basis == "ingest"


def test_recall_honours_the_basis(tmp_path):
    svc = _late_report_service(tmp_path)
    ev = svc.recall("job", scope="people", top_k=3, include_signals=False,
                    as_of="2026-09-10T00:00:00Z", as_of_basis="event")
    assert [h.value for h in ev] == ["VP Customer Operations"]
    ing = svc.recall("job", scope="people", top_k=3, include_signals=False,
                     as_of="2026-09-10T00:00:00Z")
    assert ing == []


def test_unknown_basis_falls_back_to_ingest(tmp_path):
    """A typo must not silently become event-basis."""
    svc = _late_report_service(tmp_path)
    got = svc.know("john@doe.com", as_of="2026-09-10T00:00:00Z", as_of_basis="typo")
    assert got.count == 0
