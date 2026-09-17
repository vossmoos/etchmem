"""
Declarative claim/etch extensions.

Drop YAML files into the ext dir (default ./ext) to teach the claim extractor
your domain vocabulary — WITHOUT touching core types or the DuckDB schema.

Two declaration blocks, both additive:

`properties:` — extra *properties* to look for. Each declared property still
becomes a plain (entity, property, value) claim, and therefore a plain etch
(i.e. `entity.<property>`). Declared enums and entity_types are enforced on the
way back; undeclared properties still pass, so the core stays open-vocabulary.

`entities:` — which *subjects* matter, and how strictly to identify them. This
is how a corpus whose entities are catalogue identifiers (article numbers, SKUs,
part numbers, case ids) stops relying on embedding similarity for identity:

    domain: catalogue-support
    properties:
      - name: lifecycle_status
        description: Whether the article can still be ordered.
        values: [active, superseded, discontinued, recalled]
        entity_types: [product, part]
    entities:
      - type: product
        description: A module identified by its article number.
        match: exact                       # never fuzzy-merge two products
        identifier_pattern: '[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?'
      - type: ticket
        ignore: true                       # ticket numbers are not knowledge

`identifier_pattern` is the important one. When it matches inside the surface
name, the normalized match becomes the entity's canonical key, so
"INNOXEL Modul MSM-0808", "MSM 0808" and "msm-0808" resolve to ONE entity by
string equality — and `MSM-0809` can never be merged into it, which embedding
similarity at any threshold cannot guarantee.

A property may also declare `distributive: true`, meaning a fact the model
reads as being about several subjects may be written to each of them. It is a
ceiling on what the model is allowed to spread, never a way of detecting it:
deciding whether a sentence names a list or a relation is language
understanding, and it belongs to the extractor that read the sentence.

`kind: attribute` is reserved for non-belief annotations (a sidecar that should
NOT go through conflict resolution). It is parsed and surfaced in the registry
but, until a generic `attributes` column exists, is not yet persisted — only
`kind: property` (the default) flows end-to-end today.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import yaml

MATCH_EXACT = "exact"
MATCH_FUZZY = "fuzzy"

_IDENT_PUNCT = re.compile(r"[\s_]+")

@dataclass(frozen=True)
class PropSpec:
    name: str
    description: str = ""
    kind: str = "property"                       # "property" (→ etch) | "attribute"
    values: tuple[str, ...] | None = None        # allowed enum, lowercased
    entity_types: tuple[str, ...] | None = None  # restrict to these subject types
    # True when a claim naming SEVERAL subjects is genuinely true of each, so it
    # may be split into one claim per subject. Opt-in and declared, never
    # inferred: `known_fault` distributes, `lifecycle_status` must not — the
    # source said one of them is discontinued, not both.
    distributive: bool = False
    # When set, this property's VALUE is another entity of that type, not a
    # literal. The value is resolved through the same rules as a subject, so
    # "NT 2405" and "NT-2405" stop being two values of one belief, and the edge
    # can be followed in both directions.
    relation: str | None = None
    domain: str = ""


@dataclass(frozen=True)
class EntitySpec:
    """How to recognise and identify one kind of subject.

    match               "fuzzy" (default, embedding fallback) or "exact"
                        (normalized-name equality only — never merge by
                        similarity).
    identifier_pattern  Regex for the stable identifier carried inside the
                        surface name (an article number, SKU, part number).
                        When it matches, the normalized match becomes the
                        canonical key AND resolution is exact by construction.
    sim_threshold       Per-type override of settings.entity_sim_threshold,
                        used only when this type still resolves fuzzily.
    ignore              Drop claims about this subject type entirely. For
                        identifiers that are addressing, not knowledge —
                        ticket numbers, message ids, order references.
    """
    type: str
    description: str = ""
    match: str = MATCH_FUZZY
    identifier_pattern: str | None = None
    sim_threshold: float | None = None
    ignore: bool = False
    domain: str = ""

    @property
    def regex(self) -> re.Pattern[str] | None:
        if not self.identifier_pattern:
            return None
        try:
            return re.compile(self.identifier_pattern)
        except re.error:
            return None


def normalize_identifier(raw: str) -> str:
    """Canonical form of an extracted identifier: lowercase, separators unified.

    'MSM 0808' / 'msm_0808' / 'MSM-0808'  ->  'msm-0808'
    Digits and letters are never dropped, so MSM-0808 and MSM-0809 stay apart.
    """
    return _IDENT_PUNCT.sub("-", raw.strip()).replace("--", "-").lower()


@dataclass
class ExtRegistry:
    specs: list[PropSpec] = field(default_factory=list)
    entities: list[EntitySpec] = field(default_factory=list)

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def by_name(self) -> dict[str, PropSpec]:
        return {s.name: s for s in self.specs}

    def prompt_block(self) -> str:
        """System-prompt fragment appended to the extractor's instructions."""
        return self._property_block() + self._entity_block()

    def _property_block(self) -> str:
        props = [s for s in self.specs if s.kind == "property"]
        if not props:
            return ""
        lines = []
        for s in props:
            line = f"- {s.name}: {s.description}".rstrip()
            if s.values:
                line += f" Allowed values: {', '.join(s.values)}."
            if s.entity_types:
                line += f" (only for entity_type: {', '.join(s.entity_types)})"
            if s.relation:
                line += (f" The VALUE is another {s.relation}: give its"
                         " identifier alone, no surrounding words.")
            lines.append(line)
        return (
            "\n\nExtended properties — when the signal supports them, extract "
            "claims using these EXACT property names and, where listed, only the "
            "allowed values:\n" + "\n".join(lines)
        )

    def accept(self, *, property: str, value: str, entity_type: str | None = None) -> bool:
        """Is this (property, value) admissible?

        Unknown properties always pass — the core stays open-vocabulary. A
        *declared* property enforces its enum and entity_types so extraction
        stays consistent (no sales_intent='very high' drifting in).
        """
        spec = self.by_name.get(property)
        if spec is None:
            return True
        if spec.entity_types and entity_type and entity_type not in spec.entity_types:
            return False
        if spec.values and value.strip().lower() not in spec.values:
            return False
        return True

    # ── entities ───────────────────────────────────────────────────────────

    @property
    def by_entity_type(self) -> dict[str, EntitySpec]:
        return {e.type: e for e in self.entities}

    def _entity_block(self) -> str:
        if not self.entities:
            return ""
        kept = [e for e in self.entities if not e.ignore]
        dropped = [e for e in self.entities if e.ignore]
        lines = []
        for e in kept:
            line = f"- {e.type}: {e.description}".rstrip()
            if e.identifier_pattern:
                line += (" Name this subject by its identifier exactly as it "
                         "appears in the text, without surrounding words, and "
                         "with EXACTLY ONE identifier per claim.")
            lines.append(line)
        block = ""
        if lines:
            block += (
                "\n\nSubject types — prefer these EXACT entity_type values when "
                "the signal is about one of them:\n" + "\n".join(lines)
                + "\n\nONE SUBJECT PER CLAIM. When a sentence states something "
                "about several subjects, emit one claim per subject, each named "
                "on its own. Never put two identifiers in one entity_name: "
                "'MSM-0808 und MSM-0810' must become two claims, and "
                "'NT-2405 for MSM-0808' names ONE subject with the other as "
                "context — decide which one the fact is about."
            )
        if dropped:
            block += (
                "\n\nNever create claims whose subject is one of these; they "
                "identify a record, not a thing worth remembering: "
                + ", ".join(e.type for e in dropped) + "."
            )
        return block

    def entity_spec(self, entity_type: str | None) -> EntitySpec | None:
        if not entity_type:
            return None
        return self.by_entity_type.get(entity_type)

    def is_ignored(self, entity_type: str | None) -> bool:
        spec = self.entity_spec(entity_type)
        return bool(spec and spec.ignore)

    def canonical_key(self, entity_type: str | None, name: str) -> str | None:
        """The stable identifier inside `name`, normalized — or None.

        A hit means identity is decided by string equality and the caller must
        NOT fall back to embedding similarity.
        """
        return self.canonical_key_detail(entity_type, name)[0]

    def canonical_key_detail(
        self, entity_type: str | None, name: str
    ) -> tuple[str | None, str]:
        """`(key, reason)` — the key, plus why there isn't one.

        reason: ok | not_declared | no_pattern | no_match | ambiguous

        AMBIGUOUS is the interesting one. A surface name carrying two distinct
        identifiers ("NT-2405 for MSM-0808") does not say which one the claim
        is about, and taking the first match silently attaches the fact to the
        wrong subject. There is no safe guess here, so there is no key: the
        caller counts it and the fact stays unattached rather than wrong.
        """
        spec = self.entity_spec(entity_type)
        if spec is None:
            return None, "not_declared"
        rx = spec.regex
        if rx is None:
            return None, "no_pattern"
        found = rx.findall(name)
        # findall returns tuples when the pattern has groups; re-run for spans.
        matches = [m.group(0) for m in rx.finditer(name)]
        if not matches:
            return None, "no_match"
        keys = {normalize_identifier(m) for m in matches}
        if len(keys) > 1:
            return None, "ambiguous"
        return keys.pop(), "ok"

    def identifier_candidates(self, entity_type: str | None, name: str) -> list[str]:
        """Every distinct identifier found in `name` — for diagnostics."""
        spec = self.entity_spec(entity_type)
        rx = spec.regex if spec else None
        if rx is None:
            return []
        seen, out = set(), []
        for m in rx.finditer(name):
            k = normalize_identifier(m.group(0))
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def subject_count(self, entity_type: str | None, name: str) -> int:
        """How many distinct identifiers `entity_name` carries.

        A COUNT, not a language judgement: it behaves identically in German,
        French, Italian and Romansh with nothing to configure. More than one
        means the extractor produced a malformed claim — which subject the fact
        belongs to is a reading of the original sentence, and that sentence is
        not available here. The model that read it decides; see
        `Pipeline._subjects_for`.
        """
        return len(self.identifier_candidates(entity_type, name))

    def relation_type(self, property: str) -> str | None:
        """The entity type this property's value points at, or None (a literal)."""
        spec = self.by_name.get(property)
        return spec.relation if spec else None

    def distributes(self, property: str) -> bool:
        """May a fact about several subjects be written to each of them?

        A declared ceiling, not a detector. Even when the model reads a
        sentence as naming several subjects, `lifecycle_status` must not be
        spread across them: "MSM-0808 or MSM-0810 is discontinued" says one of
        them is. Same principle as a declared enum — the model proposes, the
        declaration constrains.
        """
        spec = self.by_name.get(property)
        return bool(spec and spec.distributive)

    def is_exact_match(self, entity_type: str | None) -> bool:
        spec = self.entity_spec(entity_type)
        return bool(spec and spec.match == MATCH_EXACT)

    def sim_threshold(self, entity_type: str | None, default: float) -> float:
        spec = self.entity_spec(entity_type)
        if spec and spec.sim_threshold is not None:
            return spec.sim_threshold
        return default


def _coerce_spec(raw: dict, domain: str) -> PropSpec:
    vals = raw.get("values")
    ents = raw.get("entity_types")
    return PropSpec(
        name=str(raw["name"]).strip(),
        description=str(raw.get("description", "")).strip(),
        kind=str(raw.get("kind", "property")).strip().lower(),
        distributive=bool(raw.get("distributive", False)),
        relation=(str(raw["relation"]).strip() if raw.get("relation") else None),
        values=tuple(str(v).strip().lower() for v in vals) if vals else None,
        entity_types=tuple(str(v).strip() for v in ents) if ents else None,
        domain=domain,
    )


def _coerce_entity(raw: dict, domain: str) -> EntitySpec:
    thresh = raw.get("sim_threshold")
    match = str(raw.get("match", MATCH_FUZZY)).strip().lower()
    if match not in (MATCH_EXACT, MATCH_FUZZY):
        match = MATCH_FUZZY
    pattern = raw.get("identifier_pattern")
    return EntitySpec(
        type=str(raw["type"]).strip(),
        description=str(raw.get("description", "")).strip(),
        match=match,
        identifier_pattern=str(pattern) if pattern else None,
        sim_threshold=float(thresh) if thresh is not None else None,
        ignore=bool(raw.get("ignore", False)),
        domain=domain,
    )


def load_extensions(path: str | None = None) -> ExtRegistry:
    """Read every *.yaml / *.yml in `path` into a registry. Last file wins on
    duplicate property / entity-type names. Missing folder → empty registry."""
    if path is None:
        from app.config import settings
        path = settings.ext_dir
    if not path or not os.path.isdir(path):
        return ExtRegistry([], [])

    props: dict[str, PropSpec] = {}
    ents: dict[str, EntitySpec] = {}
    for fp in sorted(glob.glob(os.path.join(path, "*.yml"))
                     + glob.glob(os.path.join(path, "*.yaml"))):
        with open(fp, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        domain = str(doc.get("domain") or os.path.splitext(os.path.basename(fp))[0])
        for raw in doc.get("properties") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            spec = _coerce_spec(raw, domain)
            props[spec.name] = spec
        for raw in doc.get("entities") or []:
            if not isinstance(raw, dict) or not raw.get("type"):
                continue
            espec = _coerce_entity(raw, domain)
            ents[espec.type] = espec
    return ExtRegistry(list(props.values()), list(ents.values()))
