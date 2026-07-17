"""
Declarative claim/etch extensions.

Drop YAML files into the ext dir (default ./ext) to teach the claim extractor
extra *properties* to look for — WITHOUT touching core types or the DuckDB
schema. Each declared property still becomes a plain (entity, property, value)
claim, and therefore a plain etch (i.e. `entity.<property>`). Extensions are
additive vocabulary: they steer extraction and may constrain values, but they
never override or replace the core triple.

One domain per file, e.g. ext/sales.yaml:

    domain: sales
    properties:
      - name: sales_intent
        description: How strongly the subject signals intent to purchase.
        values: [none, low, medium, high]    # optional enum (case-insensitive)
        entity_types: [company, person]       # optional subject filter
      - name: budget_status
        description: Funding state of the opportunity.
        values: [unknown, unfunded, allocated, approved]

`kind: attribute` is reserved for non-belief annotations (a sidecar that should
NOT go through conflict resolution). It is parsed and surfaced in the registry
but, until a generic `attributes` column exists, is not yet persisted — only
`kind: property` (the default) flows end-to-end today.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class PropSpec:
    name: str
    description: str = ""
    kind: str = "property"                       # "property" (→ etch) | "attribute"
    values: tuple[str, ...] | None = None        # allowed enum, lowercased
    entity_types: tuple[str, ...] | None = None  # restrict to these subject types
    domain: str = ""


@dataclass
class ExtRegistry:
    specs: list[PropSpec] = field(default_factory=list)

    @property
    def by_name(self) -> dict[str, PropSpec]:
        return {s.name: s for s in self.specs}

    def prompt_block(self) -> str:
        """System-prompt fragment appended to the extractor's instructions."""
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


def _coerce_spec(raw: dict, domain: str) -> PropSpec:
    vals = raw.get("values")
    ents = raw.get("entity_types")
    return PropSpec(
        name=str(raw["name"]).strip(),
        description=str(raw.get("description", "")).strip(),
        kind=str(raw.get("kind", "property")).strip().lower(),
        values=tuple(str(v).strip().lower() for v in vals) if vals else None,
        entity_types=tuple(str(v).strip() for v in ents) if ents else None,
        domain=domain,
    )


def load_extensions(path: str | None = None) -> ExtRegistry:
    """Read every *.yaml / *.yml in `path` into a registry. Last file wins on
    duplicate property names. Missing folder → empty (no-op) registry."""
    if path is None:
        from app.config import settings
        path = settings.ext_dir
    if not path or not os.path.isdir(path):
        return ExtRegistry([])

    dedup: dict[str, PropSpec] = {}
    for fp in sorted(glob.glob(os.path.join(path, "*.yml"))
                     + glob.glob(os.path.join(path, "*.yaml"))):
        with open(fp, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        domain = str(doc.get("domain") or os.path.splitext(os.path.basename(fp))[0])
        for raw in doc.get("properties") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            spec = _coerce_spec(raw, domain)
            dedup[spec.name] = spec
    return ExtRegistry(list(dedup.values()))
