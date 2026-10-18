"""
Offline tests for the declarative extension registry (app/ext.py).
No network, no LLM. Run:  pytest -q   (from etchmem-server/)
"""
from __future__ import annotations

import textwrap

from app.ext import ExtRegistry, load_extensions


def _write(dirpath, name, body):
    fp = dirpath / name
    fp.write_text(textwrap.dedent(body), encoding="utf-8")
    return fp


def test_missing_dir_is_noop():
    reg = load_extensions("/no/such/dir")
    assert reg.specs == []
    assert reg.prompt_block() == ""


def test_load_parses_specs(tmp_path):
    _write(tmp_path, "sales.yaml", """
        domain: sales
        properties:
          - name: sales_intent
            description: Buying intent.
            values: [none, low, medium, high]
            entity_types: [company, person]
          - name: decision_maker
            description: Who owns the call.
            entity_types: [company]
    """)
    reg = load_extensions(str(tmp_path))
    names = {s.name for s in reg.specs}
    assert names == {"sales_intent", "decision_maker"}
    si = reg.by_name["sales_intent"]
    assert si.values == ("none", "low", "medium", "high")
    assert si.entity_types == ("company", "person")
    assert si.domain == "sales"


def test_prompt_block_lists_properties(tmp_path):
    _write(tmp_path, "sales.yaml", """
        domain: sales
        properties:
          - name: sales_intent
            description: Buying intent.
            values: [low, high]
    """)
    block = load_extensions(str(tmp_path)).prompt_block()
    assert "sales_intent" in block
    assert "Allowed values: low, high." in block


def test_accept_enforces_enum_and_entity_type():
    from app.ext import PropSpec
    reg = ExtRegistry([
        PropSpec(name="sales_intent", values=("low", "high"),
                 entity_types=("company",)),
    ])
    # unknown property → always allowed (core stays open-vocabulary)
    assert reg.accept(property="contract_status", value="signed")
    # declared enum: case-insensitive match passes, junk fails
    assert reg.accept(property="sales_intent", value="High", entity_type="company")
    assert not reg.accept(property="sales_intent", value="very high", entity_type="company")
    # entity_type filter
    assert not reg.accept(property="sales_intent", value="high", entity_type="person")


def test_last_file_wins_on_duplicate(tmp_path):
    _write(tmp_path, "a.yaml", """
        domain: a
        properties:
          - name: churn_risk
            values: [low]
    """)
    _write(tmp_path, "b.yaml", """
        domain: b
        properties:
          - name: churn_risk
            values: [low, medium, high]
    """)
    reg = load_extensions(str(tmp_path))
    assert reg.by_name["churn_risk"].values == ("low", "medium", "high")
