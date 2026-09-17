"""
Offline check of an `entities:` declaration against real subject names.

Run this BEFORE loading an archive. Entity resolution is the failure that
hurts most and shows least: two article numbers merging into one subject
cross-contaminates two histories, and every answer that comes out afterwards
looks entirely plausible. This command makes the outcome visible while it is
still cheap to change the pattern.

    python -m app.validate_entities --names names.txt --type product
    python -m app.validate_entities --names names.txt --ext-dir ./ext --json

`names.txt` is one surface name per line — real article numbers and, more
usefully, real entity names as they appear in your records ("INNOXEL Modul
MSM-0808", "das alte Modul", "NT-2405 for MSM-0808").

What it reports, and why each matters:

  OK          a single identifier was found; identity is string equality.
  AMBIGUOUS   two identifiers in one name. There is no safe guess, so no key
              is produced and the fact would stay unattached. Usually a prompt
              problem: the extractor should name ONE subject.
  NO MATCH    the pattern found nothing. With `match: exact` this becomes a
              new entity per spelling variant — fragmentation, not corruption,
              but still wrong.
  COLLISION   two different names resolved to the same key. Expected for real
              variants of one article; a bug if the names are different things.
  NEAR MISS   two keys within one edit of each other. These are the pairs that
              a similarity threshold would have merged — the reason the
              declared-identifier path exists. Informational, not an error.

Exit code is 1 when anything AMBIGUOUS or NO MATCH was found, so it can gate a
load in CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from app.ext import load_extensions


def _levenshtein_le_1(a: str, b: str) -> bool:
    """True when `a` and `b` differ by at most one edit. Cheap, no library."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] != long[j]:
            if skipped:
                return False
            skipped = True
            j += 1
            continue
        i += 1
        j += 1
    return True


def analyse(names: list[str], entity_type: str, ext_dir: str | None) -> dict:
    reg = load_extensions(ext_dir)
    spec = reg.entity_spec(entity_type)
    rows: list[dict] = []
    by_key: dict[str, list[str]] = defaultdict(list)

    for name in names:
        key, reason = reg.canonical_key_detail(entity_type, name)
        rows.append({
            "name": name,
            "key": key,
            "reason": reason,
            "candidates": reg.identifier_candidates(entity_type, name),
        })
        if key:
            by_key[key].append(name)

    collisions = {k: v for k, v in by_key.items() if len(v) > 1}
    keys = sorted(by_key)
    near = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
            if _levenshtein_le_1(a, b)]

    return {
        "entity_type": entity_type,
        "declared": spec is not None,
        "pattern": spec.identifier_pattern if spec else None,
        "match": spec.match if spec else None,
        "total": len(names),
        "ok": sum(1 for r in rows if r["reason"] == "ok"),
        "ambiguous": [r for r in rows if r["reason"] == "ambiguous"],
        "no_match": [r for r in rows if r["reason"] == "no_match"],
        "not_declared": [r for r in rows if r["reason"] == "not_declared"],
        "no_pattern": [r for r in rows if r["reason"] == "no_pattern"],
        "collisions": collisions,
        "near_misses": near,
        "rows": rows,
    }


def _report(res: dict) -> None:
    w = sys.stdout.write
    w(f"\nentity type : {res['entity_type']}\n")
    if not res["declared"]:
        w("  NOT DECLARED in the extension dir — every name of this type will\n"
          "  resolve by embedding similarity. That is the case this check exists\n"
          "  to talk you out of.\n\n")
        return
    w(f"pattern     : {res['pattern']}\nmatch       : {res['match']}\n")
    w(f"names       : {res['total']}   keyed: {res['ok']}\n")

    def block(title: str, rows: list[dict], show_cands: bool = False) -> None:
        if not rows:
            return
        w(f"\n{title} ({len(rows)})\n")
        for r in rows[:40]:
            extra = f"   candidates: {', '.join(r['candidates'])}" if show_cands else ""
            w(f"  - {r['name']}{extra}\n")
        if len(rows) > 40:
            w(f"  ... and {len(rows) - 40} more\n")

    block("AMBIGUOUS — two identifiers, no safe key", res["ambiguous"], True)
    block("NO MATCH — pattern found nothing", res["no_match"])
    block("NO PATTERN — type declared without identifier_pattern", res["no_pattern"])

    if res["collisions"]:
        w(f"\nCOLLISIONS — one key, several names ({len(res['collisions'])})\n")
        for k, names in list(res["collisions"].items())[:20]:
            w(f"  {k}: {', '.join(names)}\n")
        w("  Expected for spelling variants of one article. A bug if any pair\n"
          "  above is genuinely two different things.\n")

    if res["near_misses"]:
        w(f"\nNEAR MISSES — keys one edit apart ({len(res['near_misses'])})\n")
        for a, b in res["near_misses"][:20]:
            w(f"  {a}  vs  {b}\n")
        w("  These stay separate here. A similarity threshold would have been\n"
          "  at risk of merging them.\n")

    bad = len(res["ambiguous"]) + len(res["no_match"])
    w(f"\n{'FAIL' if bad else 'OK'}: {bad} name(s) produced no usable key.\n\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check an entities: declaration against real subject names.")
    ap.add_argument("--names", required=True,
                    help="File with one surface name per line ('-' for stdin).")
    ap.add_argument("--type", default="product", help="Declared entity type to test.")
    ap.add_argument("--ext-dir", default=None, help="Extension dir (default: settings).")
    ap.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = ap.parse_args(argv)

    src = sys.stdin if args.names == "-" else open(args.names, encoding="utf-8")
    with src as fh:
        names = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

    res = analyse(names, args.type, args.ext_dir)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        _report(res)
    return 1 if (res["ambiguous"] or res["no_match"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
