#!/usr/bin/env python3
"""Inventory of top-level 'cropping_info*' keys across the FantasyID tree.
Run this BEFORE re-running build_manifest, to confirm which crop-metadata key
variants exist and whether any JSON carries more than one.

  python crop_keys_inventory.py --root "C:\\Users\\senor\\09DISS\\FANTASY"
"""
import argparse, json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
root = Path(ap.parse_args().root)

counts, multiple, none_examples = Counter(), [], []
for p in root.rglob("*.json"):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        counts["<parse_error>"] += 1; continue
    keys = [k for k in data if isinstance(k, str) and k.startswith("cropping_info")]
    if not keys:
        counts["<none>"] += 1
        if len(none_examples) < 5: none_examples.append(str(p))
    for k in keys:
        counts[k] += 1
    if len(keys) > 1:
        multiple.append((str(p), keys))

print("Cropping-info key counts (by JSON occurrence):")
for k, n in counts.most_common():
    print(f"  {k}: {n}")
print(f"\nJSONs with >1 crop block: {len(multiple)}")
for p, ks in multiple[:10]:
    print("  ", p, ks)
if none_examples:
    print("\nExamples with NO cropping_info* key:")
    for p in none_examples: print("  ", p)
