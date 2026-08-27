#!/usr/bin/env python3
"""Report environment + data layout on the lab machine."""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

print("=" * 60)
print(f"HOST: {os.uname().nodename}   CWD: {ROOT}")
print("=" * 60)

print("\n--- GPU ---")
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10)
    print(out.stdout.strip() or "no output")
except Exception as e:
    print(f"unavailable: {e}")

print("\n--- Python ---")
print(f"{sys.version.split()[0]}  ({sys.executable})")
try:
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
except ImportError:
    print("torch not installed in this env")

print("\n--- Repo tree (depth 2, dirs only) ---")
for p in sorted(ROOT.rglob("*")):
    if not p.is_dir() or ".git" in p.parts:
        continue
    rel = p.relative_to(ROOT)
    if len(rel.parts) <= 2:
        n = sum(1 for f in p.iterdir() if f.is_file())
        print(f"  {'  ' * (len(rel.parts) - 1)}{rel.name}/  ({n} files)")

print("\n--- Data directories (size + count) ---")
for name in ["data", "datasets", "manifests", "splits", "results", "checkpoints"]:
    d = ROOT / name
    if not d.exists():
        print(f"  {name}/  ABSENT")
        continue
    files = [f for f in d.rglob("*") if f.is_file()]
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  {name}/  {len(files)} files, {mb:.1f} MB")

print("\nDone.")
