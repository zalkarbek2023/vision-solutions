#!/usr/bin/env python3
"""Break an eval_*.npz down: erased words by kind, language, size, layout, font; frames where a name leaks.

Usage: report.py <eval.npz> [conf=0.05] [variant="rect 1.0/0.25"]
"""
import sys
from collections import defaultdict

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
pad = sys.argv[3] if len(sys.argv) > 3 else "rect 1.0/0.25"
ci, pi = list(z["confs"]).index(conf), list(z["pads"]).index(pad)
rows, cov = z["rows"], z["cover"][:, ci, pi]
erased = cov >= 0.98
cols = ["sid", "k", "wi", "kind", "lang", "layout", "size", "font", "color"]
print(f"conf {conf}, mask {pad}: {erased.mean() * 100:.3f} % of {len(rows)} words erased; "
      f"{(cov == 0).mean() * 100:.3f} % untouched; picture erased by mistake {z['tissue_lost'][:, ci, pi].mean() * 100:.3f} %, "
      f"in total {z['tissue_all'][:, ci, pi].mean() * 100:.3f} %")
for name in ["kind", "lang", "size", "layout", "font", "color"]:
    c = cols.index(name)
    print(f"\nby {name}")
    for v in sorted(set(rows[:, c])):
        m = rows[:, c] == v
        print(f"  {v:22s} {erased[m].mean() * 100:6.2f} %   missed {int((~erased[m]).sum()):4d} of {int(m.sum())}")
frames = defaultdict(lambda: {"name": True, "pii": True, "all": True})
for r, e in zip(rows, erased):
    f = frames[(r[0], r[1])]
    f["all"] &= bool(e)
    if r[3] == "name":
        f["name"] &= bool(e)
    if r[3] in ("name", "id", "dob"):
        f["pii"] &= bool(e)
n = len(frames)
print(f"\nframes: {n}; name fully erased in {sum(f['name'] for f in frames.values()) / n * 100:.2f} %, "
      f"name+ID+birth date in {sum(f['pii'] for f in frames.values()) / n * 100:.2f} %, "
      f"every word in {sum(f['all'] for f in frames.values()) / n * 100:.2f} %")
bad = defaultdict(int)
for r, e in zip(rows, erased):
    if not e:
        bad[r[0]] += 1
print("sessions with misses:", len(bad), "of", len(set(rows[:, 0])), "| worst:", sorted(bad.items(), key=lambda x: -x[1])[:8])
