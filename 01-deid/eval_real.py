#!/usr/bin/env python3
"""Check the eraser on REAL burned-in text: the processor overlay left in a REAL-Colon clip (date, time, settings).

There is no hand labelling. The ground truth is built from the clip itself: the endoscope picture is one big bright
disk on a black frame, so anything bright outside that disk is overlay text. Regions = connected groups of such
pixels over the whole clip (date, time, "*1/100", ...). For every frame and region: the share of its text pixels
under the mask; a region counts as erased in a frame when that share is at least 98 %.

Runs the full pipeline (decode -> detector -> mask -> black out -> encode) and times each stage.

Usage: eval_real.py <clip.mp4> <model.pt|model.engine> [device=0] [out_dir=out/real] [memory=30]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from eraser import Eraser

clip, weights = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "0"
out_dir = sys.argv[4] if len(sys.argv) > 4 else "out/real"
memory = int(sys.argv[5]) if len(sys.argv) > 5 else 30
os.makedirs(out_dir, exist_ok=True)
name = os.path.splitext(os.path.basename(clip))[0]
TEXT, ERASED = 70, 0.98

# 1. The disk: temporal mean of 60 frames, the largest bright component, grown a little.
cap = cv2.VideoCapture(clip)
n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
acc = None
for i in np.linspace(0, n - 1, 60).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    acc = g if acc is None else acc + g
mean = acc / 60
k, lab, stats, _ = cv2.connectedComponentsWithStats((mean > 12).astype(np.uint8))
disk = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
disk = cv2.morphologyEx(disk, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
disk = cv2.dilate(disk, np.ones((21, 21), np.uint8))
outside = disk == 0


def gt_of(frame):
    """Text pixels: bright glyphs on the black frame; on a solid bright plate (a mode label) the dark glyphs inside it."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = ((g > TEXT) & outside).astype(np.uint8)
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(bright)
    for c in range(1, nb):
        x, y, w, h, a = sb[c]
        if a > 300 and a > 0.6 * w * h:  # a filled plate, not a glyph
            bright[lb == c] = 0
            bright[y + 2:y + h - 2, x + 2:x + w - 2] |= (g[y + 2:y + h - 2, x + 2:x + w - 2] < 100).astype(np.uint8)
    return bright.astype(bool)


# 2. Regions: union of text pixels over the sampled frames, grown so the letters of one label join.
union = np.zeros(outside.shape, np.uint8)
for i in np.linspace(0, n - 1, 60).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    union |= gt_of(f).astype(np.uint8)
nreg, regions, rstats, _ = cv2.connectedComponentsWithStats(cv2.dilate(union, np.ones((9, 25), np.uint8)))
keep = [r for r in range(1, nreg) if (union[regions == r]).sum() >= 15]
boxes = {r: [int(v) for v in rstats[r, :4]] for r in keep}
print(f"{name}: {n} frames, {fps:.2f} fps, {outside.shape[1]}x{outside.shape[0]}, disk {disk.mean() * 100:.1f} % of the picture")
print("text regions (x, y, w, h): " + json.dumps(list(boxes.values())))

# 3. The pipeline over every frame, twice: without memory and with it.
res = {}
for mem in (0, memory):
    er = Eraser(weights, device=device, memory=mem)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    wr = cv2.VideoWriter(f"{out_dir}/{name}_erased_m{mem}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (outside.shape[1], outside.shape[0]))
    cov, t_dec, t_era, t_enc, tissue = [], [], [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        ok, f = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        out, mask = er(f)
        t2 = time.perf_counter()
        wr.write(out)
        t3 = time.perf_counter()
        if i >= 5:  # first frames warm the engine up
            t_dec.append(t1 - t0); t_era.append(t2 - t1); t_enc.append(t3 - t2)
        g = gt_of(f)
        row = []
        for r in keep:
            px = g & (regions == r)
            row.append(np.nan if px.sum() < 15 else (mask[px] > 0).mean())
        cov.append(row)
        tissue.append((mask[disk > 0] > 0).mean())
        if mem and i in (0, n // 3, 2 * n // 3):
            cv2.imwrite(f"{out_dir}/{name}_{i:05d}_before.jpg", f)
            cv2.imwrite(f"{out_dir}/{name}_{i:05d}_after.jpg", out)
    wr.release()
    cov = np.array(cov, dtype=float)
    present = ~np.isnan(cov)
    erased = np.where(present, cov >= ERASED, True)
    ms = lambda a: round(1000 * float(np.median(a)), 2)  # noqa: E731
    res[mem] = {"frames": len(cov), "region_frames": int(present.sum()), "region_frames_erased": int((erased & present).sum()),
                "frames_with_a_leak": int((~erased.all(1)).sum()), "per_region_erased_%": [round(100 * float(erased[present[:, j], j].mean()), 2) for j in range(len(keep))],
                "text_pixels_erased_%": round(100 * float(np.nanmean(cov)), 2), "tissue_in_mask_%": round(100 * float(np.mean(tissue)), 3),
                "ms_decode": ms(t_dec), "ms_erase": ms(t_era), "ms_encode": ms(t_enc), "fps_end_to_end": round(1 / float(np.median(np.array(t_dec) + np.array(t_era) + np.array(t_enc))), 1)}
    print(f"memory {mem}: " + json.dumps(res[mem]))
json.dump({"clip": name, "model": weights, "regions": list(boxes.values()), "results": res}, open(f"{out_dir}/{name}_{os.path.basename(weights)}.json", "w"), indent=1)
np.save(f"{out_dir}/{name}_disk.npy", disk)
