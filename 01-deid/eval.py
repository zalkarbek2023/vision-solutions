#!/usr/bin/env python3
"""Run a YOLO26-Text detector over the test set and measure how much of the burned-in text gets erased.

A word counts as erased when at least 98 % of its glyph pixels are under the redaction mask.
Mask variants: the filled detection polygons as they are ("poly"), grown by 8 px ("poly+8"), and bounding
rectangles grown by a share of the word height sideways and vertically ("rect 0.5/0.15", "rect 1.0/0.25").
Tissue erased by mistake is the share of the picture covered by the mask farther than 12 px from any
ground-truth word (for the rect variants: farther than the growth itself would reach).

Usage: eval.py <set_dir> <model.pt|model.engine> [device=mps] [imgsz=1280]
Writes <set_dir>/../eval_<model>_<imgsz>.npz (per-word coverage for every threshold) and prints a summary.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

from eraser import rect_mask

CONFS = [0.05, 0.1, 0.25, 0.5]
PADS = ["poly", "poly+8", "rect 0.5/0.15", "rect 1.0/0.25"]
ERASED = 0.98

set_dir, weights = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "mps"
imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
model = YOLO(weights, task="obb")
meta = json.load(open(f"{set_dir}/meta.json"))

rows, cover, tissue_lost, tissue_all, ms = [], [], [], [], []  # rows: (sid, frame k, word i) -> coverage[conf, pad]
t_all = time.time()
for m in meta:
    for k, f in enumerate(m["frames"]):
        frame = cv2.imread(f"{set_dir}/frames/{f['name']}.jpg")
        gt = cv2.imread(f"{set_dir}/gt/{f['name']}.png")
        alpha, tissue = gt[..., 0], gt[..., 1] > 0
        t = time.perf_counter()
        r = model.predict(frame, imgsz=imgsz, conf=min(CONFS), max_det=1000, device=device, verbose=False)[0]
        polys, conf = r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy()
        ms.append((time.perf_counter() - t) * 1000)
        cov = np.zeros((len(f["words"]), len(CONFS), len(PADS)), np.float32)
        lost = np.zeros((len(CONFS), len(PADS)), np.float32)
        total = np.zeros((len(CONFS), len(PADS)), np.float32)  # all tissue under the mask, text over tissue included
        for ci, c in enumerate(CONFS):
            keep = polys[conf >= c]
            base = np.zeros(alpha.shape, np.uint8)
            for p in keep:  # one by one: a single fillPoly call XORs overlapping polygons into holes
                cv2.fillPoly(base, [p.round().astype(np.int32)], 1)
            masks = [base, cv2.dilate(base, np.ones((17, 17), np.uint8)), rect_mask(keep, alpha.shape, 0.5, 0.15),
                     rect_mask(keep, alpha.shape, 1.0, 0.25)]
            for pi, (mask, (gx, gy)) in enumerate(zip(masks, [(0, 0), (8, 8), (0.5, 0.15), (1.0, 0.25)])):
                near = np.zeros(alpha.shape, np.uint8)  # ground-truth words plus what the growth may cover around them
                for w in f["words"]:
                    x0, y0, x1, y1 = w["box"]
                    h = y1 - y0
                    mx, my = (12 + gx, 12 + gy) if gx >= 8 or gx == 0 else (12 + int(gx * h * 1.3), 12 + int(gy * h * 1.3))
                    near[max(0, y0 - my):y1 + my, max(0, x0 - mx):x1 + mx] = 1
                lost[ci, pi] = (mask.astype(bool) & tissue & (near == 0)).sum() / max(1, tissue.sum())
                total[ci, pi] = (mask.astype(bool) & tissue).sum() / max(1, tissue.sum())
                for wi, w in enumerate(f["words"]):
                    x0, y0, x1, y1 = w["box"]
                    g = alpha[y0:y1, x0:x1] > 64
                    cov[wi, ci, pi] = (mask[y0:y1, x0:x1].astype(bool) & g).sum() / max(1, g.sum())
        for wi, w in enumerate(f["words"]):
            rows.append((m["sid"], k, wi, w["kind"], m["lang"], m["layout"], m["size"], m["font"], m["color"]))
        cover.append(cov)
        tissue_lost.append(lost)
        tissue_all.append(total)
cover, tissue_lost, tissue_all = np.concatenate(cover), np.stack(tissue_lost), np.stack(tissue_all)
rows = np.array(rows, dtype=object)
tag = f"{os.path.splitext(os.path.basename(weights))[0]}_{imgsz}"
np.savez_compressed(f"{set_dir}/../eval_{tag}.npz", rows=rows, cover=cover, tissue_lost=tissue_lost, tissue_all=tissue_all, ms=np.array(ms),
                    confs=CONFS, pads=PADS)

erased = cover >= ERASED
print(f"\n{tag}: {len(ms)} frames, {len(rows)} words, {np.median(ms[5:]):.1f} ms/frame median on {device}, "
      f"total {time.time() - t_all:.0f} s")
print("words erased, % (rows: conf, columns: " + " | ".join(PADS) + ")")
for ci, c in enumerate(CONFS):
    print(f"  conf {c:<4}: " + "  ".join(f"{erased[:, ci, pi].mean() * 100:6.2f}" for pi in range(len(PADS))))
print("tissue erased by mistake, % of the picture")
for ci, c in enumerate(CONFS):
    print(f"  conf {c:<4}: " + "  ".join(f"{tissue_lost[:, ci, pi].mean() * 100:6.3f}" for pi in range(len(PADS))))
print("picture under the mask in total, %")
for ci, c in enumerate(CONFS):
    print(f"  conf {c:<4}: " + "  ".join(f"{tissue_all[:, ci, pi].mean() * 100:6.3f}" for pi in range(len(PADS))))
