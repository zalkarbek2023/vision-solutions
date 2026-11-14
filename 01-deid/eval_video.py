#!/usr/bin/env python3
"""Measure the eraser on consecutive video frames, with and without memory.

For each of the 135 procedures of the test set (same patient, font, layout as in meta.json) it takes N consecutive
frames from a random place in the clip, renders the overlay, runs the detector and scores two masks per frame:
"now" (this frame's detections only) and "memory" (plus text that was stable over the last second).
Nothing is written to disk except the result.

Usage: eval_video.py <clips_dir> <set_dir> <model> [device=mps] [frames=36] [memory=30]
"""
import json
import random
import sys
import time

import cv2
import numpy as np

import overlay as o
from eraser import Eraser, rect_mask

clips_dir, set_dir, weights = sys.argv[1:4]
device = sys.argv[4] if len(sys.argv) > 4 else "mps"
n_frames = int(sys.argv[5]) if len(sys.argv) > 5 else 36
memory = int(sys.argv[6]) if len(sys.argv) > 6 else 30
fonts = {f.split("/")[-1]: f for f in o.FONTS}
rng = random.Random(1)
rows, cov, lost, t0 = [], [], [], time.time()
for m in json.load(open(f"{set_dir}/meta.json")):
    s = {**m, "font": fonts[m["font"]]}
    cap = cv2.VideoCapture(f"{clips_dir}/{m['clip']}")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    start = rng.randint(0, n - n_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    eraser = Eraser(weights, device=device, memory=memory) if not rows else eraser
    eraser.recent.clear()
    eraser.count = None
    for k in range(n_frames):
        ok, im = cap.read()
        if not ok:
            break
        frame, _, words, alpha, tissue = o.render(s, im, (start + k) / fps)
        now = rect_mask(eraser.detect(frame), frame.shape[:2], eraser.kx, eraser.ky)
        masks = [now, eraser.remember(now)]
        near = np.zeros(alpha.shape, np.uint8)
        for w in words:
            x0, y0, x1, y1 = w["box"]
            h = y1 - y0
            near[max(0, y0 - 12 - int(0.33 * h)):y1 + 12 + int(0.33 * h), max(0, x0 - 12 - int(1.3 * h)):x1 + 12 + int(1.3 * h)] = 1
        lost.append([(mk.astype(bool) & tissue & (near == 0)).sum() / max(1, tissue.sum()) for mk in masks])
        for wi, w in enumerate(words):
            x0, y0, x1, y1 = w["box"]
            g = alpha[y0:y1, x0:x1] > 64
            cov.append([(mk[y0:y1, x0:x1].astype(bool) & g).sum() / max(1, g.sum()) for mk in masks])
            rows.append((m["sid"], k, wi, w["kind"], m["lang"], m["layout"], m["size"]))
    if m["sid"] % 15 == 14:
        print(f"session {m['sid'] + 1}/135, {time.time() - t0:.0f} s", flush=True)
rows, cov, lost = np.array(rows, dtype=object), np.array(cov, np.float32), np.array(lost, np.float32)
np.savez_compressed(f"{set_dir}/../evalvideo_{weights.split('/')[-1].split('.')[0]}.npz", rows=rows, cover=cov, tissue_lost=lost,
                    memory=memory, frames=n_frames)
erased = cov >= 0.98
k = rows[:, 1].astype(int)
frame_id = rows[:, 0].astype(int) * 1000 + k
print(f"\n{len(set(frame_id))} consecutive frames, {len(rows)} words, memory {memory} frames")
for name, sel in [("all frames", np.ones(len(rows), bool)), (f"after warm-up (frame {memory}+)", k >= memory)]:
    if not sel.any():
        continue
    for j, label in enumerate(["now", "memory"]):
        ok = erased[sel, j]
        bad_frames = len(set(frame_id[sel][~ok]))
        names = sel & (rows[:, 3] == "name")
        bad_name = len(set(frame_id[names][~erased[names, j]]))
        print(f"  {name:28s} {label:7s} words erased {ok.mean() * 100:7.3f} %  missed {int((~ok).sum()):4d}  "
              f"frames with any miss {bad_frames}/{len(set(frame_id[sel]))}  frames with a name miss {bad_name}")
print("picture erased by mistake, %: now", round(float(lost[:, 0].mean() * 100), 3), " memory", round(float(lost[:, 1].mean() * 100), 3))
