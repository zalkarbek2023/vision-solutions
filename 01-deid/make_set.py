#!/usr/bin/env python3
"""Build the test set: real colonoscopy frames (REAL-Colon clips) with synthetic patient overlays and exact ground truth.

135 fictitious 'procedures' = 5 clips x 3 layouts x 3 languages x 3 text sizes; font, colour, blur and JPEG
quality are random per procedure. N frames per procedure, evenly spaced, so the clock and the tissue change.

Usage: make_set.py <clips_dir> <out_dir> [frames_per_session=8]
Writes <out_dir>/frames/*.jpg, <out_dir>/gt/*.png (channel 0: glyph alpha, channel 1: tissue mask) and meta.json.
"""
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

import overlay as o

clips_dir, out = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(f"{out}/frames", exist_ok=True)
os.makedirs(f"{out}/gt", exist_ok=True)
rng = random.Random(20261114)
clips = sorted(glob.glob(f"{clips_dir}/*.mp4"))
meta, sid = [], 0
for clip in clips:
    cap = cv2.VideoCapture(clip)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    for layout in o.LAYOUTS:
        for lang in o.LANGS:
            for size in o.SIZES:
                s = o.make_session(rng, lang, layout, size)
                frames = []
                for k in range(per):
                    idx = int((k + rng.random()) * (n - 1) / per)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, im = cap.read()
                    if not ok:
                        continue
                    frame, _, words, alpha, tissue = o.render(s, im, idx / fps)
                    name = f"{sid:03d}_{k}"
                    cv2.imwrite(f"{out}/frames/{name}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    gt = np.dstack([alpha, tissue.astype(np.uint8) * 255, np.zeros_like(alpha)])
                    cv2.imwrite(f"{out}/gt/{name}.png", gt)
                    frames.append({"name": name, "clip_frame": idx, "words": words})
                meta.append({"sid": sid, "clip": os.path.basename(clip), **{k: (os.path.basename(v) if k == "font" else v)
                             for k, v in s.items()}, "frames": frames})
                sid += 1
    print(os.path.basename(clip), "sessions so far:", sid, flush=True)
json.dump(meta, open(f"{out}/meta.json", "w"), ensure_ascii=False)
print("frames:", sum(len(m["frames"]) for m in meta), "words:", sum(len(f["words"]) for m in meta for f in m["frames"]))
