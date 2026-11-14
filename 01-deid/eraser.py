#!/usr/bin/env python3
"""Erase burned-in text from video frames with a YOLO26-Text detector (PyTorch weights or a TensorRT engine).

Three ideas, each measured in eval.py and eval_video.py:
1. A low confidence threshold: a missed word is a privacy leak, a false box only hides a few pixels.
2. Each detected word becomes a rectangle grown by its own height sideways (kx) and a quarter of it up and down (ky):
   the detector tends to leave out punctuation next to a word ("*1/100", "Dr.", "рожд.:").
3. Memory: burned-in text does not move, so a pixel that was text in a third of the last frames stays erased
   even when the detector misses it in the current one.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def rect_mask(polys: np.ndarray, shape: tuple[int, int], kx: float = 1.0, ky: float = 0.25) -> np.ndarray:
    """Mask (uint8 0/1) of the polygons' bounding rectangles, each grown by kx and ky of its own height."""
    mask = np.zeros(shape, np.uint8)
    for p in polys:
        x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
        h = y1 - y0
        a, b = max(0, int(x0 - kx * h)), max(0, int(y0 - ky * h))
        mask[b:int(np.ceil(y1 + ky * h)), a:int(np.ceil(x1 + kx * h))] = 1
    return mask


class Eraser:
    """Detector plus memory. Call it on a BGR frame, get the frame with the text blacked out and the mask."""

    def __init__(self, weights: str = "yolo26s-text.pt", device: str | int | None = None, imgsz: int = 1280,
                 conf: float = 0.05, kx: float = 1.0, ky: float = 0.25, memory: int = 30, stable: float = 0.33):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="obb")
        self.args = {"imgsz": imgsz, "conf": conf, "max_det": 1000, "device": device, "verbose": False}
        self.kx, self.ky, self.stable = kx, ky, stable
        self.recent: deque[np.ndarray] = deque(maxlen=memory)
        self.count: np.ndarray | None = None  # per pixel: in how many remembered frames it was text

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Word polygons (N, 4, 2) in frame pixels."""
        return self.model.predict(frame, **self.args)[0].obb.xyxyxyxy.cpu().numpy()

    def remember(self, now: np.ndarray) -> np.ndarray:
        """Add this frame's mask to the memory and return the mask to erase: now, or stable text of recent frames."""
        if self.recent.maxlen == 0:
            return now
        if self.count is None:
            self.count = np.zeros(now.shape, np.uint16)
        if len(self.recent) == self.recent.maxlen:
            self.count -= self.recent[0]
        self.recent.append(now)
        self.count += now
        return now | (self.count >= max(2, self.stable * len(self.recent))).astype(np.uint8)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self.remember(rect_mask(self.detect(frame), frame.shape[:2], self.kx, self.ky))
        out = frame.copy()
        out[mask > 0] = 0
        return out, mask
