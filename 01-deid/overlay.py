#!/usr/bin/env python3
"""Synthetic endoscopy-tower text overlays with exact ground truth.

Every patient and physician here is fictitious: names are random pairs from the short lists below.
A frame is 1920x1080: the colonoscopy picture plus burned-in text, the way a video processor writes it.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
_SYS, _SUP = "/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/"
# macOS system fonts; all eight cover the Kyrgyz letters ө ү ң (checked with fontTools, see check_fonts)
FONTS = [_SYS + "Menlo.ttc", _SYS + "Monaco.ttf", _SYS + "Helvetica.ttc", _SUP + "PTMono.ttc", _SUP + "Courier New Bold.ttf",
         _SUP + "Arial.ttf", _SUP + "Tahoma.ttf", _SUP + "Verdana.ttf"]
# em size in px at 1080p; the real overlay left in REAL-Colon 002-008 is about 30
SIZES = {"small": 22, "typical": 29, "large": 37}
COLORS = {"white": (238, 238, 238), "grey": (188, 188, 188), "yellow": (236, 200, 70), "green": (120, 230, 130)}
LAYOUTS = ["column", "corners", "on_image"]
LANGS = ["en", "ru", "ky"]

NAMES = {
    "en": {
        "sur": ["BAKER", "HOLLOWAY", "PRESCOTT", "WHITFIELD", "LANGDON", "MERCER", "ASHWORTH", "DELANEY"],
        "m": ["Thomas", "Robert", "Victor", "Samuel", "Edward", "Martin"],
        "f": ["Emily", "Hannah", "Olivia", "Grace", "Laura", "Diana"],
        "mid_m": ["J.", "R.", "A.", "W."], "mid_f": ["M.", "E.", "K.", "L."],
    },
    "ru": {
        "sur": ["Соколов", "Кузнецов", "Морозов", "Лебедев", "Виноградов", "Захаров", "Орлов", "Беляков"],
        "m": ["Андрей", "Сергей", "Виктор", "Николай", "Дмитрий", "Павел"],
        "f": ["Анна", "Мария", "Елена", "Ольга", "Татьяна", "Наталья"],
        "mid_m": ["Петрович", "Иванович", "Алексеевич", "Михайлович"],
        "mid_f": ["Петровна", "Ивановна", "Алексеевна", "Михайловна"],
    },
    "ky": {
        "sur": ["Өмүрбеков", "Төлөгөнов", "Үсөнов", "Жаңыбаев", "Сүйүнбаев", "Көчкөнбаев", "Бөрүбаев", "Түмөнбаев"],
        "m": ["Өмүрбек", "Үсөн", "Төлөгөн", "Эркин", "Бакыт", "Нурлан"],
        "f": ["Гүлзат", "Жаңыл", "Айгүл", "Нургүл", "Мээрим", "Чолпон"],
        "mid_m": ["Үсөн уулу", "Төлөгөн уулу", "Бакыт уулу", "Эркин уулу"],
        "mid_f": ["Үсөн кызы", "Төлөгөн кызы", "Бакыт кызы", "Эркин кызы"],
    },
}
LABELS = {
    "en": {"name": "Name:", "dob": "D.O.B:", "sex": "Sex:", "age": "Age:", "dr": "Dr.", "m": "M", "f": "F"},
    "ru": {"name": "ФИО:", "dob": "Дата рожд.:", "sex": "Пол:", "age": "Возраст:", "dr": "Врач:", "m": "М", "f": "Ж"},
    "ky": {"name": "Аты-жөнү:", "dob": "Туулган күнү:", "sex": "Жынысы:", "age": "Жашы:", "dr": "Дарыгер:", "m": "Э", "f": "А"},
}
DEVICE = [["*1/100"], ["AUTO"], ["Lv+5"], ["BLI"], ["NBI"], ["Zoom", "x1.5"], ["Eh:A5", "Cm:1"], ["WLI"]]


def check_fonts() -> None:
    """Print which candidate fonts cover every character the generator can write."""
    from fontTools.ttLib import TTFont

    need = set("".join(w for d in NAMES.values() for v in d.values() for w in v) + "".join(
        w for d in LABELS.values() for w in d.values()) + "0123456789:/.*+- ")
    for f in FONTS:
        cmap = TTFont(f, fontNumber=0).getBestCmap()
        missing = sorted(c for c in need if ord(c) not in cmap and c != " ")
        print(f"{f.split('/')[-1]:22s} missing: {''.join(missing) or 'none'}")


def person(rng: random.Random, lang: str) -> tuple[list[str], str]:
    """A fictitious full name as words, and the sex letter key ('m' or 'f')."""
    n, sex = NAMES[lang], rng.choice("mf")
    sur = rng.choice(n["sur"])
    if sex == "f" and lang in ("ru", "ky"):
        sur += "а"
    words = [sur + ("," if lang == "en" else ""), rng.choice(n[sex]), *rng.choice(n["mid_" + sex]).split()]
    return words, sex


def make_session(rng: random.Random, lang: str, layout: str, size: str | None = None) -> dict:
    """One 'procedure': a fixed patient, physician, font, colour and layout."""
    size = size or rng.choice(list(SIZES))
    name, sex = person(rng, lang)
    doctor, _ = person(rng, lang)
    year = rng.randint(1941, 2004)
    sep = "/" if lang == "en" else "."
    return {
        "lang": lang, "layout": layout, "size": size, "font": rng.choice(FONTS),
        "color": rng.choice(list(COLORS)), "outline": layout == "on_image" and rng.random() < 0.5,
        "blur": rng.choice([0.0, 0.5, 0.8]), "jpeg": rng.choice([95, 80, 65]),
        "name": name, "sex": sex, "age": 2026 - year,
        "id": f"{rng.randint(2019, 2026)}-{rng.randint(0, 999999):06d}",
        "dob": f"{rng.randint(1, 28):02d}{sep}{rng.randint(1, 12):02d}{sep}{year}",
        "date": f"{rng.randint(1, 28):02d}{sep}{rng.randint(1, 12):02d}{sep}{rng.randint(2021, 2026)}",
        "t0": rng.randint(8 * 3600, 17 * 3600), "doctor": doctor[:2],
        "device": rng.sample(DEVICE, 3),
    }


def _wrap(words: list[tuple[str, str]], font: ImageFont.FreeTypeFont, max_w: float) -> list[list[tuple[str, str]]]:
    """Greedy wrap of (word, kind) pairs to lines no wider than max_w."""
    lines, cur = [], []
    for w in words:
        if cur and font.getlength(" ".join(x for x, _ in cur + [w])) > max_w:
            lines.append(cur)
            cur = []
        cur.append(w)
    return lines + [cur] if cur else lines


def _blocks(s: dict, t: float, font: ImageFont.FreeTypeFont, x0: int, iw: int) -> list[dict]:
    """Text blocks for this layout: anchor, alignment and lines of (word, kind)."""
    L, lh = LABELS[s["lang"]], round(SIZES[s["size"]] * 1.45)
    sec = s["t0"] + int(t)
    clock = f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
    name = [(L["name"], "label"), *[(w, "name") for w in s["name"]]]
    patient = lambda mw: [  # noqa: E731
        [("ID:", "label"), (s["id"], "id")],
        *_wrap(name, font, mw),
        [(L["sex"], "label"), (L[s["sex"]], "sexage"), (L["age"], "label"), (str(s["age"]), "sexage")],
        *_wrap([*[(w, "label") for w in L["dob"].split()], (s["dob"], "dob")], font, mw),
    ]
    when = [[(s["date"], "date")], [(clock, "time")]]
    doctor = lambda mw: _wrap([(L["dr"], "label"), *[(w, "doctor") for w in s["doctor"]]], font, mw)  # noqa: E731
    device = [[(w, "device") for w in line] for line in s["device"]]
    if s["layout"] == "column":  # picture flush right, text in the black column on the left
        mw = x0 - 56
        top = [*patient(mw), [], *when, [], *doctor(mw)]
        return [{"x": 28, "y": 48, "align": "l", "lines": top},
                {"x": 28, "y": H - 48 - lh * len(device), "align": "l", "lines": device}]
    if s["layout"] == "corners":  # picture centred, text in the corners (as in REAL-Colon 002-008)
        d = doctor(520)
        return [{"x": 24, "y": 22, "align": "l", "lines": [*patient(520), *when]},
                {"x": W - 24, "y": 22, "align": "r", "lines": device},
                {"x": 24, "y": H - 22 - lh * len(d), "align": "l", "lines": d}]
    # on_image: text written over the picture itself
    return [{"x": x0 + 64, "y": 66, "align": "l", "lines": patient(640)},
            {"x": x0 + iw - 64, "y": 66, "align": "r", "lines": device},
            {"x": x0 + iw - 64, "y": H - 66 - lh * 2, "align": "r", "lines": when}]


def render(s: dict, clip_bgr: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
    """Compose one frame.

    Returns (frame with text, frame without text, words, glyph alpha, tissue mask). Each word is
    {"text", "kind", "box": [x0, y0, x1, y1]}; its glyph pixels are alpha[box] > 64.
    """
    iw = round(clip_bgr.shape[1] * H / clip_bgr.shape[0])
    img = cv2.resize(clip_bgr, (iw, H), interpolation=cv2.INTER_CUBIC)
    x0 = W - iw if s["layout"] == "column" else (W - iw) // 2
    clean = np.zeros((H, W, 3), np.uint8)
    clean[:, x0:x0 + iw] = img
    tissue = np.zeros((H, W), bool)
    tissue[:, x0:x0 + iw] = img.max(2) > 24
    font = ImageFont.truetype(s["font"], SIZES[s["size"]])
    lh = round(SIZES[s["size"]] * 1.45)
    alpha, stroke = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    d, ds = ImageDraw.Draw(alpha), ImageDraw.Draw(stroke)
    space, words = font.getlength(" "), []
    for b in _blocks(s, t, font, x0, iw):
        for k, line in enumerate(b["lines"]):
            if not line:
                continue
            x = b["x"] - (font.getlength(" ".join(w for w, _ in line)) if b["align"] == "r" else 0)
            y = b["y"] + k * lh
            for text, kind in line:
                d.text((x, y), text, font=font, fill=255)
                if s["outline"]:
                    ds.text((x, y), text, font=font, fill=255, stroke_width=2)
                words.append({"text": text, "kind": kind, "box": [int(v) for v in d.textbbox((x, y), text, font=font)]})
                x += font.getlength(text) + space
    a = np.asarray(alpha, np.float32)[..., None] / 255
    frame = clean.astype(np.float32)
    if s["outline"]:
        frame *= 1 - np.asarray(stroke, np.float32)[..., None] / 255
    frame = frame * (1 - a) + np.array(COLORS[s["color"]][::-1], np.float32) * a
    frame = frame.clip(0, 255).astype(np.uint8)
    if s["blur"]:  # the softening of a capture chain
        frame = cv2.GaussianBlur(frame, (0, 0), s["blur"])
    if s["jpeg"] < 95:  # and of video compression
        frame = cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, s["jpeg"]])[1], 1)
    return frame, clean, words, np.asarray(alpha), tissue


if __name__ == "__main__":
    check_fonts()
