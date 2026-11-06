"""OCR with Tesseract (word boxes + confidence via image_to_data), with a low-quality check and a fallback pass.

Primary pass:  grayscale -> light denoise -> Otsu threshold -> Tesseract (psm 6)
Fallback pass: 2x upscale -> stronger denoise -> adaptive threshold -> deskew -> Tesseract (psm 4)
The better of the two (by mean word confidence) is kept. If it is still below the thresholds,
the document is flagged for manual review.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

from .config import get_settings

DPI = 300


@dataclass
class OcrPage:
    text: str
    mean_conf: float
    words: int


@dataclass
class OcrResult:
    text: str
    mean_conf: float
    chars: int
    pages: int
    path: str  # "text-layer" | "primary" | "fallback"
    low_quality: bool
    notes: list[str] = field(default_factory=list)


def _to_images(data: bytes, kind: str) -> list[np.ndarray]:
    if kind == "pdf":
        imgs = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                imgs.append(arr)
        return imgs
    img = Image.open(io.BytesIO(data))
    img = img.convert("L")
    return [np.array(img)]


def _pdf_text_layer(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(p.get_text("text") for p in doc).strip()


def _prep_primary(gray: np.ndarray) -> np.ndarray:
    g = cv2.fastNlMeansDenoising(gray, h=10)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _deskew(img: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 50:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 10:
        return img
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _prep_fallback(gray: np.ndarray) -> np.ndarray:
    g = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g = cv2.fastNlMeansDenoising(g, h=20)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return _deskew(th)


def _tesseract(img: np.ndarray, psm: int) -> OcrPage:
    cfg = f"--oem 1 --psm {psm}"
    d = pytesseract.image_to_data(img, lang="eng", config=cfg, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, w in zip(d["conf"], d["text"]) if w.strip() and float(c) >= 0]
    # Rebuild text line by line from the word boxes. A wide horizontal gap becomes "    " so columns
    # (e.g. "Patient Name: X        Age/Sex: 42/M") stay separate for the extractors.
    lines: dict[tuple, list[int]] = {}
    for i, w in enumerate(d["text"]):
        if w.strip():
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
    out_lines = []
    for _, idx in sorted(lines.items()):
        idx.sort(key=lambda i: d["left"][i])
        parts = [d["text"][idx[0]]]
        for prev, cur in zip(idx, idx[1:]):
            gap = d["left"][cur] - (d["left"][prev] + d["width"][prev])
            h = max(d["height"][prev], d["height"][cur], 1)
            parts.append(("    " if gap > 1.6 * h else " ") + d["text"][cur])
        out_lines.append("".join(parts))
    text = "\n".join(out_lines)
    return OcrPage(text=text, mean_conf=float(np.mean(confs)) if confs else 0.0, words=len(confs))


def _run(images: list[np.ndarray], prep, psm: int) -> tuple[str, float]:
    pages = [_tesseract(prep(im), psm) for im in images]
    total_words = sum(p.words for p in pages) or 1
    conf = sum(p.mean_conf * p.words for p in pages) / total_words
    return "\n\n".join(p.text for p in pages).strip(), conf


def run_ocr(data: bytes, kind: str) -> OcrResult:
    s = get_settings()
    notes: list[str] = []

    # A born-digital PDF already has exact text; use it and skip OCR.
    if kind == "pdf":
        layer = _pdf_text_layer(data)
        if len(layer) >= s.min_ocr_chars:
            with fitz.open(stream=data, filetype="pdf") as doc:
                n = doc.page_count
            return OcrResult(layer, 100.0, len(layer), n, "text-layer", False, ["PDF text layer used (no OCR needed)"])

    images = _to_images(data, kind)
    text, conf = _run(images, _prep_primary, 6)
    path = "primary"
    low = len(text) < s.min_ocr_chars or conf < s.min_ocr_confidence
    if low:
        notes.append(f"Primary pass low quality ({len(text)} chars, confidence {conf:.0f}); trying fallback preprocessing")
        t2, c2 = _run(images, _prep_fallback, 4)
        if c2 > conf or len(t2) > len(text) * 1.2:
            text, conf, path = t2, c2, "fallback"
        low = len(text) < s.min_ocr_chars or conf < s.min_ocr_confidence
        if low:
            notes.append("Still low quality after fallback - routed to manual review")
    return OcrResult(text, round(conf, 1), len(text), len(images), path, low, notes)
