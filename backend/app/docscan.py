"""Layout-aware reading of the four KYC document types (GST certificate, Udyam certificate,
signatory ID, address proof).

Why this exists (ADR-013): the first OCR layer was tuned on self-generated images. On the official
specimens every document landed in manual review, because the diagonal watermark sits on top of the
GSTIN and Udyam values. Here the watermark is removed by colour (ink is neutral, watermark is
red-tinted), image sharpness is measured before trusting any text, and fields are read by their
printed labels instead of being guessed from the raw text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

from .validators import repair_gstin_structured

# Calibrated on the 47 official specimens: the one blurred scan scores 26, every other document >= 1000.
MIN_SHARPNESS = 300.0
_ISO = r"(\d{4}-\d{2}-\d{2})"


@dataclass
class ScannedDoc:
    doc_type: str                      # gst_certificate | udyam_certificate | signatory_id | address_proof | unknown
    fields: dict[str, str | None] = field(default_factory=dict)
    sharpness: float = 0.0
    raw_text: str = ""

    @property
    def readable(self) -> bool:
        return self.sharpness >= MIN_SHARPNESS


def _remove_watermark(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = (r + g + b) // 3
    reddish = (r - (g + b) // 2) > 18
    return np.where(reddish & (lum > 100), 255, lum).astype(np.uint8)


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low values mean a blurred scan."""
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total, sum_all = gray.size, float((np.arange(256) * hist).sum())
    w_b = sum_b = 0.0
    best, thr = -1.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        between = w_b * w_f * (sum_b / w_b - (sum_all - sum_b) / w_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


def _skew_angle(gray: np.ndarray) -> float:
    """Rotation (degrees) that makes text rows horizontal: the angle whose row profile is most 'peaky'."""
    small = Image.fromarray(gray).resize((500, max(1, int(gray.shape[0] * 500 / gray.shape[1]))))
    ink = Image.fromarray(((np.asarray(small) < 128) * 255).astype(np.uint8))

    def score(angle: float) -> float:
        rot = ink.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        return float(np.asarray(rot, dtype=np.float64).sum(axis=1).var())

    coarse = max(np.arange(-15, 15.1, 1.0), key=score)
    return float(max(np.arange(coarse - 1, coarse + 1.01, 0.25), key=score))


def _binarise(path: Path) -> tuple[Image.Image, float]:
    gray = _remove_watermark(Image.open(path))
    sharp = _sharpness(gray)                      # measured before any rotation blur
    angle = _skew_angle(gray)
    if abs(angle) > 0.5:
        gray = np.asarray(Image.fromarray(gray).rotate(angle, resample=Image.BICUBIC, fillcolor=255))
    return Image.fromarray(((gray > _otsu(gray)) * 255).astype(np.uint8)), sharp


def read_image(path: Path) -> tuple[str, float]:
    """OCR a document after watermark removal. Returns (text, sharpness)."""
    bw, sharp = _binarise(path)
    return pytesseract.image_to_string(bw, config="--psm 6"), sharp


def read_value_column(bw: Image.Image, label: str) -> str | None:
    """Re-read one value (printed in a monospace font) from the right-hand column of the row labelled
    `label`, upscaled and restricted to the characters an identifier can contain."""
    data = pytesseract.image_to_data(bw, config="--psm 6", output_type=pytesseract.Output.DICT)
    for i, word in enumerate(data["text"]):
        if word.strip().upper().startswith(label) and data["left"][i] < 0.3 * bw.width:
            top, h = data["top"][i], data["height"][i]
            crop = bw.crop((int(0.40 * bw.width), max(0, top - 8), bw.width, top + h + 8))
            cfg = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            return pytesseract.image_to_string(crop, config=cfg).strip() or None
    return None


def classify(text: str) -> str:
    t = text.lower()
    if "gst regist" in t or ("gstin" in t and "legal name" in t):
        return "gst_certificate"
    if "udyam registration" in t:
        return "udyam_certificate"
    if "signatory id" in t or "authorised signatory" in t:
        return "signatory_id"
    if "bill" in t or "address proof" in t:
        return "address_proof"
    return "unknown"


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _after(lines: list[str], label: str, allow_next: bool = False) -> str | None:
    """Value printed after `label` on the same line (or, with allow_next, on the line below it)."""
    pat = re.compile(rf"^{label}\s*[:\-]?\s*(.*)$", re.I)
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m and m.group(1).strip():
            return m.group(1).strip()
        if m and allow_next and i + 1 < len(lines):
            return lines[i + 1]
    return None


def _wrapped(lines: list[str], label: str, stop: str) -> str | None:
    """Value that wraps onto following lines until a `stop` label. The printed layout breaks long
    text at a fixed width, even mid-word, so a full-width first line is joined without a space."""
    out: list[str] = []
    started = False
    for ln in lines:
        if not started:
            m = re.match(rf"^{label}\s*(.*)$", ln, re.I)
            if m:
                started = True
                out.append(m.group(1).strip())
            continue
        if re.match(rf"^{stop}", ln, re.I):
            break
        out.append(re.sub(r"^place of business\s*", "", ln, flags=re.I).strip())
    parts = [p for p in out if p]
    if not parts:
        return None
    joined = parts[0]
    for p in parts[1:]:
        joined += p if len(joined) >= 44 else " " + p
    return joined


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def parse(doc_type: str, text: str) -> dict[str, str | None]:
    L = _lines(text)
    if doc_type == "gst_certificate":
        return {
            "gstin": _find(r"GSTIN\s*[:\-]?\s*([0-9A-Z]{15})", text),
            "legal_name": _after(L, "Legal name"),
            "trade_name": _after(L, "Trade name"),
            "constitution": _after(L, "Constitution"),
            "address": _wrapped(L, "Address of principal", "Date of liability"),
            "date_of_liability": _find(r"Date of liability\s*" + _ISO, text),
        }
    if doc_type == "udyam_certificate":
        return {
            "udyam_number": _find(r"(UDYAM-[A-Z]{2}-\d{2}-\d{7})", text),  # absent on the official specimens
            "enterprise_name": _after(L, "Enterprise name"),
            "state": _after(L, "State"),
            "city": _after(L, "City"),
            "date_of_registration": _find(r"Date of registration\s*" + _ISO, text),
        }
    if doc_type == "signatory_id":
        biz = None
        for i, ln in enumerate(L):
            if re.match(r"^For business", ln, re.I):
                inline = re.sub(r"^For business\s*", "", ln, flags=re.I).strip()
                biz = inline or (L[i + 1] if i + 1 < len(L) else None)
        return {
            "name": _after(L, "Name", allow_next=True),
            "designation": _after(L, "Designation", allow_next=True),
            "business": biz,
            "id_no": _find(r"(SYN-ID-\d+)", text),
            "valid_till": _find(r"Valid till\s*" + _ISO, text),
        }
    if doc_type == "address_proof":
        return {
            "billed_to": _after(L, "Billed to"),
            "address": _wrapped(L, r"Supply,? address", "Bill date"),
            "bill_date": _find(r"Bill date\s*" + _ISO, text),
        }
    return {}


def scan(path: Path) -> ScannedDoc:
    bw, sharp = _binarise(path)
    text = pytesseract.image_to_string(bw, config="--psm 6")
    dt = classify(text)
    fields = parse(dt, text)
    if dt == "gst_certificate" and sharp >= MIN_SHARPNESS:
        raw = read_value_column(bw, "GSTIN")
        fields["gstin_raw"] = raw
        fixed = repair_gstin_structured(raw or "")
        fields["gstin_candidates"] = ",".join(fixed) or None
        fields["gstin"] = fixed[0] if len(fixed) == 1 else fields.get("gstin")
    return ScannedDoc(doc_type=dt, fields=fields, sharpness=sharp, raw_text=text)
