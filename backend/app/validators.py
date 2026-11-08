"""Identifier validation, OCR repair and masking.

Design note: OCR output is *untrusted input*. We never accept a GSTIN because OCR
said so; we accept it because it passes structure + checksum + registry lookup.
"""
from __future__ import annotations
import itertools
import re

_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Structure: 2 state digits, 10-char PAN, entity no., 'Z', checksum. README: PAN 4th char is 'X' in this dataset.
_GSTIN_RE = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]")
_UDYAM_RE = re.compile(r"UDYAM-[A-Z]{2}-\d{2}-\d{7}")

# Characters Tesseract commonly confuses on watermarked / monospace text.
_CONFUSABLE = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S",
               "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G"}


def _checksum_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        p = _CHARSET.index(ch) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return _CHARSET[(36 - total % 36) % 36]


def is_valid_gstin(gstin: str | None) -> bool:
    """Format + checksum only. Says nothing about whether the GSTIN exists or is Active."""
    if not gstin or not _GSTIN_RE.fullmatch(gstin):
        return False
    return _checksum_char(gstin[:14]) == gstin[14]


def is_valid_udyam(udyam: str | None) -> bool:
    return bool(udyam) and bool(_UDYAM_RE.fullmatch(udyam))


def repair_gstin(raw: str, max_edits: int = 2) -> list[str]:
    """Return checksum-valid GSTINs reachable from `raw` by swapping up to `max_edits`
    OCR-confusable characters. Layer 1 of KYC repair; only fixes simple confusions.
    Callers must still verify the result against the registry (layer 2)."""
    raw = re.sub(r"[^0-9A-Z]", "", (raw or "").upper())
    if len(raw) != 15:
        return []
    positions = [i for i, c in enumerate(raw) if c in _CONFUSABLE]
    found: list[str] = []
    for n in range(0, max_edits + 1):
        for combo in itertools.combinations(positions, n):
            chars = list(raw)
            for i in combo:
                chars[i] = _CONFUSABLE[chars[i]]
            cand = "".join(chars)
            if is_valid_gstin(cand) and cand not in found:
                found.append(cand)
        if found:           # prefer the fewest edits
            return found
    return found


def mask_id(value: str | None, keep_start: int = 4, keep_end: int = 3) -> str:
    """Mask identifiers for logs and UI (privacy: data minimisation in observability)."""
    if not value:
        return ""
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return value[:keep_start] + "*" * (len(value) - keep_start - keep_end) + value[-keep_end:]


# --- Structure-aware repair (ADR-013) -------------------------------------------------------------
# A GSTIN has a fixed shape: 2 digits, 5 letters, 4 digits, 1 letter, 1 entity char, constant 'Z', 1 checksum char.
# Tesseract's lookalike errors (Z read as 2 or 7, I as 1, O as 0 ...) are fixed by forcing each position into
# its class, then confirmed by the checksum. Callers must still confirm against the registry.
_DIGIT_POS = {0, 1, 7, 8, 9, 10}
_LETTER_POS = {2, 3, 4, 5, 6, 11}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


def _force_structure(s: str) -> str:
    out = []
    for i, c in enumerate(s):
        if i in _DIGIT_POS:
            c = _TO_DIGIT.get(c, c)
        elif i in _LETTER_POS:
            c = _TO_LETTER.get(c, c)
        elif i == 13:
            c = "Z"
        out.append(c)
    return "".join(out)


def repair_gstin_structured(raw: str) -> list[str]:
    """Checksum-valid GSTINs reachable from noisy OCR text. Handles one inserted or lost character."""
    s = re.sub(r"[^0-9A-Z]", "", (raw or "").upper())
    variants: set[str] = set()
    if len(s) == 15:
        variants.add(s)
    elif len(s) == 16:
        variants |= {s[:i] + s[i + 1:] for i in range(16)}
    elif len(s) == 14:
        variants |= {s[:i] + "Z" + s[i:] for i in range(15)}
    found: list[str] = []
    for v in variants:
        n = _force_structure(v)
        derived = n[:14] + _checksum_char(n[:14]) if re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z", n[:14]) else None
        for cand in ([n] if is_valid_gstin(n) else repair_gstin(n, max_edits=2)) + ([derived] if derived else []):
            if cand not in found:
                found.append(cand)
    return sorted(found)
