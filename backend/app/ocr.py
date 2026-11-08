"""OCR extraction and document parsing service for Smart KYC Pipeline.

Performs deterministic image preprocessing, Tesseract OCR extraction,
document classification, and lenient field extraction.

Architecture:
- Layer 1: Image preprocessing (grayscale + adaptive thresholding for watermarks)
- Layer 2: Tesseract OCR execution with confidence scoring
- Layer 3: Rule-based document type detection & lenient regex field extraction
- Deterministic verification: Extracted fields are strictly verified downstream
  by checksum validation and mock registry lookups.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]

from .validators import mask_id

logger = logging.getLogger(__name__)

# Strict GSTIN: 2 state digits, 5 uppercase alpha, 4 digits, 1 uppercase alpha, 1 entity char, 'Z', 1 checksum char
_STRICT_GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")

# Lenient GSTIN regex to catch common OCR misreads (e.g. 'O' for '0', '2' for 'Z', etc.)
# Follows pattern: [0-9O]{2}[A-Z0-9]{5}[0-9O]{4}[A-Z0-9][1-9A-Z][A-Z0-9][0-9A-Z]
_LENIENT_GSTIN_RE = re.compile(
    r"\b([0-9O]{2}[A-Z0-9]{5}[0-9O]{4}[A-Z0-9][1-9A-Z][A-Z0-9][0-9A-Z])\b"
)

# Labeled GSTIN search pattern - robust to OCR braces, colons confused as 'S' or ';', etc.
_LABELED_GSTIN_RE = re.compile(
    r"(?:GSTIN|GST|Registration\s+Number(?:\s*[\(\{<\[]?\s*GSTIN\s*[\)\}>\]]?)?)\s*[:\-\./;=S\s]*([0-9O]{2}[A-Z0-9]{5}[0-9O]{4}[A-Z0-9][1-9A-Z][A-Z0-9][0-9A-Z])",
    re.IGNORECASE,
)

# Strict and lenient Udyam registration patterns
_STRICT_UDYAM_RE = re.compile(r"\b(UDYAM-[A-Z]{2}-\d{2}-\d{7})\b", re.IGNORECASE)
_LENIENT_UDYAM_RE = re.compile(r"\b(UDYAM-[A-Z]{2}-[0-9O]{2}-[0-9O]{7})\b", re.IGNORECASE)


@dataclass
class ExtractedDocument:
    doc_type: str  # 'gstin_cert' | 'udyam_cert' | 'signatory_id' | 'address_proof' | 'unknown'
    raw_text: str
    extracted_gstin: str | None = None
    extracted_udyam: str | None = None
    extracted_name: str | None = None
    extracted_address: str | None = None
    confidence: float = 0.0  # 0.0-100.0, Tesseract mean confidence


def _configure_tesseract() -> None:
    """Locate and configure tesseract executable if not in PATH on Windows."""
    if pytesseract is None:
        return
    if shutil.which("tesseract"):
        return
    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    for candidate in common_paths:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            logger.info("Configured Tesseract binary path: %s", candidate)
            return


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for OCR extraction:
    - Convert to grayscale and apply dynamic range autocontrast (handles faint/low-contrast ink)
    - Apply adaptive thresholding to eliminate background watermarks on GST certificates
    - Return preprocessed PIL Image
    """
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=2)

    # Method 1: OpenCV adaptive thresholding if available
    try:
        import cv2
        import numpy as np

        img_np = np.array(gray)
        thresh = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        return Image.fromarray(thresh)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("OpenCV adaptive thresholding failed: %s", e)

    # Method 2: NumPy + PIL BoxBlur adaptive thresholding
    try:
        import numpy as np

        blurred = gray.filter(ImageFilter.BoxBlur(radius=15))
        arr = np.array(gray, dtype=np.int16)
        blur_arr = np.array(blurred, dtype=np.int16)
        # Background is lighter than text; subtract offset to preserve dark text while removing watermarks
        thresh = np.where(arr > (blur_arr - 10), 255, 0).astype(np.uint8)
        return Image.fromarray(thresh)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("NumPy adaptive thresholding failed: %s", e)

    # Method 3: Pure Pillow fallback
    return gray


def extract_text(image_path: Path) -> tuple[str, float]:
    """Load image from path, preprocess, and run Tesseract OCR.

    Returns:
        tuple of (raw_text, confidence_score) where confidence is 0.0 - 100.0.
        Gracefully handles missing files or uninstalled Tesseract by logging a warning and returning ("", 0.0).
    """
    path = Path(image_path)
    if not path.is_file():
        logger.warning("Image file does not exist: %s", path)
        return "", 0.0

    if pytesseract is None:
        logger.warning("pytesseract is not available on this system; returning empty OCR text.")
        return "", 0.0

    _configure_tesseract()

    try:
        with Image.open(path) as img:
            preprocessed = preprocess_image(img)

            raw_text = pytesseract.image_to_string(preprocessed)

            # Compute mean confidence across recognized words
            data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
            confs: list[float] = []
            for c, t in zip(data.get("conf", []), data.get("text", [])):
                try:
                    val = float(c)
                    if val >= 0 and str(t).strip():
                        confs.append(val)
                except (ValueError, TypeError):
                    continue

            confidence = float(sum(confs) / len(confs)) if confs else 0.0
            return raw_text, round(confidence, 2)

    except (FileNotFoundError, OSError) as e:
        logger.warning("Tesseract binary not found or failed to execute: %s", e)
        return "", 0.0
    except Exception as e:
        logger.warning("Unexpected error during OCR extraction for %s: %s", path, e)
        return "", 0.0


def detect_doc_type(raw_text: str) -> str:
    """Detect document type from OCR extracted text.

    Classification priority:
    1. 'gstin_cert': Contains GST keywords or candidate GSTIN pattern
    2. 'udyam_cert': Contains Udyam/MSME keywords
    3. 'signatory_id': Contains Aadhaar, PAN, or Voter ID markers
    4. 'address_proof': Contains Electricity, Utility, Rent, or Lease markers
    5. 'unknown': Otherwise
    """
    if not raw_text or not raw_text.strip():
        return "unknown"

    text = raw_text.upper()

    # 1. Udyam Certificate (check prior to general 'Registration Certificate')
    if any(k in text for k in ["UDYAM", "UDYOG AADHAAR", "UDYOG ADHAAR", "MINISTRY OF MICRO", "MSME"]) or "UDYAM-" in text:
        return "udyam_cert"

    # 2. GSTIN Certificate (handles noisy titles e.g. Form GST, Registration Certificate, or presence of GSTIN)
    if any(k in text for k in ["GSTIN", "GOODS AND SERVICES TAX", "FORM GST", "G.S.T"]) or ("REGISTRATION CERTIFICATE" in text and "GST" in text) or re.search(r"\b\d{2}[A-Z0-9]{10,13}\b", text):
        return "gstin_cert"

    # 3. Signatory ID (Aadhaar, PAN, Voter ID)
    if (
        "AADHAAR" in text
        or "AADHAR" in text
        or "VOTER" in text
        or re.search(r"\bPAN\b|PERMANENT ACCOUNT NUMBER", text)
    ):
        return "signatory_id"

    # 4. Address Proof (Electricity, Utility, Rent, Lease)
    if any(k in text for k in ["ELECTRICITY", "UTILITY", "RENT", "LEASE"]):
        return "address_proof"

    return "unknown"


def _clean_name(cand: str | None) -> str | None:
    """Clean and validate extracted entity / individual name."""
    if not cand:
        return None
    # Cut off if another header keyword appears on the same line
    cand = re.split(
        r"\b(?:GSTIN|Trade\s+Name|Legal\s+Name|Date|PAN|Address|State|Status|Registration|Type|Constitution)\b",
        cand,
        flags=re.IGNORECASE,
    )[0]
    cand = cand.strip(" \t\r\n:;,-|/")
    if len(cand) <= 2 or not re.search(r"[A-Za-z]", cand):
        return None
    return cand


def _extract_name(raw_text: str) -> str | None:
    """Extract entity or owner name using prioritized pattern matching."""
    patterns = [
        r"(?:Legal\s+Name(?:\s+of\s+Business)?)\s*[:\-\–]?\s*([^\n\r]+)",
        r"(?:Enterprise\s+Name|Name\s+of\s+Enterprise)\s*[:\-\–]?\s*([^\n\r]+)",
        r"(?:Trade\s+Name(?:\s*,\s*if\s+any)?)\s*[:\-\–]?\s*([^\n\r]+)",
        r"(?:Name\s+of\s+(?:Business|Taxpayer|Signatory|Owner|Applicant))\s*[:\-\–]?\s*([^\n\r]+)",
        r"\bName\s*[:\-\–]\s*([^\n\r]+)",
    ]
    for pat in patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            cand = _clean_name(m.group(1))
            if cand:
                return cand

    # Check newline patterns where header and value are on consecutive lines
    newline_patterns = [
        r"(?:Legal\s+Name(?:\s+of\s+Business)?)\s*[:\-\–]?\s*\n\s*([^\n\r]+)",
        r"(?:Enterprise\s+Name|Name\s+of\s+Enterprise)\s*[:\-\–]?\s*\n\s*([^\n\r]+)",
        r"(?:Trade\s+Name(?:\s*,\s*if\s+any)?)\s*[:\-\–]?\s*\n\s*([^\n\r]+)",
        r"(?:Name\s+of\s+(?:Business|Taxpayer|Signatory|Owner|Applicant))\s*[:\-\–]?\s*\n\s*([^\n\r]+)",
        r"\bName\s*[:\-\–]?\s*\n\s*([^\n\r]+)",
    ]
    for pat in newline_patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            cand = _clean_name(m.group(1))
            if cand:
                return cand

    # For Udyam certificates, name often appears on the line immediately following the UDYAM registration number
    udyam_match = re.search(r"UDYAM-[A-Z]{2}-[0-9O]{2}-[0-9O]{7}", raw_text, re.IGNORECASE)
    if udyam_match:
        lines = [line.strip() for line in raw_text[udyam_match.end():].splitlines() if line.strip()]
        for next_line in lines[:3]:
            if not re.search(r"\b(Micro|Small|Medium|Services|Trading|Manufacturing|Date|Male|Female|General|OBC|SC|ST)\b", next_line, re.IGNORECASE):
                cand = _clean_name(next_line)
                if cand and len(cand) >= 4:
                    return cand

    return None


def _extract_address(raw_text: str) -> str | None:
    """Extract postal address including multi-line continuations."""
    patterns = [
        r"(?:Principal\s+Place\s+of\s+Business|Principal\s+Place)\s*[:\-\–]?",
        r"(?:Registered\s+Office(?:\s+Address)?)\s*[:\-\–]?",
        r"\bAddress\s*(?:of\s+Enterprise|of\s+Business)?\s*[:\-\–]?",
    ]
    lines = raw_text.splitlines()
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for i, line in enumerate(lines):
            match = regex.search(line)
            if match:
                same_line = line[match.end():].strip(" \t:;,-|")
                addr_parts: list[str] = []
                if same_line:
                    # Truncate if subsequent header appears on same line
                    part = re.split(
                        r"\b(?:State|Date|GSTIN|UDYAM|PAN|Status)\b", same_line, flags=re.IGNORECASE
                    )[0].strip(" \t:;,-|")
                    if part:
                        addr_parts.append(part)

                # Collect subsequent continuation lines (up to 3)
                for next_line in lines[i + 1 : i + 4]:
                    cleaned = next_line.strip()
                    if not cleaned:
                        break
                    if re.match(
                        r"^(?:State|Date|GSTIN|UDYAM|PAN|Trade\s+Name|Legal\s+Name|Status|Constitution|Type|Nature)\b",
                        cleaned,
                        re.IGNORECASE,
                    ):
                        break
                    addr_parts.append(cleaned)

                if addr_parts:
                    full_addr = ", ".join(addr_parts)
                    full_addr = re.sub(r",\s*,+", ",", full_addr)
                    full_addr = re.sub(r"\s+", " ", full_addr).strip(" \t:;,-|")
                    if len(full_addr) >= 5 and re.search(r"[A-Za-z]", full_addr):
                        return full_addr
    return None


def parse_fields(raw_text: str) -> dict[str, str | None]:
    """Parse key identity fields from OCR raw text.

    Extracts:
    - GSTIN: via strict regex, labeled match, and lenient confusable regex
    - Udyam number: UDYAM-[A-Z]{2}-\\d{2}-\\d{7}
    - Name: Legal Name, Trade Name, Enterprise Name, or generic Name
    - Address: Address, Principal Place, Registered Office
    """
    if not raw_text or not raw_text.strip():
        return {
            "extracted_gstin": None,
            "extracted_udyam": None,
            "extracted_name": None,
            "extracted_address": None,
            "gstin": None,
            "udyam": None,
            "name": None,
            "address": None,
        }

    # 1. GSTIN Extraction
    gstin: str | None = None
    # Check labeled GSTIN first
    labeled_match = _LABELED_GSTIN_RE.search(raw_text)
    if labeled_match:
        gstin = labeled_match.group(1).upper()
    else:
        # Check strict regex anywhere in text
        strict_match = _STRICT_GSTIN_RE.search(raw_text.upper())
        if strict_match:
            gstin = strict_match.group(1).upper()
        else:
            # Fall back to lenient confusable pattern
            lenient_match = _LENIENT_GSTIN_RE.search(raw_text.upper())
            if lenient_match:
                gstin = lenient_match.group(1).upper()

    # If still not found, check individual tokens for embedded GSTINs or prefix noise (e.g. S32TTNXB...)
    if not gstin:
        for token in raw_text.split():
            clean_tok = re.sub(r"^[^0-9A-Z]+", "", token.upper())
            clean_tok = re.sub(r"[^0-9A-Z]+$", "", clean_tok)
            if len(clean_tok) == 15 and _LENIENT_GSTIN_RE.fullmatch(clean_tok):
                gstin = clean_tok
                break
            elif len(clean_tok) == 16 and clean_tok[0] in "S:;-." and _LENIENT_GSTIN_RE.fullmatch(clean_tok[1:]):
                gstin = clean_tok[1:]
                break

    # 2. Udyam Extraction
    udyam: str | None = None
    udyam_match = _STRICT_UDYAM_RE.search(raw_text)
    if udyam_match:
        udyam = udyam_match.group(1).upper()
    else:
        lenient_udyam = _LENIENT_UDYAM_RE.search(raw_text)
        if lenient_udyam:
            raw_u = lenient_udyam.group(1).upper()
            parts = raw_u.split("-")
            udyam = f"{parts[0]}-{parts[1]}-{parts[2].replace('O', '0')}-{parts[3].replace('O', '0')}"

    # 3. Name Extraction
    name = _extract_name(raw_text)

    # 4. Address Extraction
    address = _extract_address(raw_text)

    return {
        "extracted_gstin": gstin,
        "extracted_udyam": udyam,
        "extracted_name": name,
        "extracted_address": address,
        # Convenience aliases
        "gstin": gstin,
        "udyam": udyam,
        "name": name,
        "address": address,
    }


def extract_document(image_path: Path) -> ExtractedDocument:
    """Main OCR pipeline entry point.

    1. Loads and preprocesses image
    2. Runs Tesseract OCR with confidence calculation
    3. Detects document type
    4. Parses structured fields
    5. Returns ExtractedDocument data model
    """
    path = Path(image_path)
    raw_text, confidence = extract_text(path)
    doc_type = detect_doc_type(raw_text)
    fields = parse_fields(raw_text)

    doc = ExtractedDocument(
        doc_type=doc_type,
        raw_text=raw_text,
        extracted_gstin=fields.get("extracted_gstin"),
        extracted_udyam=fields.get("extracted_udyam"),
        extracted_name=fields.get("extracted_name"),
        extracted_address=fields.get("extracted_address"),
        confidence=confidence,
    )

    logger.info(
        "Extracted document '%s': type=%s, confidence=%.1f, gstin=%s, udyam=%s",
        path.name,
        doc.doc_type,
        doc.confidence,
        mask_id(doc.extracted_gstin),
        mask_id(doc.extracted_udyam),
    )
    return doc
