"""Probe: can plain Tesseract read the specimen docs well enough? (decides OCR strategy)"""
from pathlib import Path
import pytesseract
from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

print("=== Probing data/kyc_documents (Synthetic Smoke Set) ===")
for p in sorted((DATA_DIR / "kyc_documents").glob("*.png")):
    txt = pytesseract.image_to_string(Image.open(p))
    first_lines = [line.strip() for line in txt.splitlines() if line.strip()][:3]
    print(f"[{p.name}] -> {first_lines}")

if (DATA_DIR / "kyc_official").exists():
    print("\n=== Probing data/kyc_official (Official Hackathon Specimen Set) ===")
    for kind in ["gst_certificate", "udyam_certificate", "signatory_id", "address_proof"]:
        matches = sorted((DATA_DIR / "kyc_official").glob(f"K01_{kind}.png"))
        if matches:
            p = matches[0]
            txt = pytesseract.image_to_string(Image.open(p))
            first_lines = [line.strip() for line in txt.splitlines() if line.strip()][:3]
            print(f"[{p.name}] -> {first_lines}")
