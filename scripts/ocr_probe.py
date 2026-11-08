"""Probe: can plain Tesseract read the specimen docs well enough? (decides OCR strategy)"""
import glob, os, pytesseract, re
from PIL import Image
for kind in ["gst_certificate", "udyam_certificate", "signatory_id", "address_proof"]:
    f = sorted(glob.glob(f"data/kyc_documents/K0[1-3]_{kind}.png"))[1]  # a sample per type
    txt = pytesseract.image_to_string(Image.open(f))
    print("=====", os.path.basename(f)); print("\n".join(l for l in txt.splitlines() if l.strip())[:700])
