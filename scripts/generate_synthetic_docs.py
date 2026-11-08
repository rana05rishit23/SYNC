"""Generate synthetic KYC document images for testing and demonstration.

Creates:
1. data/kyc_documents/K01_gstin_active.png (Active GSTIN: Balaji Dental Care LLP)
2. data/kyc_documents/K02_gstin_cancelled.png (Cancelled GSTIN: Vindhya Tours & Travels)
3. data/kyc_documents/K03_udyam_active.png (Udyam Certificate: Balaji Dental Care LLP)
4. data/kyc_documents/K04_gstin_watermarked.png (GSTIN with diagonal watermark)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "kyc_documents"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_font(size: int = 24):
    for font_name in ["arial.ttf", "calibri.ttf", "segoeui.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_gstin_cert(
    filename: str,
    gstin: str,
    legal_name: str,
    trade_name: str,
    address: str,
    watermark: str | None = None,
):
    img = Image.new("RGB", (1000, 700), color="white")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(28)
    header_font = _get_font(22)
    body_font = _get_font(18)

    # Optional diagonal watermark
    if watermark:
        wm_font = _get_font(52)
        # Draw light gray diagonal text
        wm_img = Image.new("RGBA", (1000, 700), (255, 255, 255, 0))
        wm_draw = ImageDraw.Draw(wm_img)
        for y in range(100, 700, 200):
            wm_draw.text((150, y), watermark, fill=(210, 210, 210, 120), font=wm_font)
        img.paste(Image.alpha_composite(img.convert("RGBA"), wm_img).convert("RGB"))
        draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([(20, 20), (980, 680)], outline="black", width=2)
    draw.rectangle([(25, 25), (975, 675)], outline="#555555", width=1)

    # Header
    draw.text((250, 45), "Government of India - Form GST REG-06", fill="black", font=title_font)
    draw.text((360, 85), "Registration Certificate", fill="#333333", font=header_font)
    draw.line([(50, 120), (950, 120)], fill="black", width=2)

    # Fields
    y = 150
    fields = [
        ("Registration Number (GSTIN):", gstin),
        ("Legal Name:", legal_name),
        ("Trade Name:", trade_name),
        ("Constitution of Business:", "Limited Liability Partnership"),
        ("Principal Place of Business:", address),
        ("Date of Liability:", "01/07/2017"),
        ("Date of Validity:", "From 01/07/2017 To Permanent"),
        ("Type of Registration:", "Regular"),
    ]

    for label, val in fields:
        draw.text((60, y), label, fill="#222222", font=body_font)
        draw.text((380, y), val, fill="black", font=body_font)
        y += 45

    # Footer
    draw.line([(50, 580), (950, 580)], fill="#888888", width=1)
    draw.text((60, 600), "Note: This is a system generated certificate issued under GST Act.", fill="#666666", font=_get_font(14))

    out_path = DATA_DIR / filename
    img.save(out_path, "PNG")
    print(f"Created: {out_path}")
    return out_path


def create_udyam_cert(
    filename: str,
    udyam_no: str,
    enterprise_name: str,
    enterprise_type: str,
    city: str,
    state: str,
):
    img = Image.new("RGB", (1000, 700), color="white")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(26)
    header_font = _get_font(20)
    body_font = _get_font(18)

    # Border
    draw.rectangle([(20, 20), (980, 680)], outline="#004080", width=3)

    # Header
    draw.text((230, 45), "MINISTRY OF MICRO, SMALL & MEDIUM ENTERPRISES", fill="#004080", font=title_font)
    draw.text((370, 85), "UDYAM REGISTRATION CERTIFICATE", fill="#222222", font=header_font)
    draw.line([(50, 120), (950, 120)], fill="#004080", width=2)

    # Fields
    y = 150
    fields = [
        ("UDYAM REGISTRATION NUMBER:", udyam_no),
        ("NAME OF ENTERPRISE:", enterprise_name),
        ("TYPE OF ENTERPRISE:", enterprise_type),
        ("MAJOR ACTIVITY:", "Services"),
        ("CITY:", city),
        ("STATE:", state),
        ("DATE OF INCORPORATION:", "12/04/2018"),
        ("DATE OF UDYAM REGISTRATION:", "15/08/2020"),
    ]

    for label, val in fields:
        draw.text((60, y), label, fill="#111111", font=body_font)
        draw.text((420, y), val, fill="black", font=body_font)
        y += 45

    out_path = DATA_DIR / filename
    img.save(out_path, "PNG")
    print(f"Created: {out_path}")
    return out_path


if __name__ == "__main__":
    create_gstin_cert(
        "K01_gstin_active.png",
        gstin="32TTNXB7384T1ZW",
        legal_name="Balaji Dental Care LLP",
        trade_name="Balaji Dental Care",
        address="12/4 Marine Drive, Kochi, Kerala, 682001",
    )

    create_gstin_cert(
        "K02_gstin_cancelled.png",
        gstin="37IHKXV5167L1ZS",
        legal_name="Neha Kaur (Prop. Vindhya Tours & Travels)",
        trade_name="Vindhya Tours & Travels",
        address="45 Station Road, Gwalior, Madhya Pradesh, 474002",
    )

    create_udyam_cert(
        "K03_udyam_active.png",
        udyam_no="UDYAM-KL-00-3078801",
        enterprise_name="Balaji Dental Care LLP",
        enterprise_type="Micro",
        city="Kochi",
        state="Kerala",
    )

    create_gstin_cert(
        "K04_gstin_watermarked.png",
        gstin="32TTNXB7384T1ZW",
        legal_name="Balaji Dental Care LLP",
        trade_name="Balaji Dental Care",
        address="12/4 Marine Drive, Kochi, Kerala, 682001",
        watermark="GOVERNMENT OF INDIA",
    )
