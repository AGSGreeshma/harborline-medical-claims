"""Generate 20 synthetic medical documents (10 hospital bills, 5 prescriptions, 5 lab reports) with ground truth.

ReportLab draws each document; it is then rasterized and degraded (noise, slight rotation, blur, JPEG
compression) so OCR is really exercised. Faker (en_IN) supplies fake names, phones, Aadhaar-style numbers.
All people and numbers are fictional.

Usage:  python scripts/generate_samples.py            -> samples/docs/*.{png,jpg,pdf} + samples/ground_truth.csv
"""
from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import cv2
import fitz
import numpy as np
from faker import Faker
from num2words import num2words
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "docs"
GT = ROOT / "samples" / "ground_truth.csv"

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

HOSPITALS = ["Sunrise Multispeciality Hospital", "Lotus Care Hospital", "Ganga Medical Centre", "Shreeji Heart Institute",
             "Nirmal Nursing Home", "Aarogya Superspeciality Hospital", "Seva Sadan Hospital", "Medihope Hospital"]
LABS = ["Precision Diagnostics Laboratory", "CityCare Pathology Lab", "Healthline Diagnostics Centre"]
CLINICS = ["Dr. Mehra's Family Clinic", "Wellness Polyclinic", "Sanjeevani Clinic"]
DIAGNOSES = ["Dengue fever", "Acute gastroenteritis", "Typhoid fever", "Pneumonia", "Kidney stone", "Appendicitis", "Viral fever", "Fracture of left radius"]
TESTS = ["Complete Blood Count", "Lipid Profile", "HbA1c", "Thyroid Profile", "Liver Function Test"]


def words(n: float) -> str:
    return "Rupees " + num2words(int(n), lang="en_IN").replace(",", "").title() + " Only"


def inr(n: float) -> str:
    s = f"{n:,.2f}"
    whole, dec = s.split(".")
    w = whole.replace(",", "")
    if len(w) > 3:
        head, tail = w[:-3], w[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{dec}"


def person():
    return f"{fake.first_name()} {fake.last_name()}"


def aadhaar():
    return f"{random.randint(2,9)}{random.randint(100,999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"


def mobile():
    return f"+91 {random.randint(6,9)}{random.randint(1000,9999)} {random.randint(10000,99999)}"


def date():
    return f"{random.randint(1,28):02d}/{random.randint(1,10):02d}/2026"


class Doc:
    def __init__(self):
        self.buf = io.BytesIO()
        self.c = canvas.Canvas(self.buf, pagesize=A4)
        self.y = A4[1] - 60

    def line(self, text, size=11, bold=False, x=50, gap=17):
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.drawString(x, self.y, text)
        self.y -= gap

    def row(self, left, right, bold=False):
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
        self.c.drawString(60, self.y, left)
        self.c.drawRightString(540, self.y, right)
        self.y -= 17

    def rule(self):
        self.c.line(50, self.y + 8, 545, self.y + 8)
        self.y -= 6

    def pdf(self) -> bytes:
        self.c.showPage(); self.c.save()
        return self.buf.getvalue()


def header(d: Doc, org: str, title: str):
    d.line(org, 17, True, gap=20)
    d.line(f"{fake.building_number()}, {fake.street_name()}, {fake.city()} {fake.postcode()}", 9, gap=13)
    d.line(f"Ph: {mobile()}   GSTIN: 27{fake.bothify('?????####?1Z#').upper()}", 9, gap=24)
    d.line(title, 14, True, gap=24)


def hospital_bill(total: float, subtotal_trap: bool = False, words_text: str | None = None, advance: float = 0):
    d = Doc()
    hosp, pat, dt, diag = random.choice(HOSPITALS), person(), date(), random.choice(DIAGNOSES)
    header(d, hosp, "INPATIENT FINAL BILL")
    d.line(f"Patient Name: {pat}        Age/Sex: {random.randint(18,80)}/{random.choice('MF')}")
    d.line(f"UHID: {fake.bothify('UH-####-#####')}     Policy No: {fake.bothify('HLI-#####-##')}")
    d.line(f"Aadhaar: {aadhaar()}     Mobile: {mobile()}")
    d.line(f"Bill Date: {dt}     Consultant: Dr. {person()}")
    d.line(f"Diagnosis: {diag}", gap=24)
    d.row("Description", "Amount (Rs.)", True); d.rule()
    gst = round(total * 0.05 / 1.05, 2) if subtotal_trap else 0.0
    base = total - gst
    items = [("Room charges", 0.30), ("Doctor consultation", 0.12), ("Pharmacy and consumables", 0.28),
             ("Investigations", 0.18), ("Nursing charges", 0.12)]
    acc = 0.0
    for i, (name, share) in enumerate(items):
        amt = round(base * share, 2) if i < len(items) - 1 else round(base - acc, 2)
        acc += amt
        d.row(name, inr(amt))
    d.rule()
    if subtotal_trap:
        d.row("Sub Total", inr(base))
        d.row("GST 5%", inr(gst))
    d.row("Grand Total", inr(total), True)
    if advance:
        d.row("Advance paid", inr(advance))
        d.row("Balance payable", inr(total - advance), True)
    d.y -= 8
    d.line(f"Amount in words: {words_text or words(total)}", 10)
    d.y -= 30
    d.line("Authorised Signatory", 10)
    return d.pdf(), dict(patient_name=pat, hospital_name=hosp, bill_date=dt, total_amount=total,
                         diagnosis_or_test=diag, document_type="Hospital Bill")


def prescription():
    d = Doc()
    clinic, pat, dt, diag = random.choice(CLINICS), person(), date(), random.choice(DIAGNOSES)
    header(d, clinic, "PRESCRIPTION")
    d.line(f"Patient Name: {pat}        Age: {random.randint(5,80)}")
    d.line(f"Date: {dt}     Mobile: {mobile()}")
    d.line(f"Diagnosis: {diag}", gap=24)
    d.line("Rx", 16, True, gap=22)
    for med in random.sample(["Tab. Paracetamol 650 mg  1-0-1 after food  5 days", "Cap. Amoxicillin 500 mg  1-1-1  7 days",
                              "Syrup Ondansetron 5 ml  SOS", "Tab. Pantoprazole 40 mg  1-0-0 before food  10 days",
                              "ORS sachets  as advised"], 3):
        d.line(med, 11, x=70)
    d.y -= 20
    d.line("Advice: plenty of fluids, review after 5 days", 10)
    d.y -= 30
    d.line(f"Dr. {person()}  MBBS, MD   Reg. No. {fake.bothify('MMC-######')}", 10)
    return d.pdf(), dict(patient_name=pat, hospital_name=clinic, bill_date=dt, total_amount=None,
                         diagnosis_or_test=diag, document_type="Prescription")


def lab_report():
    d = Doc()
    lab, pat, dt, test = random.choice(LABS), person(), date(), random.choice(TESTS)
    header(d, lab, "LABORATORY TEST REPORT")
    d.line(f"Patient Name: {pat}        Age/Sex: {random.randint(18,80)}/{random.choice('MF')}")
    d.line(f"Report Date: {dt}     Referred By: Dr. {person()}")
    d.line(f"Test Name: {test}     Sample Type: Blood", gap=24)
    d.row("Investigation          Result      Units      Reference Range", "", True); d.rule()
    for name, val, unit, ref in [("Haemoglobin", "13.4", "g/dL", "13.0 - 17.0"), ("Total Leucocyte Count", "7,200", "/cumm", "4,000 - 11,000"),
                                 ("Platelet Count", "2.1", "lakh/cumm", "1.5 - 4.5"), ("Fasting Glucose", "96", "mg/dL", "70 - 100")]:
        d.line(f"{name:<24}{val:<12}{unit:<12}{ref}", 10, x=60, gap=16)
    d.y -= 20
    d.line("End of report. Results to be correlated clinically.", 9)
    return d.pdf(), dict(patient_name=pat, hospital_name=lab, bill_date=dt, total_amount=None,
                         diagnosis_or_test=test, document_type="Lab Report")


def degrade(pdf: bytes, level: float) -> np.ndarray:
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=170, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(np.float32)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-1.2, 1.2) * level, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    img = cv2.GaussianBlur(img, (3, 3), 0.4 + 0.6 * level)
    img += np.random.normal(0, 6 + 14 * level, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def save(img: np.ndarray, path: Path):
    if path.suffix == ".jpg":
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    elif path.suffix == ".pdf":  # a scanned PDF: image only, no text layer
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        doc = fitz.open()
        page = doc.new_page(width=A4[0], height=A4[1])
        page.insert_image(page.rect, stream=jpg.tobytes())
        doc.save(str(path), deflate=True)
    else:
        img = (img // 16) * 16 + 8  # 16 grey levels: same for OCR, far smaller PNG
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        # (generator, kwargs, extension, degradation level, expected decision, note)
        (hospital_bill, dict(total=18450.00), ".png", 0.3, "Within Limit", "small bill"),
        (hospital_bill, dict(total=49999.00), ".jpg", 0.4, "Within Limit", "one rupee under the limit"),
        (hospital_bill, dict(total=50000.00), ".png", 0.3, "Within Limit", "exactly the limit (inclusive)"),
        (hospital_bill, dict(total=50001.00), ".pdf", 0.4, "Needs Review", "one rupee over"),
        (hospital_bill, dict(total=148825.00), ".jpg", 0.5, "Needs Review", "lakh-scale bill"),
        (hospital_bill, dict(total=52440.00, subtotal_trap=True), ".png", 0.4, "Needs Review", "subtotal 49,942.86 is under, grand total is over"),
        (hospital_bill, dict(total=48300.00, subtotal_trap=True), ".pdf", 0.4, "Within Limit", "with GST lines"),
        (hospital_bill, dict(total=45000.00, words_text="Rupees Forty Thousand Only"), ".png", 0.3, "Needs Review", "amount in words does not match"),
        (hospital_bill, dict(total=62500.00, advance=20000.00), ".jpg", 0.4, "Needs Review", "advance paid; grand total is used"),
        (hospital_bill, dict(total=32750.00), ".jpg", 1.0, "Within Limit", "heavily degraded scan"),
    ] + [(prescription, {}, ext, lvl, "Not Applicable", "prescription") for ext, lvl in
         [(".png", 0.3), (".jpg", 0.5), (".pdf", 0.4), (".png", 0.6), (".jpg", 0.3)]] + [
        (lab_report, {}, ext, lvl, "Not Applicable", "lab report") for ext, lvl in
        [(".png", 0.3), (".pdf", 0.4), (".jpg", 0.5), (".png", 0.4), (".jpg", 0.6)]]

    rows = []
    for i, (gen, kw, ext, lvl, decision, note) in enumerate(specs, 1):
        pdf, truth = gen(**kw)
        name = f"{i:02d}_{truth['document_type'].lower().replace(' ', '_')}{ext}"
        save(degrade(pdf, lvl), OUT / name)
        rows.append({"file": name, **truth, "expected_decision": decision, "note": note})
        print(f"{name:<34} {decision:<15} {note}")

    with open(GT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} documents -> {OUT}\nground truth -> {GT}")


if __name__ == "__main__":
    main()
