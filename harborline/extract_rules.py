"""Deterministic extractor. Runs on MASKED text, like the model.

Used (1) as the fallback when Ollama is not reachable, and (2) to cross-check the model's total:
the labelled grand total / amount payable always beats a model guess, and a pre-tax subtotal is never used.
"""
from __future__ import annotations

import re
from typing import Optional

from .amounts import amounts_in_line
from .schemas import LLMOutput

_DOC_KEYWORDS = {
    "Hospital Bill": ["bill", "invoice", "grand total", "amount payable", "net payable", "room charges", "discharge",
                      "ipd", "receipt", "gst", "sub total", "subtotal", "admission", "bed charges"],
    "Prescription": ["prescription", "rx", "tab.", "tab ", "cap.", "syrup", "dosage", "once daily", "twice daily",
                     "after food", "before food", "bd", "tds", "od ", "1-0-1", "1-1-1", "0-0-1", "refill", "advice"],
    "Lab Report": ["laboratory", "lab report", "test report", "reference range", "biological reference", "units",
                   "specimen", "sample collected", "sample type", "haemoglobin", "hemoglobin", "investigation", "pathology",
                   "result", "method"],
}

# Lines that hold the amount actually payable, in priority order.
_TOTAL_LABELS = [
    r"grand\s*total", r"net\s*(?:amount\s*)?payable", r"amount\s*payable", r"total\s*payable", r"balance\s*payable",
    r"total\s*amount\s*payable", r"net\s*amount", r"bill\s*amount", r"total\s*amount", r"total",
]
_NOT_TOTAL = re.compile(r"sub\s*-?\s*total|taxable|before\s*tax|pre[- ]tax|discount|advance|deposit|amount\s*paid|already\s*paid|in\s*words", re.I)
_TAX_LINE = re.compile(r"^\s*(?:c|s|i)?gst\b|^\s*tax\b", re.I)
_HOSPITAL_WORDS = re.compile(r"hospital|clinic|medical|health|diagnostic|laborator|lab\b|nursing\s*home|institute|centre|center|care|pathology|medicare|polyclinic", re.I)


def classify(text: str) -> Optional[str]:
    low = text.lower()
    scores = {k: sum(low.count(w) for w in ws) for k, ws in _DOC_KEYWORDS.items()}
    # strong signals
    if re.search(r"\b(lab(?:oratory)?\s+report|test\s+report|reference\s+range)\b", low):
        scores["Lab Report"] += 5
    if re.search(r"\b(prescription|rx)\b", low):
        scores["Prescription"] += 5
    if re.search(r"\b(grand\s+total|amount\s+payable|net\s+payable|invoice|hospital\s+bill|inpatient\s+bill|final\s+bill)\b", low):
        scores["Hospital Bill"] += 5
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# Labels that can follow a value on the same line; a value stops where one of these begins.
_NEXT_LABEL = (r"\s+(?:sample\s*type|age\s*/\s*sex|age|sex|gender|consultant|referred\s*by|ref\.?\s*by|mobile|ph|phone|"
               r"uhid|mrn|policy\s*no\.?|report\s*date|bill\s*date|date|reg\.?\s*no\.?|ip\s*no\.?|dob|aadhaar)\s*[:.]")


def _labelled(text: str, labels: str) -> Optional[str]:
    m = re.search(rf"(?:{labels})\s*[:\-]\s*(.+?)(?=\s{{2,}}|{_NEXT_LABEL}|$)", text, re.I | re.M)
    return m.group(1).strip(" .:-") if m else None


def find_total(text: str) -> tuple[Optional[float], Optional[str]]:
    """Return (amount, label) from the highest-priority total line; never a subtotal, tax or advance line."""
    lines = text.splitlines()
    for lab in _TOTAL_LABELS:
        rx = re.compile(rf"\b{lab}\b", re.I)
        for line in reversed(lines):  # totals sit at the bottom; the last matching line wins
            m = rx.search(line)
            if not m or _NOT_TOTAL.search(line) or _TAX_LINE.search(line):
                continue
            vals = [v for v in amounts_in_line(line[m.end():]) if v >= 1]
            if vals:
                return vals[-1], m.group(0)
    return None, None


def find_words(text: str) -> Optional[str]:
    m = re.search(r"(?:amount|total|rupees)?\s*in\s*words\s*[:\-]?\s*(.+)$", text, re.I | re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"^\s*(rupees\s+[a-z\s-]+only)\s*$", text, re.I | re.M)
    return m.group(1).strip() if m else None


def extract(masked_text: str) -> LLMOutput:
    lines = [l.strip() for l in masked_text.splitlines() if l.strip()]
    doc_type = classify(masked_text)

    hospital = next((l for l in lines[:8] if _HOSPITAL_WORDS.search(l) and not re.search(r"bill|invoice|report|prescription", l, re.I)), None)
    if hospital is None and lines:
        hospital = lines[0]

    patient = _labelled(masked_text, r"patient(?:'s)?\s*name|name\s+of\s+patient|patient|name")
    date = _labelled(masked_text, r"bill\s*date|invoice\s*date|report\s*date|date\s*of\s*report|collected\s*on|prescription\s*date|date")

    diag = _labelled(masked_text, r"final\s+diagnosis|provisional\s+diagnosis|diagnosis")
    if doc_type == "Lab Report":
        diag = _labelled(masked_text, r"test\s*name|investigation|test") or diag
    if doc_type == "Prescription" and not diag:
        diag = _labelled(masked_text, r"complaint|chief\s+complaint")

    total, words = (None, None)
    if doc_type == "Hospital Bill":
        total, _ = find_total(masked_text)
        words = find_words(masked_text)
    return LLMOutput(patient_name=patient, hospital_name=hospital, bill_date=date, total_amount=total,
                     total_in_words=words, diagnosis_or_test=diag, document_type=doc_type)
