"""Rupee amounts: parse figures, parse amounts written in words, and compare the two using num2words."""
from __future__ import annotations

import re
from typing import Optional

from num2words import num2words

_AMOUNT = re.compile(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?:/-)?", re.IGNORECASE)


def parse_amount(s: str | None) -> Optional[float]:
    """'₹ 1,23,456.50' -> 123456.5 ; 'Rs. 48,825/-' -> 48825.0"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = _AMOUNT.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def amounts_in_line(line: str) -> list[float]:
    vals = []
    for m in _AMOUNT.finditer(line):
        raw = m.group(1)
        # skip percentages ("GST 5%") and tiny integers that are quantities
        after = line[m.end(): m.end() + 1]
        if after == "%":
            continue
        try:
            vals.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return vals


_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
           "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000, "billion": 1_000_000_000}


def words_to_number(text: str | None) -> Optional[float]:
    """'Rupees Forty Eight Thousand Eight Hundred Twenty Five Only' -> 48825.0 (paise supported)."""
    if not text:
        return None
    t = text.lower().replace("-", " ")
    t = re.sub(r"(amount|total)?\s*in\s*words\s*[:\-]?", " ", t)
    t = re.sub(r"[^a-z ]", " ", t)
    m = re.match(r"(.*)\band\b(.*)\bpaise\b", t)
    rupee_part, paise_part = (m.group(1), m.group(2)) if m else (t.replace("paise", ""), "")

    def parse(chunk: str) -> Optional[int]:
        total, current, seen = 0, 0, False
        for w in chunk.split():
            if w in {"rupees", "rupee", "rs", "inr", "only", "and", "indian"}:
                continue
            if w in _UNITS:
                current += _UNITS[w]; seen = True
            elif w in _TENS:
                current += _TENS[w]; seen = True
            elif w == "hundred":
                current = (current or 1) * 100; seen = True
            elif w in _SCALES:
                total += (current or 1) * _SCALES[w]; current = 0; seen = True
            else:
                return None  # unknown word -> unreadable
        return total + current if seen else None

    r = parse(rupee_part)
    if r is None:
        return None
    p = parse(paise_part) if paise_part else 0
    return float(r) + (p or 0) / 100


def words_for(amount: float) -> str:
    """Indian-English words for a figure, e.g. 148825 -> 'one lakh, forty-eight thousand, eight hundred and twenty-five'."""
    return num2words(int(round(amount)), lang="en_IN")


def words_match(amount: float | None, words_text: str | None) -> str:
    """'match' | 'mismatch' | 'not_present'. Compares by value, and cross-checks with num2words."""
    if not words_text or not words_text.strip():
        return "not_present"
    if amount is None:
        return "mismatch"
    parsed = words_to_number(words_text)
    if parsed is None:
        return "mismatch"
    if abs(parsed - amount) < 0.5 or abs(parsed - round(amount)) < 0.5:
        return "match"
    # num2words cross-check: normalise both renderings (handles "and", commas, hyphens)
    norm = lambda s: re.sub(r"\b(rupees?|only|and)\b|[^a-z]", "", s.lower())
    return "match" if norm(words_for(amount)) == norm(words_text) else "mismatch"
