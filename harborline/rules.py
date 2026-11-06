"""Business rules. Pure functions - every edge case is unit-tested in tests/test_rules.py."""
from __future__ import annotations

from typing import Optional

from .amounts import words_match
from .config import get_settings
from .schemas import ExtractedFields, RuleResult


def _fmt(limit: float) -> str:
    # Indian grouping: 50000 -> 50,000 ; 150000 -> 1,50,000
    n = int(round(limit))
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def apply_rules(fields: ExtractedFields, total_in_words: Optional[str], low_quality: bool = False,
                limit: Optional[float] = None) -> RuleResult:
    limit = get_settings().bill_limit if limit is None else limit
    L = f"₹{_fmt(limit)}"

    if fields.document_type in ("Prescription", "Lab Report"):
        return RuleResult(decision="Not Applicable", limit=limit,
                          reason=f"The {L} limit applies to hospital bills only; fields are still extracted.")

    if low_quality:
        return RuleResult(decision="Needs Review", limit=limit, total_used=fields.total_amount,
                          reason="Scan quality is too low to trust automatically; sent for manual review.")

    if fields.document_type is None:
        return RuleResult(decision="Needs Review", limit=limit, reason="Document type could not be determined.")

    # Hospital bill
    total = fields.total_amount
    if total is None or total <= 0:
        return RuleResult(decision="Needs Review", limit=limit, reason="Total could not be read.",
                          words_check="not_checked")
    check = words_match(total, total_in_words)
    if check == "mismatch":
        return RuleResult(decision="Needs Review", limit=limit, total_used=total, words_check=check,
                          reason="Total could not be read. (Amount in words does not match the figures.)")
    if total > limit:
        return RuleResult(decision="Needs Review", limit=limit, total_used=total, words_check=check,
                          reason=f"Total is above {L}.")
    return RuleResult(decision="Within Limit", limit=limit, total_used=total, words_check=check,
                      reason=f"Total is within {L} (the limit is inclusive).")
