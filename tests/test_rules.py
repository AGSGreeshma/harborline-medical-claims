"""Business-rule edge cases. The limit is INCLUSIVE: exactly ₹50,000 is Within Limit."""
import pytest

from harborline.rules import apply_rules
from harborline.schemas import ExtractedFields

W = "Rupees Fifty Thousand Only"


def bill(total):
    return ExtractedFields(document_type="Hospital Bill", total_amount=total)


@pytest.mark.parametrize("total", [1, 18450, 49999, 49999.99, 50000, 50000.00])
def test_at_or_below_limit_is_within(total):
    r = apply_rules(bill(total), None, limit=50000)
    assert r.decision == "Within Limit"
    assert r.total_used == total


@pytest.mark.parametrize("total", [50000.01, 50001, 52440, 148825])
def test_above_limit_needs_review(total):
    r = apply_rules(bill(total), None, limit=50000)
    assert r.decision == "Needs Review"
    assert r.reason == "Total is above ₹50,000."


@pytest.mark.parametrize("total", [None, 0, -5])
def test_unreadable_total_needs_review(total):
    r = apply_rules(bill(total), None, limit=50000)
    assert r.decision == "Needs Review"
    assert r.reason.startswith("Total could not be read.")


def test_words_match_keeps_decision():
    r = apply_rules(bill(50000), W, limit=50000)
    assert r.decision == "Within Limit" and r.words_check == "match"


def test_words_mismatch_needs_review_even_if_under_limit():
    r = apply_rules(bill(45000), "Rupees Forty Thousand Only", limit=50000)
    assert r.decision == "Needs Review"
    assert r.reason.startswith("Total could not be read.")
    assert r.words_check == "mismatch"


def test_unparseable_words_count_as_mismatch():
    r = apply_rules(bill(45000), "Rupees Forty Thousnd Only", limit=50000)
    assert r.decision == "Needs Review"


@pytest.mark.parametrize("dtype", ["Prescription", "Lab Report"])
def test_non_bills_are_not_applicable_even_with_an_amount(dtype):
    r = apply_rules(ExtractedFields(document_type=dtype, total_amount=99999), None, limit=50000)
    assert r.decision == "Not Applicable"


def test_low_quality_scan_goes_to_manual_review():
    r = apply_rules(bill(1000), None, low_quality=True, limit=50000)
    assert r.decision == "Needs Review" and "manual review" in r.reason


def test_unknown_document_type_needs_review():
    assert apply_rules(ExtractedFields(), None, limit=50000).decision == "Needs Review"


def test_limit_is_configurable():
    assert apply_rules(bill(60000), None, limit=75000).decision == "Within Limit"
    assert apply_rules(bill(60000), None, limit=75000).reason == "Total is within ₹75,000 (the limit is inclusive)."
