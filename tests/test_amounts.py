import pytest

from harborline.amounts import parse_amount, words_match, words_to_number


@pytest.mark.parametrize("raw,want", [
    ("₹ 1,23,456.50", 123456.5), ("Rs. 48,825/-", 48825), ("INR 50,000.00", 50000), ("48825", 48825), ("Total 3,200", 3200),
])
def test_parse_amount(raw, want):
    assert parse_amount(raw) == want


@pytest.mark.parametrize("text,want", [
    ("Rupees Forty Eight Thousand Eight Hundred Twenty Five Only", 48825),
    ("Rupees Fifty Thousand Only", 50000),
    ("One Lakh Forty-Eight Thousand Eight Hundred and Twenty-Five Rupees", 148825),
    ("Rupees Two Crore Five Lakh Only", 20500000),
    ("Rupees One Thousand and Fifty Paise Only", 1000.5),
    ("Amount in words: Rupees Nine Hundred Ninety Nine Only", 999),
])
def test_words_to_number(text, want):
    assert words_to_number(text) == pytest.approx(want)


def test_words_to_number_rejects_garbage():
    assert words_to_number("Rupees Fourty Thousnd") is None


def test_words_match():
    assert words_match(48825, "Rupees Forty Eight Thousand Eight Hundred Twenty Five Only") == "match"
    assert words_match(48825, "Rupees Forty Eight Thousand Only") == "mismatch"
    assert words_match(48825, None) == "not_present"
