import fitz
import pytest

from conftest import SAMPLES
from harborline.validation import ValidationError, validate_upload


def test_real_types_accepted():
    assert validate_upload((SAMPLES / "01_hospital_bill.png").read_bytes(), "a.png") == "png"
    assert validate_upload((SAMPLES / "02_hospital_bill.jpg").read_bytes(), "a.jpg") == "jpeg"
    assert validate_upload((SAMPLES / "04_hospital_bill.pdf").read_bytes(), "a.pdf") == "pdf"


def test_type_comes_from_bytes_not_extension():
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload(b"just text pretending to be a pdf", "bill.pdf")
    # a PNG named .pdf is still accepted as a PNG
    assert validate_upload((SAMPLES / "01_hospital_bill.png").read_bytes(), "bill.pdf") == "png"


def test_empty_rejected():
    with pytest.raises(ValidationError, match="empty"):
        validate_upload(b"", "x.png")


def test_page_limit(monkeypatch):
    monkeypatch.setenv("MAX_PAGES", "2")
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    with pytest.raises(ValidationError, match="3 pages"):
        validate_upload(doc.tobytes(), "long.pdf")


def test_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    with pytest.raises(ValidationError, match="larger than"):
        validate_upload((SAMPLES / "01_hospital_bill.png").read_bytes(), "a.png")
