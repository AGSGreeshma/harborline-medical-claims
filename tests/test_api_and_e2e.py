"""End to end through the real OCR on the synthetic set, and API == page JSON."""
import pytest
from fastapi.testclient import TestClient

from api import app
from conftest import SAMPLES
from harborline.pipeline import run

client = TestClient(app)


@pytest.mark.parametrize("file,decision,total", [
    ("03_hospital_bill.png", "Within Limit", 50000.0),   # exactly the limit
    ("04_hospital_bill.pdf", "Needs Review", 50001.0),   # one rupee over, scanned PDF
    ("06_hospital_bill.png", "Needs Review", 52440.0),   # subtotal under, grand total over
    ("08_hospital_bill.png", "Needs Review", 45000.0),   # words do not match figures
    ("13_prescription.pdf", "Not Applicable", None),
    ("17_lab_report.pdf", "Not Applicable", None),
])
def test_end_to_end_decisions(file, decision, total):
    r = run((SAMPLES / file).read_bytes(), file)
    assert r.result.decision == decision
    assert r.fields.total_amount == total
    assert [t.stage for t in r.trace] == ["ocr", "pii", "llm", "rule"]


def test_api_returns_same_json_as_the_page():
    data = (SAMPLES / "06_hospital_bill.png").read_bytes()
    api = client.post("/v1/extract", files={"file": ("06.png", data, "image/png")}).json()
    page = run(data, "06.png").model_dump(mode="json")
    for key in ["fields", "result", "masked_text", "entities_masked", "extractor", "quality"]:
        assert api[key] == page[key], key
    assert set(api) == set(page)


def test_api_rejects_bad_files_with_400():
    r = client.post("/v1/extract", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]


def test_health():
    assert client.get("/health").json()["status"] == "ok"
