"""The model path, tested against a mock Ollama server.

Proves that (1) only masked text is sent to the model, (2) placeholders in the model's answer are restored
locally, (3) a subtotal picked by the model is overridden by the labelled grand total, and (4) the pipeline
falls back to the rule extractor when Ollama is down.
"""
import json

import httpx

from conftest import SAMPLES
from harborline.pipeline import run

SECRETS_06 = ["Chatura", "Chaudhari", "4371 3287 5040", "Appendicitis", "07/10/2026"]


def ollama_mock(reply: dict, seen: list):
    def handler(request: httpx.Request):
        seen.append(request.content.decode())
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(reply)}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_model_sees_only_masked_text_and_values_are_restored(monkeypatch):
    monkeypatch.setenv("DISABLE_LLM", "0")
    seen = []
    reply = {"document_type": "Hospital Bill", "patient_name": "<PERSON_2>", "hospital_name": "Lotus Care Hospital",
             "bill_date": "<DATE_TIME_1>", "total_amount": 52440, "total_in_words": None,
             "diagnosis_or_test": "<MEDICAL_CONDITION_1>"}
    r = run((SAMPLES / "06_hospital_bill.png").read_bytes(), "06.png", llm_client=ollama_mock(reply, seen))

    sent = seen[0]
    for secret in SECRETS_06:
        assert secret not in sent, f"real value sent to the model: {secret}"
    body = json.loads(sent)
    assert body["options"]["temperature"] == 0 and body["format"]["type"] == "object"

    assert r.extractor == "ollama"
    assert r.fields.patient_name == "Chatura Chaudhari"
    assert r.fields.bill_date == "07/10/2026"
    assert r.fields.diagnosis_or_test == "Appendicitis"
    assert r.result.decision == "Needs Review"
    assert r.cleanup["placeholder_map_cleared"] is True


def test_model_subtotal_is_overridden_by_grand_total(monkeypatch):
    monkeypatch.setenv("DISABLE_LLM", "0")
    reply = {"document_type": "Hospital Bill", "total_amount": 49942.86}  # the pre-tax subtotal: under the limit!
    r = run((SAMPLES / "06_hospital_bill.png").read_bytes(), "06.png", llm_client=ollama_mock(reply, []))
    assert r.fields.total_amount == 52440.0
    assert r.result.decision == "Needs Review"
    assert any("replaced by labelled" in n for n in r.trace[2].data["notes"])


def test_falls_back_when_ollama_is_down(monkeypatch):
    monkeypatch.setenv("DISABLE_LLM", "0")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here
    r = run((SAMPLES / "03_hospital_bill.png").read_bytes(), "03.png")
    assert r.extractor == "rules" and r.trace[2].status == "fallback"
    assert r.result.decision == "Within Limit"  # exactly ₹50,000
