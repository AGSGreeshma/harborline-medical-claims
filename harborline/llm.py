"""Local model via Ollama (Qwen2.5). JSON output constrained by a schema, temperature 0.

The model only ever receives MASKED text. If Ollama is not running, the caller falls back to the
deterministic extractor and the trace says so.
"""
from __future__ import annotations

import json

import httpx

from .config import get_settings
from .schemas import LLMOutput

SYSTEM = """You extract fields from Indian medical documents for an insurance claims team.
The text is OCR output. Personal data has been replaced by placeholders like <PERSON_1>, <DATE_TIME_2>,
<MEDICAL_CONDITION_1>. When a field's value is a placeholder, return the placeholder exactly as written.

Return JSON only, with these keys:
- document_type: "Hospital Bill", "Prescription" or "Lab Report"
- patient_name: the patient (not the doctor)
- hospital_name: the hospital, clinic or laboratory that issued the document
- bill_date: the bill / report / prescription date as written
- total_amount: number in rupees, ONLY for hospital bills. Use the GRAND TOTAL / NET or AMOUNT PAYABLE
  (after tax and discounts). NEVER use a subtotal, a taxable value, a single line item or an advance paid.
  Use null for prescriptions and lab reports, or when the total cannot be read.
- total_in_words: the amount written in words exactly as printed, or null
- diagnosis_or_test: the diagnosis (bills, prescriptions) or the main test name (lab reports)
Use null for anything not present. Do not guess."""


class LLMUnavailable(RuntimeError):
    pass


def extract_with_ollama(masked_text: str, client: httpx.Client | None = None) -> tuple[LLMOutput, str]:
    """Returns (output, model_name). Raises LLMUnavailable if Ollama can't be used."""
    s = get_settings()
    if s.disable_llm:
        raise LLMUnavailable("LLM disabled (DISABLE_LLM=1)")
    body = {
        "model": s.ollama_model,
        "stream": False,
        "format": LLMOutput.model_json_schema(),
        "options": {"temperature": 0, "seed": 0},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Document text:\n\"\"\"\n{masked_text[:12000]}\n\"\"\""},
        ],
    }
    own = client is None
    client = client or httpx.Client(timeout=s.ollama_timeout)
    try:
        r = client.post(f"{s.ollama_url}/api/chat", json=body)
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"Ollama not reachable at {s.ollama_url} ({type(e).__name__})") from e
    finally:
        if own:
            client.close()
    if r.status_code == 404:
        raise LLMUnavailable(f"Model '{s.ollama_model}' not found. Run: ollama pull {s.ollama_model}")
    if r.status_code >= 400:
        raise LLMUnavailable(f"Ollama error {r.status_code}: {r.text[:200]}")
    try:
        content = r.json()["message"]["content"]
        data = json.loads(content)
        return LLMOutput.model_validate(data), s.ollama_model
    except Exception as e:
        raise LLMUnavailable(f"Model returned invalid JSON ({type(e).__name__})") from e
