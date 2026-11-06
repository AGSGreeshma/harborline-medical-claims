"""Pydantic models. The page and POST /v1/extract both return ExtractResponse.model_dump()."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["Hospital Bill", "Prescription", "Lab Report"]
Decision = Literal["Within Limit", "Needs Review", "Not Applicable"]


class ExtractedFields(BaseModel):
    """The six fields the brief asks for. Values are restored (real) values, never placeholders."""

    patient_name: Optional[str] = None
    hospital_name: Optional[str] = None
    bill_date: Optional[str] = Field(None, description="As printed on the document")
    total_amount: Optional[float] = Field(None, description="Grand total / amount payable in rupees")
    diagnosis_or_test: Optional[str] = None
    document_type: Optional[DocType] = None


class LLMOutput(BaseModel):
    """What the model must return. It only ever sees masked text, so names come back as placeholders."""

    patient_name: Optional[str] = None
    hospital_name: Optional[str] = None
    bill_date: Optional[str] = None
    total_amount: Optional[float] = None
    total_in_words: Optional[str] = None
    diagnosis_or_test: Optional[str] = None
    document_type: Optional[DocType] = None


class StageTrace(BaseModel):
    stage: Literal["ocr", "pii", "llm", "rule"]
    status: Literal["ok", "fallback", "skipped", "error"]
    ms: float
    detail: str
    data: dict = Field(default_factory=dict)


class RuleResult(BaseModel):
    decision: Decision
    reason: str
    limit: float
    total_used: Optional[float] = None
    words_check: Literal["match", "mismatch", "not_present", "not_checked"] = "not_checked"


class ExtractResponse(BaseModel):
    request_id: str
    filename: str
    fields: ExtractedFields
    result: RuleResult
    quality: dict
    masked_text: str = Field(description="OCR text with personal data replaced by placeholders - what the model saw")
    entities_masked: dict = Field(description="Count of masked entities by type (no values)")
    extractor: Literal["ollama", "rules"]
    trace: list[StageTrace]
    cleanup: dict
