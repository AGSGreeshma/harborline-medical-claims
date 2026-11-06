"""End-to-end pipeline:

  1 upload -> 2 validate -> 3 OCR (+ low-quality check / fallback) -> 4 detect & mask personal data
  -> 5 model extraction on masked text -> 6 restore real values locally -> 7 business rules
  -> 8 result (+ trace, cleanup)

The Streamlit page and POST /v1/extract both call run() and return the same JSON.
"""
from __future__ import annotations

import time
import uuid

from .config import get_settings
from .extract_rules import extract as rule_extract
from .extract_rules import find_total
from .llm import LLMUnavailable, extract_with_ollama
from .ocr import run_ocr
from .pii import has_placeholder, mask, restore
from .rules import apply_rules
from .schemas import ExtractedFields, ExtractResponse, LLMOutput, StageTrace
from .telemetry import log, tracer
from .validation import validate_upload


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def run(data: bytes, filename: str, llm_client=None) -> ExtractResponse:
    s = get_settings()
    rid = uuid.uuid4().hex[:12]
    tr = tracer()
    trace: list[StageTrace] = []

    with tr.start_as_current_span("extract") as root:
        root.set_attribute("request.id", rid)
        kind = validate_upload(data, filename)  # raises ValidationError
        root.set_attribute("file.kind", kind)

        # 3. OCR
        t0 = time.perf_counter()
        with tr.start_as_current_span("ocr") as sp:
            ocr = run_ocr(data, kind)
            sp.set_attribute("ocr.path", ocr.path)
            sp.set_attribute("ocr.chars", ocr.chars)
            sp.set_attribute("ocr.confidence", ocr.mean_conf)
            sp.set_attribute("ocr.low_quality", ocr.low_quality)
        trace.append(StageTrace(
            stage="ocr", status="fallback" if ocr.path == "fallback" else "ok", ms=_ms(t0),
            detail=f"{ocr.chars} characters · confidence {ocr.mean_conf:.0f} · {ocr.path}" + (" · LOW QUALITY" if ocr.low_quality else ""),
            data={"pages": ocr.pages, "path": ocr.path, "notes": ocr.notes},
        ))

        # 4. Detect and mask personal data
        t0 = time.perf_counter()
        with tr.start_as_current_span("pii") as sp:
            m = mask(ocr.text)
            sp.set_attribute("pii.entities", sum(m.counts.values()))
            for k, v in m.counts.items():
                sp.set_attribute(f"pii.{k}", v)
        trace.append(StageTrace(stage="pii", status="ok", ms=_ms(t0),
                                detail=f"{sum(m.counts.values())} values masked across {len(m.counts)} types",
                                data={"counts": m.counts}))

        # 5. Model extraction on masked text (fallback: deterministic extractor)
        t0 = time.perf_counter()
        rules_out = rule_extract(m.masked_text)
        notes: list[str] = []
        with tr.start_as_current_span("llm") as sp:
            try:
                out, model = extract_with_ollama(m.masked_text, client=llm_client)
                extractor, status, detail = "ollama", "ok", f"{model} · temperature 0 · JSON schema"
                # Fill gaps from the deterministic extractor
                for f in LLMOutput.model_fields:
                    if getattr(out, f) in (None, "") and getattr(rules_out, f) not in (None, ""):
                        setattr(out, f, getattr(rules_out, f))
                        notes.append(f"{f} filled by rule extractor")
            except LLMUnavailable as e:
                out, extractor, status = rules_out, "rules", "fallback"
                detail = f"Rule-based extractor ({e})"
            sp.set_attribute("llm.extractor", extractor)

            # Total guard: the labelled grand total / amount payable beats the model; never a subtotal.
            if out.document_type == "Hospital Bill":
                labelled, label = find_total(m.masked_text)
                if labelled is not None and out.total_amount is not None and abs(labelled - out.total_amount) > 0.5:
                    notes.append(f"model total {out.total_amount:g} replaced by labelled '{label}' {labelled:g}")
                    out.total_amount = labelled
                elif labelled is not None and out.total_amount is None:
                    out.total_amount = labelled
            else:
                out.total_amount = None
        trace.append(StageTrace(stage="llm", status=status, ms=_ms(t0), detail=detail, data={"notes": notes}))

        # 6. Restore real values locally
        fields = ExtractedFields(
            patient_name=restore(out.patient_name, m.vault),
            hospital_name=restore(out.hospital_name, m.vault),
            bill_date=restore(out.bill_date, m.vault),
            total_amount=out.total_amount,
            diagnosis_or_test=restore(out.diagnosis_or_test, m.vault),
            document_type=out.document_type,
        )
        unresolved = [k for k, v in fields.model_dump().items() if has_placeholder(v)]

        # 7. Business rules
        t0 = time.perf_counter()
        with tr.start_as_current_span("rule") as sp:
            result = apply_rules(fields, restore(out.total_in_words, m.vault), low_quality=ocr.low_quality)
            sp.set_attribute("rule.decision", result.decision)
        trace.append(StageTrace(stage="rule", status="ok", ms=_ms(t0), detail=f"{result.decision} — {result.reason}",
                                data={"words_check": result.words_check, "unresolved_placeholders": unresolved}))

        # 8. Cleanup: the placeholder map holds the only copy of the real values outside the result.
        n_vault = len(m.vault)
        m.vault.clear()
        log.info("extract rid=%s kind=%s ocr=%s conf=%.0f masked=%d extractor=%s decision=%s",
                 rid, kind, ocr.path, ocr.mean_conf, n_vault, extractor, result.decision)

        return ExtractResponse(
            request_id=rid,
            filename=filename,
            fields=fields,
            result=result,
            quality={"ocr_path": ocr.path, "confidence": ocr.mean_conf, "characters": ocr.chars,
                     "low_quality": ocr.low_quality, "thresholds": {"min_chars": s.min_ocr_chars, "min_confidence": s.min_ocr_confidence}},
            masked_text=m.masked_text,
            entities_masked=m.counts,
            extractor=extractor,
            trace=trace,
            cleanup={"upload_stored": False, "placeholder_map_cleared": True, "placeholders_cleared": n_vault},
        )
