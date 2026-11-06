"""Streamlit front end.   streamlit run streamlit_app.py

Calls the same pipeline as POST /v1/extract, so the page and the API return identical JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz
import httpx
import streamlit as st

from harborline.config import get_settings
from harborline.pipeline import run
from harborline.validation import ValidationError

ROOT = Path(__file__).parent
SAMPLES = sorted((ROOT / "samples" / "docs").glob("*.*"))

st.set_page_config(page_title="Harborline · Medical Bill Reader", page_icon="🩺", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.decision {padding: 14px 18px; border-radius: 10px; font-size: 1.05rem; margin: 6px 0 14px;}
.within {background:#e7f5ec; color:#14532d; border:1px solid #9bd1ad;}
.review {background:#fdecea; color:#7f1d1d; border:1px solid #f1a8a0;}
.na {background:#eef2f7; color:#1e293b; border:1px solid #cbd5e1;}
.stage {border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; height:100%;}
.stage h4 {margin:0 0 4px; font-size:0.95rem;}
.stage .ms {float:right; font-family:monospace; color:#64748b; font-size:0.8rem;}
.pill {display:inline-block; padding:1px 8px; border-radius:20px; font-size:0.72rem; font-weight:600; margin-bottom:6px;}
.ok {background:#e7f5ec; color:#166534;} .fallback {background:#fef3c7; color:#92400e;}
.skipped {background:#f1f5f9; color:#475569;} .error {background:#fdecea; color:#991b1b;}
.small {color:#64748b; font-size:0.82rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=15)
def ollama_status() -> str:
    s = get_settings()
    if s.disable_llm:
        return "disabled (DISABLE_LLM=1) — rule extractor"
    try:
        r = httpx.get(f"{s.ollama_url}/api/tags", timeout=2)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if any(n.startswith(s.ollama_model.split(":")[0]) for n in names):
            return f"ready · {s.ollama_model}"
        return f"running, but `{s.ollama_model}` is not pulled — run `ollama pull {s.ollama_model}`"
    except Exception:
        return "not running — the rule extractor is used instead"


s = get_settings()
with st.sidebar:
    st.markdown("### 🩺 Harborline")
    st.caption("Medical Bill Reader · Team Gamma · AI FDE Fellowship")
    st.markdown(f"**Local model:** {ollama_status()}")
    st.markdown(f"**Hospital-bill limit:** ₹{s.bill_limit:,.0f} (inclusive)")
    st.markdown(f"**Low-quality threshold:** < {s.min_ocr_chars} characters or confidence < {s.min_ocr_confidence:g}")
    st.divider()
    st.markdown(
        "**Privacy**  \nOCR, masking and the model all run on this machine. The model only sees masked text; "
        "real values are restored locally, and the placeholder map is cleared after each document. Nothing is stored."
    )

st.title("Medical Bill Reader")
st.caption("Upload a hospital bill, prescription or lab report → OCR → mask personal data → local model → restore → ₹50,000 rule")

c1, c2 = st.columns([3, 2])
with c1:
    up = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp"])
with c2:
    pick = st.selectbox("…or try a synthetic sample", ["—"] + [p.name for p in SAMPLES])

data, name = None, None
if up is not None:
    data, name = up.getvalue(), up.name
elif pick != "—":
    p = ROOT / "samples" / "docs" / pick
    data, name = p.read_bytes(), p.name

if data is None:
    st.info("Upload a document or pick a sample to start. Synthetic samples contain fictional people only.")
    st.stop()

with st.spinner("Reading the document…"):
    try:
        res = run(data, name)
    except ValidationError as e:
        st.error(f"File rejected: {e}")
        st.stop()

out = res.model_dump()
d = res.result
cls = {"Within Limit": "within", "Needs Review": "review", "Not Applicable": "na"}[d.decision]
icon = {"Within Limit": "✅", "Needs Review": "⚠️", "Not Applicable": "ℹ️"}[d.decision]
st.markdown(f'<div class="decision {cls}">{icon} <b>{d.decision}</b> — {d.reason}</div>', unsafe_allow_html=True)

# ---- four-stage trace panel
st.subheader("Trace")
cols = st.columns(4)
labels = {"ocr": "1 · OCR", "pii": "2 · Mask personal data", "llm": "3 · Model extraction", "rule": "4 · Business rule"}
for col, t in zip(cols, res.trace):
    with col:
        notes = "<br>".join(t.data.get("notes", []) or [])
        st.markdown(
            f'<div class="stage"><span class="ms">{t.ms:.0f} ms</span><h4>{labels[t.stage]}</h4>'
            f'<span class="pill {t.status}">{t.status.upper()}</span><div class="small">{t.detail}</div>'
            f'<div class="small">{notes}</div></div>',
            unsafe_allow_html=True,
        )

# ---- fields + documents
left, right = st.columns([1, 1])
with left:
    st.subheader("Extracted fields")
    f = res.fields
    rows = {
        "Document type": f.document_type or "—",
        "Patient name": f.patient_name or "—",
        "Hospital name": f.hospital_name or "—",
        "Bill date": f.bill_date or "—",
        "Total amount": f"₹{f.total_amount:,.2f}" if f.total_amount is not None else "—",
        "Diagnosis or test": f.diagnosis_or_test or "—",
    }
    st.table({"Field": list(rows), "Value": list(rows.values())})
    st.caption(f"Extractor: **{res.extractor}** · amount-in-words check: **{d.words_check}** · "
               f"OCR: {res.quality['ocr_path']}, confidence {res.quality['confidence']}")
    if res.entities_masked:
        st.caption("Masked before the model: " + ", ".join(f"{k} ×{v}" for k, v in res.entities_masked.items()))

with right:
    t1, t2 = st.tabs(["What the model saw (masked)", "Original"])
    with t1:
        st.code(res.masked_text, language=None)
    with t2:
        if name.lower().endswith(".pdf"):
            with fitz.open(stream=data, filetype="pdf") as doc:
                st.image(doc[0].get_pixmap(dpi=110).tobytes("png"), use_container_width=True)
        else:
            st.image(data, use_container_width=True)

with st.expander("JSON (identical to POST /v1/extract)"):
    st.json(out)
st.download_button("Download JSON", json.dumps(out, indent=2, ensure_ascii=False), file_name=f"{Path(name).stem}.json",
                   mime="application/json")
