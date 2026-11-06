"""FastAPI backend.

    uvicorn api:app --reload --port 8000

POST /v1/extract   multipart/form-data, field "file"  ->  the same JSON the Streamlit page shows
GET  /health       liveness + whether the local model is reachable
"""
from __future__ import annotations

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from harborline.config import get_settings
from harborline.pipeline import run
from harborline.schemas import ExtractResponse
from harborline.validation import ValidationError

app = FastAPI(
    title="Harborline Medical Bill Reader",
    version="1.0.0",
    description="OCR -> mask personal data -> local model -> restore -> ₹50,000 rule. Everything runs locally.",
)


@app.get("/health")
def health():
    s = get_settings()
    ollama = "disabled" if s.disable_llm else "unreachable"
    if not s.disable_llm:
        try:
            r = httpx.get(f"{s.ollama_url}/api/tags", timeout=2)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            ollama = "ready" if any(n.startswith(s.ollama_model.split(":")[0]) for n in names) else f"model {s.ollama_model} not pulled"
        except Exception:
            pass
    return {"status": "ok", "ollama": ollama, "model": s.ollama_model, "bill_limit": s.bill_limit}


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    s = get_settings()
    data = await file.read(int(s.max_upload_mb * 1024 * 1024) + 1)
    try:
        # OCR and the model are blocking; keep the event loop free.
        return await run_in_threadpool(run, data, file.filename or "upload")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        await file.close()
