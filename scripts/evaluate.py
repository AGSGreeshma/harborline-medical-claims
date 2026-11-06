"""Run the pipeline over the 20-document set and score it against samples/ground_truth.csv.

Three accuracy numbers:
  1. Field accuracy      - share of the 6 fields x 20 documents that match the ground truth
  2. Document-type accuracy
  3. Decision accuracy   - Within Limit / Needs Review / Not Applicable

Writes eval/results.csv (one row per document) and eval/summary.json.
Usage:  python scripts/evaluate.py            (uses Ollama if running, else the rule extractor)
        DISABLE_LLM=1 python scripts/evaluate.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harborline.pipeline import run  # noqa: E402

FIELDS = ["patient_name", "hospital_name", "bill_date", "total_amount", "diagnosis_or_test", "document_type"]


def norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(v).lower().replace("dr.", ""))


def same(field: str, got, want) -> bool:
    if field == "total_amount":
        w = None if want is None or (isinstance(want, float) and pd.isna(want)) or want == "" else float(want)
        return (got is None and w is None) or (got is not None and w is not None and abs(float(got) - w) < 0.5)
    return norm(got) == norm(want)


def main():
    gt = pd.read_csv(ROOT / "samples" / "ground_truth.csv")
    rows = []
    extractor = None
    for _, t in gt.iterrows():
        data = (ROOT / "samples" / "docs" / t["file"]).read_bytes()
        t0 = time.perf_counter()
        r = run(data, t["file"])
        ms = (time.perf_counter() - t0) * 1000
        extractor = r.extractor
        got = r.fields.model_dump()
        row = {"file": t["file"], "note": t["note"], "ms": round(ms)}
        for f in FIELDS:
            ok = same(f, got[f], t[f])
            row[f"{f}_ok"] = ok
            row[f"{f}_got"] = got[f]
            row[f"{f}_want"] = t[f]
        row["decision_got"] = r.result.decision
        row["decision_want"] = t["expected_decision"]
        row["decision_ok"] = r.result.decision == t["expected_decision"]
        row["ocr_confidence"] = r.quality["confidence"]
        rows.append(row)
        flag = "OK " if row["decision_ok"] else "XX "
        misses = [f for f in FIELDS if not row[f"{f}_ok"]]
        print(f"{flag}{t['file']:<26} {r.result.decision:<15} conf {r.quality['confidence']:>5}  {ms:6.0f} ms  misses: {', '.join(misses) or '-'}")

    df = pd.DataFrame(rows)
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    field_acc = float(df[[f"{f}_ok" for f in FIELDS]].to_numpy().mean())
    summary = {
        "documents": len(df),
        "extractor": extractor,
        "field_accuracy": round(field_acc, 3),
        "document_type_accuracy": round(float(df["document_type_ok"].mean()), 3),
        "decision_accuracy": round(float(df["decision_ok"].mean()), 3),
        "per_field_accuracy": {f: round(float(df[f"{f}_ok"].mean()), 3) for f in FIELDS},
        "mean_ms_per_document": round(float(df["ms"].mean())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
