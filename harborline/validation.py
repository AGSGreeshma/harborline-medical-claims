"""File validation: real type from the bytes (not the extension), size and page limits."""
from __future__ import annotations

import filetype
import fitz  # PyMuPDF

from .config import get_settings

ALLOWED = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpeg", "image/tiff": "tiff", "image/webp": "webp"}


class ValidationError(ValueError):
    pass


def validate_upload(data: bytes, filename: str) -> str:
    """Return the detected kind ('pdf', 'png', ...) or raise ValidationError."""
    s = get_settings()
    if not data:
        raise ValidationError("The file is empty.")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise ValidationError(f"The file is larger than {s.max_upload_mb:g} MB.")
    kind = filetype.guess(data)
    if kind is None or kind.mime not in ALLOWED:
        detected = kind.mime if kind else "unknown"
        raise ValidationError(f"Unsupported file type ({detected}). Upload a PDF, PNG, JPEG, TIFF or WEBP.")
    if kind.mime == "application/pdf":
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ValidationError("The PDF is password-protected.")
                if doc.page_count == 0:
                    raise ValidationError("The PDF has no pages.")
                if doc.page_count > s.max_pages:
                    raise ValidationError(f"The PDF has {doc.page_count} pages; the limit is {s.max_pages}.")
        except ValidationError:
            raise
        except Exception as e:  # corrupt PDF
            raise ValidationError(f"The PDF could not be opened ({type(e).__name__}).") from e
    return ALLOWED[kind.mime]
