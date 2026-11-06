"""OpenTelemetry spans (ocr, pii, llm, rule) and logging. Nothing personal is ever put in a span or a log line:
only counts, durations, statuses and entity TYPES."""
from __future__ import annotations

import logging
from functools import lru_cache

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from .config import get_settings

log = logging.getLogger("harborline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


@lru_cache(maxsize=1)
def tracer() -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": "harborline-bill-reader"}))
    if get_settings().otel_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    return provider.get_tracer("harborline")
