import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Tests never depend on a running Ollama; LLM-path tests inject a mock HTTP client instead.
os.environ.setdefault("DISABLE_LLM", "1")

SAMPLES = ROOT / "samples" / "docs"
