"""Detect and mask personal and health data with Microsoft Presidio + custom Indian recognizers.

Masking is reversible but only locally: every value is replaced by a placeholder such as <PERSON_1>,
and the placeholder -> value map (the "vault") never leaves this process. The model only sees the
masked text; real values are put back afterwards in restore().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .config import get_settings

# Presidio's email recognizer calls tldextract, which downloads a domain list from the internet by default.
# Force the bundled offline snapshot so masking never makes a network call.
try:  # pragma: no cover - depends on installed version
    import tldextract
    from presidio_analyzer.predefined_recognizers.generic import email_recognizer as _er

    _er.tldextract = type("_Offline", (), {"extract": staticmethod(tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None))})
except Exception:
    pass

# Entities we mask. Hospital names (ORG) and money amounts are deliberately NOT masked:
# they are not personal data and the model needs them.
ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IN_MOBILE", "LOCATION", "DATE_TIME",
    "IN_AADHAAR", "IN_PAN", "BANK_ACCOUNT", "POLICY_NUMBER", "PATIENT_ID", "MEDICAL_CONDITION",
]

# Diagnosis terms treated as sensitive health data.
CONDITIONS = [
    "type 2 diabetes mellitus", "type 2 diabetes", "type 1 diabetes", "diabetes mellitus", "diabetes",
    "hypertension", "dengue fever", "dengue", "typhoid fever", "typhoid", "malaria", "tuberculosis",
    "pneumonia", "covid-19", "covid", "hiv", "hepatitis b", "hepatitis c", "hepatitis", "asthma", "cancer",
    "carcinoma", "appendicitis", "cholecystitis", "cataract", "fracture", "migraine", "hypothyroidism",
    "hyperthyroidism", "anaemia", "anemia", "gastroenteritis", "urinary tract infection", "kidney stone",
    "renal calculus", "myocardial infarction", "angina", "stroke", "epilepsy", "depression", "anxiety disorder",
    "bronchitis", "sinusitis", "viral fever", "acute gastritis", "gastritis", "osteoarthritis", "arthritis",
]


class LabeledValueRecognizer(EntityRecognizer):
    """Finds the VALUE after a label, e.g. 'Policy No: HLI-2231-889' -> 'HLI-2231-889'."""

    def __init__(self, entity: str, label_regex: str, value_regex: str, score: float = 0.9):
        super().__init__(supported_entities=[entity], name=f"Labeled{entity}", supported_language="en")
        self.entity = entity
        self.rx = re.compile(rf"\b(?:{label_regex})\s*[:#\-.]?\s*({value_regex})", re.IGNORECASE)
        self.score = score

    def load(self) -> None:  # required by the interface
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        out = []
        for m in self.rx.finditer(text):
            val = m.group(1).strip()
            if val:
                start = m.start(1)
                out.append(RecognizerResult(self.entity, start, start + len(m.group(1).rstrip()), self.score))
        return out


def _custom_recognizers() -> list[EntityRecognizer]:
    name = r"(?:(?:Mr|Mrs|Ms|Dr|Smt|Shri)\.? )?[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z.]*){0,3}(?<!\bAge)(?<!\bSex)"
    return [
        PatternRecognizer("IN_AADHAAR", name="AadhaarRecognizer",
                          patterns=[Pattern("aadhaar", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b", 0.85)],
                          context=["aadhaar", "uid", "aadhar"]),
        PatternRecognizer("IN_PAN", name="PanRecognizer",
                          patterns=[Pattern("pan", r"\b[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]\b", 0.85)]),
        PatternRecognizer("IN_MOBILE", name="IndianMobileRecognizer",
                          patterns=[Pattern("mobile", r"(?<!\d)(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)", 0.75)]),
        LabeledValueRecognizer("BANK_ACCOUNT", r"(?:bank\s+)?(?:a/c|account)\s*(?:no\.?|number)?", r"\d[\d -]{7,20}\d"),
        LabeledValueRecognizer("POLICY_NUMBER", r"(?:policy|member\s*id|insurance\s*id|tpa\s*id|claim)\s*(?:no\.?|number|id)?", r"[A-Z0-9][A-Z0-9/-]{4,24}"),
        LabeledValueRecognizer("PATIENT_ID", r"(?:uhid|mrn|ip\s*no\.?|op\s*no\.?|patient\s*id|reg(?:istration)?\s*no\.?)", r"[A-Z0-9][A-Z0-9/-]{3,20}"),
        LabeledValueRecognizer("PERSON", r"(?:patient(?:'s)?\s*name|patient|name\s+of\s+patient|(?<!test )(?<!test)(?<!sample )(?<!drug )(?<!brand )name|consultant|doctor|referred\s+by|ref\.?\s*by|attending\s+physician)", name, 0.95),
        LabeledValueRecognizer("DATE_TIME", r"(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date)", r"[0-9A-Za-z][0-9A-Za-z /.,-]{5,20}?\d{2,4}", 0.95),
        LabeledValueRecognizer("MEDICAL_CONDITION", r"(?:diagnosis|provisional\s+diagnosis|final\s+diagnosis|condition|complaint)", r"[A-Za-z][A-Za-z0-9 ,()'/-]{2,60}?(?=\s{2,}|\n|$)", 0.9),
        PatternRecognizer("MEDICAL_CONDITION", name="ConditionTermsRecognizer", deny_list=CONDITIONS, deny_list_score=0.8),
    ]


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    s = get_settings()
    model = s.spacy_model
    if not model:
        import importlib.util
        model = "en_core_web_lg" if importlib.util.find_spec("en_core_web_lg") else "en_core_web_sm"
    provider = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]})
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    for r in _custom_recognizers():
        analyzer.registry.add_recognizer(r)
    return analyzer


@dataclass
class MaskResult:
    masked_text: str
    vault: dict[str, str] = field(default_factory=dict)  # placeholder -> real value (stays local)
    counts: dict[str, int] = field(default_factory=dict)  # entity type -> count (safe to log)


# Words spaCy sometimes tags as PERSON/LOCATION on bills that are not personal data.
_NOT_PERSONAL = re.compile(
    r"^(?:total|grand total|amount|bill|invoice|hospital|clinic|lab|laboratory|date|tablet|tab|capsule|cap|syrup|"
    r"gst|cgst|sgst|rupees|only|net payable|amount payable|room|pharmacy|consultation|investigation|report|test|"
    r"result|reference|range|units|qty|rate|sr|no|description)\b",
    re.IGNORECASE,
)
# A "name" containing any of these words is a test, drug or form label, not a person.
_NON_NAME_WORDS = {
    "blood", "count", "profile", "test", "type", "sample", "serum", "urine", "glucose", "thyroid", "lipid", "liver",
    "function", "kidney", "panel", "haemoglobin", "hemoglobin", "platelet", "tablet", "tab", "capsule", "cap", "syrup",
    "mg", "ml", "total", "grand", "amount", "charges", "room", "investigation", "investigations", "pharmacy", "nursing",
    "hospital", "clinic", "laboratory", "diagnostics", "centre", "center", "result", "range", "report", "bill",
    "consultation", "advice", "sub", "gst", "date", "age", "sex", "reference", "units", "hba1c", "complete",
}
_CALENDAR_DATE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?[ -](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ -,]*\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"^(?:rs\.?|inr|₹)?\s*[\d,]+(?:\.\d+)?$", re.IGNORECASE)

# Priority when two detections overlap: the more specific detector wins.
_PRIORITY = {"IN_AADHAAR": 9, "IN_PAN": 9, "BANK_ACCOUNT": 8, "POLICY_NUMBER": 8, "PATIENT_ID": 8, "IN_MOBILE": 7,
             "PHONE_NUMBER": 6, "EMAIL_ADDRESS": 7, "MEDICAL_CONDITION": 6, "PERSON": 5, "DATE_TIME": 4, "LOCATION": 3}


def mask(text: str) -> MaskResult:
    if not text.strip():
        return MaskResult(text)
    results = get_analyzer().analyze(text=text, language="en", entities=ENTITIES, score_threshold=0.4)

    spans = []
    for r in results:
        val = text[r.start:r.end]
        if r.entity_type in {"PERSON", "LOCATION"} and (_NOT_PERSONAL.match(val.strip()) or len(val.strip()) < 3):
            continue
        if r.entity_type == "PERSON" and r.score < 0.95 and set(re.findall(r"[a-z0-9]+", val.lower())) & _NON_NAME_WORDS:
            continue  # spaCy guess that is really a test/drug/label (labelled names score 0.95 and are kept)
        if r.entity_type == "DATE_TIME" and not _CALENDAR_DATE.search(val):
            continue  # durations ("3 days"), frequencies ("twice daily"), postcodes: not dates
        if r.entity_type in {"DATE_TIME", "PHONE_NUMBER", "PERSON", "LOCATION"} and _MONEY.match(val.strip()):
            continue  # never mask amounts
        # trim whitespace so placeholders don't swallow column spacing
        st, en = r.start, r.end
        while st < en and text[st].isspace():
            st += 1
        while en > st and text[en - 1].isspace():
            en -= 1
        # a value never crosses a line break or a column gap (2+ spaces)
        cut = re.search(r"\n|\s{2,}", text[st:en])
        if cut and r.entity_type not in {"IN_AADHAAR", "IN_MOBILE", "PHONE_NUMBER", "BANK_ACCOUNT"}:
            en = st + cut.start()
        if en > st:
            spans.append((st, en, r.entity_type, r.score))

    # Resolve overlaps: higher priority, then longer span, then higher score.
    spans.sort(key=lambda s: (-_PRIORITY.get(s[2], 0), -(s[1] - s[0]), -s[3]))
    chosen: list[tuple[int, int, str]] = []
    for st, en, et, _ in spans:
        if all(en <= c[0] or st >= c[1] for c in chosen):
            chosen.append((st, en, et))
    chosen.sort()

    vault: dict[str, str] = {}
    reverse: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    out, pos = [], 0
    for st, en, et in chosen:
        val = text[st:en]
        key = (et, val.strip().lower())
        if key not in reverse:
            counts[et] = counts.get(et, 0) + 1
            ph = f"<{et}_{counts[et]}>"
            reverse[key] = ph
            vault[ph] = val.strip()
        out.append(text[pos:st])
        out.append(reverse[key])
        pos = en
    out.append(text[pos:])
    return MaskResult("".join(out), vault, counts)


_PH = re.compile(r"<[A-Z_]+_\d+>")


def restore(value, vault: dict[str, str]):
    """Put real values back into a string produced from masked text (local only)."""
    if not isinstance(value, str):
        return value
    return _PH.sub(lambda m: vault.get(m.group(0), m.group(0)), value).strip()


def has_placeholder(value) -> bool:
    return isinstance(value, str) and bool(_PH.search(value))
