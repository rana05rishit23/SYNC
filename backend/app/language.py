"""Two-layer language detection: Unicode script check (Layer A) + fast LLM classification (Layer B).

Implements Section 3.1 of the Language-Agnostic Advisor spec:
- Layer A: Free, instant Unicode block check. Settles language for scripts unique to a single language.
- Layer B: Fast LLM (openai/gpt-oss-20b) for ambiguous scripts (Devanagari, Latin, Arabic).
- Returns BCP-47 language tag, script, romanised boolean, and confidence score.
"""
from __future__ import annotations
from dataclasses import dataclass

from .llm import FakeLLM, LLM

SCRIPTS: dict[str, tuple[int, int]] = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),
    "Latin": (0x0041, 0x024F),
}

# Scripts that definitively map to a single language without needing an LLM call
SCRIPT_TO_LANG: dict[str, str] = {
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Gurmukhi": "pa",
    "Odia": "or",
    "Bengali": "bn",
}


@dataclass(frozen=True)
class Detection:
    language: str        # BCP-47 tag, e.g. 'hi', 'mr', 'ta', 'en', 'hi-Latn'
    script: str | None   # Dominant script name
    romanised: bool      # True if an Indic language written in Latin script
    confidence: float    # 0.0 to 1.0


def dominant_script(text: str) -> str | None:
    """Identify which Unicode block most characters in the text belong to."""
    counts: dict[str, int] = {}
    for ch in text:
        code = ord(ch)
        for name, (lo, hi) in SCRIPTS.items():
            if lo <= code <= hi:
                counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else None


DETECT_SYSTEM = (
    "You are a language identification model for Indian business text. "
    "Classify the input into its standard BCP-47 language code (e.g. 'en', 'hi', 'hi-Latn', 'mr', 'ta', 'te', 'bn', 'ur', 'gu', 'kn', 'pa'). "
    "Return JSON: {\"language\": \"<code-or-hi-Latn>\", \"romanised\": <true/false>, \"confidence\": <float 0.0-1.0>}. "
    "Do not guess: if ambiguous, return your best estimate with appropriate confidence."
)


def detect(text: str, llm: LLM | None = None) -> Detection:
    """Two-layer language detection algorithm:
    1. Check dominant Unicode script (free & instant).
    2. If script uniquely settles language (e.g. Tamil, Telugu, Gujarati), return immediately.
    3. If script is ambiguous (Devanagari: Hindi vs Marathi; Latin: English vs Hinglish), call Layer B LLM.
    """
    if not text or not text.strip():
        return Detection(language="en", script=None, romanised=False, confidence=1.0)

    script = dominant_script(text)

    # Layer A: Script settles uniquely
    if script in SCRIPT_TO_LANG:
        return Detection(
            language=SCRIPT_TO_LANG[script],
            script=script,
            romanised=False,
            confidence=0.99,
        )

    # Layer B: Disambiguation with LLM
    client = llm or FakeLLM()
    try:
        out = client.json(DETECT_SYSTEM, text)
        lang = str(out.get("language", "en")).strip()
        romanised = bool(out.get("romanised", False))
        conf = float(out.get("confidence", 0.9))

        # Sanity check: if Devanagari script was detected, language shouldn't be 'en'
        if script == "Devanagari" and lang == "en":
            lang = "hi"
            romanised = False

        if script == "Latin" and lang not in ("en",) and not lang.endswith("-Latn"):
            # Mark code-mixed / romanised
            if romanised:
                lang = f"{lang}-Latn"

        return Detection(
            language=lang,
            script=script,
            romanised=romanised,
            confidence=conf,
        )
    except Exception:
        # Fallback to English on error
        return Detection(
            language="en" if script == "Latin" else "hi",
            script=script,
            romanised=False,
            confidence=0.6,
        )
