"""Language detection module supporting Indian scripts and multilingual input.

Implements Section 3.1 of the Language-Agnostic Advisor spec:
- Layer A: Fast, deterministic Unicode script detection (Devanagari, Tamil, Telugu, etc.).
- Layer B: Disambiguation with LLM for shared scripts (Devanagari, Latin).
- Returns BCP-47 language tag, script, romanised boolean, and confidence score.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass

from .llm import LLM

logger = logging.getLogger(__name__)

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
    "Latin": (0x0041, 0x007A),
}

# Unique script mappings (Layer A)
SCRIPT_TO_LANG: dict[str, str] = {
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Gurmukhi": "pa",
    "Odia": "or",
}

DETECT_SYSTEM = (
    "You are a language identification service for Indian SMB telecom intake. "
    "Identify the BCP-47 tag for the customer's text. Return JSON with keys: "
    "'language' (e.g. 'hi', 'mr', 'ta', 'te', 'bn', 'en', 'hi-Latn' for Hinglish), "
    "'romanised' (boolean, true if Indic language written in Latin script), "
    "'confidence' (float between 0.0 and 1.0)."
)


@dataclass(frozen=True)
class Detection:
    language: str       # BCP-47 tag ('hi', 'mr', 'ta', 'hi-Latn', 'en', etc.)
    script: str         # 'Devanagari', 'Tamil', 'Latin', etc.
    romanised: bool     # True if Indic written in Latin
    confidence: float   # 0.0 to 1.0


def dominant_script(text: str) -> str:
    """Count non-ASCII code points per Unicode script block; return the dominant one."""
    counts: dict[str, int] = {name: 0 for name in SCRIPTS}
    latin_count = 0

    for ch in text:
        cp = ord(ch)
        matched = False
        for name, (start, end) in SCRIPTS.items():
            if name == "Latin":
                continue
            if start <= cp <= end:
                counts[name] += 1
                matched = True
                break
        if not matched and ((0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)):
            latin_count += 1

    counts["Latin"] = latin_count
    best_script, best_count = max(counts.items(), key=lambda kv: kv[1])
    return best_script if best_count > 0 else "Latin"


def detect(text: str, llm: LLM | None = None) -> Detection:
    """Run two-layer detection: Layer A (deterministic script) -> Layer B (LLM disambiguation)."""
    if not text or not text.strip():
        return Detection(language="en", script="Latin", romanised=False, confidence=1.0)

    script = dominant_script(text)

    # Layer A: Settled immediately if unique script
    if script in SCRIPT_TO_LANG:
        return Detection(
            language=SCRIPT_TO_LANG[script],
            script=script,
            romanised=False,
            confidence=0.99,
        )

    # No LLM configured: use the offline heuristics, which report low confidence when unsure.
    if llm is None:
        return detect_offline(text)

    # Layer B: Disambiguation with LLM
    client = llm
    try:
        out = client.json(DETECT_SYSTEM, text)
        lang = str(out.get("language", "en")).strip()
        romanised = bool(out.get("romanised", False))
        conf = float(out.get("confidence", 0.85))

        # Sanity check: if Devanagari script was detected, it cannot be English
        if script == "Devanagari" and lang == "en":
            lang = "hi"
            romanised = False

        # If Latin script with romanised flag, format as 'xx-Latn'
        if script == "Latin" and romanised and not lang.endswith("-Latn"):
            lang = f"{lang}-Latn"

        return Detection(
            language=lang,
            script=script,
            romanised=romanised,
            confidence=conf,
        )
    except Exception as exc:  # network or parsing failure: say so and fall back to the offline guess
        logger.warning("Language LLM failed (%s); using offline heuristics", exc)
        return detect_offline(text)


# ---------------------------------------------------------------------------------------------------------
# Offline fallback (ADR-015). Used when no LLM is configured or the LLM call fails. It is deliberately small
# and conservative: it may be wrong, but it must never be *confidently* wrong. Confidence below 0.7 means
# "ask the owner which language they prefer". The real detector is the LLM (Layer B).
# ---------------------------------------------------------------------------------------------------------
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "ne": "Nepali", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi", "bn": "Bengali",
    "or": "Odia", "ur": "Urdu", "as": "Assamese",
}


def language_name(tag: str | None) -> str:
    """BCP-47 tag (or an already-human name) -> the name used in the advisor prompt."""
    tag = tag or "en"
    base, _, suffix = tag.partition("-")
    if tag == "hi-Latn":
        return "Hinglish (Hindi written in English letters)"
    name = LANGUAGE_NAMES.get(base)
    if name is None:
        return tag
    return f"{name} written in English letters" if suffix == "Latn" else name


_DEVANAGARI = {
    "hi": {"है", "हैं", "मुझे", "हमें", "चाहिए", "का", "की", "के", "को", "में", "नहीं", "और", "से", "हम", "मेरा", "मेरी", "क्या", "लिए", "कृपया"},
    "mr": {"आहे", "आहेत", "आम्ही", "आम्हाला", "मला", "माझे", "माझ्या", "माझा", "माझी", "पाहिजे", "हवे", "हवा", "नाही", "साठी", "आणि", "मध्ये", "तुमच्या", "काय"},
    "ne": {"छ", "छन्", "हो", "मलाई", "हामी", "हाम्रो", "चाहिन्छ", "लागि", "मेरो", "तपाईं", "तपाईंको", "हुन्छ"},
}
_ROMANISED = {
    "hi": {"chahiye", "chahie", "hai", "hain", "mujhe", "mera", "meri", "mere", "hum", "hamara", "hamare", "kya", "nahi", "nahin", "dukaan", "aap", "aapka", "ke", "liye", "karna", "kare", "bhi", "aur", "mein"},
    "mr": {"pahije", "pahijet", "aahe", "aahet", "mala", "majha", "majhi", "majhya", "amhala", "amchya", "sathi", "tumchya", "madhye"},
    "ta": {"vendum", "venum", "enakku", "illai", "ungal", "ungalukku", "naan", "kadai", "kadaikku", "irukku", "sollunga"},
    "te": {"kavali", "naaku", "naku", "undi", "ledu", "meeru", "cheppandi", "kosam"},
    "bn": {"lagbe", "amar", "ami", "apni", "apnar", "dokan", "dokaner", "jonno", "achhe", "nei", "koto", "chai"},
    "gu": {"joie", "joiye", "chhe", "mane", "maru", "mari", "mate"},
    "pa": {"chahida", "haiga", "mainu", "tuhada", "tuhadi", "sadi"},
    "kn": {"beku", "nanage", "ide", "nimma", "nanna", "bekagide"},
    "ml": {"venam", "enikku", "undu", "ente", "ningalude", "vendi"},
}
_ENGLISH = {"the", "a", "an", "for", "my", "our", "we", "i", "need", "want", "is", "are", "and", "to", "of", "in", "with", "have", "please", "can", "you", "it", "this", "that"}


def _best(scores: dict[str, int]) -> tuple[str | None, int, int]:
    """(winner, its score, margin over the runner-up); winner None if nothing scored."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] == 0:
        return None, 0, 0
    second = ranked[1][1] if len(ranked) > 1 else 0
    return ranked[0][0], ranked[0][1], ranked[0][1] - second


def detect_offline(text: str) -> Detection:
    """Heuristic detection with honest confidence. Never returns >= 0.7 without evidence."""
    script = dominant_script(text)
    if script in SCRIPT_TO_LANG:
        return Detection(SCRIPT_TO_LANG[script], script, False, 0.99)
    if script == "Arabic":  # in India almost always Urdu; Sindhi/Kashmiri also use it
        return Detection("ur", script, False, 0.8)
    if script == "Devanagari":
        tokens = set(re.findall(r"[\u0900-\u097F]+", text))
        lang, score, margin = _best({k: len(tokens & v) for k, v in _DEVANAGARI.items()})
        if lang is None:
            return Detection("hi", script, False, 0.5)  # a guess, flagged as one
        return Detection(lang, script, False, 0.9 if margin >= 2 else 0.75 if margin == 1 else 0.5)
    words = re.findall(r"[a-z']+", text.lower())
    tokens = set(words)
    lang, score, margin = _best({k: len(tokens & v) for k, v in _ROMANISED.items()})
    en = len(tokens & _ENGLISH)
    if lang is not None and score >= en:
        conf = 0.85 if score >= 2 and margin >= 1 else 0.6
        return Detection(f"{lang}-Latn", script, True, conf)
    return Detection("en", script, False, 0.9 if en >= 2 else 0.75 if en == 1 else 0.5)
