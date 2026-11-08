"""Interface localization cache: translate once, cache forever.

Implements Section 3.4 of the Language-Agnostic Advisor spec:
- Frontend labels live in a single English file: locales/en.json (the only source of truth).
- Translations are cached in SQLite keyed by (lang, key, source_hash).
- Missing keys are translated in a single batched LLM call with glossary enforcement.
- Placeholders such as {setup}, {first_payment}, {order} are strictly verified before caching.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from .llm import FakeLLM, LLM

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"
EN_PATH = LOCALES_DIR / "en.json"
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "i18n_cache.db"

_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z0-9_]+\}")


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def load_en_strings() -> dict[str, str]:
    if not EN_PATH.exists():
        return {}
    with open(EN_PATH, encoding="utf-8") as f:
        return json.load(f)


class I18nCache:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS i18n_cache (
                    lang TEXT,
                    key TEXT,
                    source_hash TEXT,
                    text TEXT,
                    status TEXT DEFAULT 'machine',
                    created_at TEXT,
                    PRIMARY KEY (lang, key, source_hash)
                );
            """)
            conn.commit()

    def get_cached_strings(self, lang: str, en_strings: dict[str, str]) -> dict[str, str]:
        if not en_strings:
            return {}
        out: dict[str, str] = {}
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT key, source_hash, text FROM i18n_cache WHERE lang = ?", (lang,))
            for key, shash, val in cur.fetchall():
                if key in en_strings and _hash_text(en_strings[key]) == shash:
                    out[key] = val
        return out

    def save_cached_strings(self, lang: str, en_strings: dict[str, str], new_translations: dict[str, str]):
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rows = []
        for k, v in new_translations.items():
            if k in en_strings:
                shash = _hash_text(en_strings[k])
                rows.append((lang, k, shash, v, "machine", now))

        if rows:
            with self._get_conn() as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO i18n_cache (lang, key, source_hash, text, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, rows)
                conn.commit()


TRANSLATION_SYSTEM = (
    "You are an interface translator for an Indian SMB business app. "
    "Translate the JSON string values from English into the requested language ({target_language}). "
    "Return a JSON object with the EXACT SAME KEYS.\n"
    "CRITICAL RULES:\n"
    "1. Keep placeholders such as {setup}, {first_payment}, {savings}, {price}, {term}, {order}, {activation_date} EXACTLY unchanged.\n"
    "2. Do NOT translate brand/standard terms: 'Accel Business', 'GSTIN', 'Udyam', 'KYC', 'UPI', 'WhatsApp'.\n"
    "3. Use plain, warm, everyday vocabulary suitable for neighbourhood shop and clinic owners.\n"
    "4. Return strictly valid JSON."
)


def _check_placeholders(orig: str, trans: str) -> bool:
    orig_ph = set(_PLACEHOLDER_RE.findall(orig))
    trans_ph = set(_PLACEHOLDER_RE.findall(trans))
    return orig_ph == trans_ph


def get_translations(
    lang: str,
    llm: LLM | None = None,
    cache: I18nCache | None = None,
) -> dict[str, str]:
    """Return UI translation dictionary for target language with instant caching."""
    en_dict = load_en_strings()
    clean_lang = (lang or "en").lower().split("-")[0]  # canonical 2-letter code

    if clean_lang == "en":
        return en_dict

    i18n_store = cache or I18nCache()
    cached = i18n_store.get_cached_strings(clean_lang, en_dict)

    # Check for missing keys
    missing_keys = {k: en_dict[k] for k in en_dict if k not in cached}
    if not missing_keys:
        return cached

    # Call LLM for missing keys in one batched request
    client = llm or FakeLLM()
    system_prompt = TRANSLATION_SYSTEM.replace("{target_language}", clean_lang)
    user_payload = json.dumps(missing_keys, ensure_ascii=False)

    try:
        translated_json = client.json(system_prompt, user_payload)
    except Exception as e:
        logger.warning(f"Translation LLM call failed for {clean_lang}: {e}")
        translated_json = {}

    to_save: dict[str, str] = {}
    result = dict(cached)

    for k, en_val in missing_keys.items():
        trans_val = str(translated_json.get(k, "")).strip()
        # Verify placeholders survived
        if trans_val and _check_placeholders(en_val, trans_val):
            result[k] = trans_val
            to_save[k] = trans_val
        else:
            # Fallback to English for this key
            result[k] = en_val

    if to_save:
        i18n_store.save_cached_strings(clean_lang, en_dict, to_save)

    return result
