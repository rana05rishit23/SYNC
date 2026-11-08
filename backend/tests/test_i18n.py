import pytest
from pathlib import Path
from backend.app.i18n import I18nCache, get_translations, load_en_strings
from backend.app.llm import FakeLLM


def test_load_en_strings():
    en = load_en_strings()
    assert len(en) >= 30
    assert "hero.heading" in en
    assert "advisor.business_type" in en


def test_i18n_returns_en_directly():
    res = get_translations("en")
    assert res["hero.heading"] == "Go digital in one sitting"


def test_i18n_translates_and_caches(tmp_path: Path):
    db_file = tmp_path / "test_i18n.db"
    cache = I18nCache(db_path=db_file)

    fake = FakeLLM(json_replies=[{
        "hero.heading": "एक बैठक में अपना बिज़नेस डिजिटल बनाएँ",
        "quote.setup_text": "+ {setup} एकमुश्त शुल्क · पहला भुगतान {first_payment}",
    }])

    res = get_translations("hi", llm=fake, cache=cache)
    assert res["hero.heading"] == "एक बैठक में अपना बिज़नेस डिजिटल बनाएँ"
    assert "{setup}" in res["quote.setup_text"]

    # Second call should fetch directly from cache without calling LLM
    empty_fake = FakeLLM(json_replies=[])
    res_cached = get_translations("hi", llm=empty_fake, cache=cache)
    assert res_cached["hero.heading"] == "एक बैठक में अपना बिज़नेस डिजिटल बनाएँ"


def test_i18n_placeholder_corruption_falls_back_to_english(tmp_path: Path):
    db_file = tmp_path / "test_i18n_err.db"
    cache = I18nCache(db_path=db_file)

    # Corrupt placeholder (dropped {first_payment})
    fake = FakeLLM(json_replies=[{
        "quote.setup_text": "+ {setup} one time fee (corrupted placeholder)",
    }])

    res = get_translations("hi", llm=fake, cache=cache)
    # Must fallback to the English template because placeholder {first_payment} was dropped
    assert "{first_payment}" in res["quote.setup_text"]
