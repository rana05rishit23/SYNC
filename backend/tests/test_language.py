import pytest
from backend.app.language import dominant_script, detect, Detection
from backend.app.llm import FakeLLM


def test_dominant_script_detection():
    # Tamil script
    assert dominant_script("வணக்கம் சென்னை") == "Tamil"
    # Devanagari script
    assert dominant_script("नमस्ते लखनऊ") == "Devanagari"
    # Telugu script
    assert dominant_script("నమస్కారం హైదరాబాద్") == "Telugu"
    # Bengali script
    assert dominant_script("নমস্কার কলকাতা") == "Bengali"
    # Latin script
    assert dominant_script("Hello I need broadband") == "Latin"


def test_layer_a_settles_unique_scripts_without_llm():
    # Tamil settled directly by Layer A
    res_ta = detect("சென்னையில் எனக்கு ஒரு மருத்துவமனை உள்ளது")
    assert res_ta.language == "ta"
    assert res_ta.script == "Tamil"
    assert not res_ta.romanised
    assert res_ta.confidence >= 0.95

    # Telugu settled directly
    res_te = detect("మాకు ఒక కిరాణా దుకాణం ఉంది")
    assert res_te.language == "te"
    assert res_te.script == "Telugu"


def test_layer_b_disambiguates_devanagari_and_latin():
    # Mock LLM for Devanagari Marathi
    fake = FakeLLM(json_replies=[{"language": "mr", "romanised": False, "confidence": 0.98}])
    res = detect("पुण्यात माझे 3 दुकाने आहेत", llm=fake)
    assert res.language == "mr"
    assert res.script == "Devanagari"
    assert not res.romanised

    # Mock LLM for Hinglish (Latin script)
    fake_latin = FakeLLM(json_replies=[{"language": "hi-Latn", "romanised": True, "confidence": 0.94}])
    res_hi = detect("Mujhe kirana shop ke liye fast internet chahiye", llm=fake_latin)
    assert res_hi.language == "hi-Latn"
    assert res_hi.romanised is True
