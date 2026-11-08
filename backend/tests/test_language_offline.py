"""Offline language detection: honest about what it knows. Dev sentences are the ones the marker lists were written
around; the held-out sentences were not, and are checked against one invariant: never confidently wrong."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.advisor import AdvisorContext, run_advisor_turn
from backend.app.catalogue import load_catalogue
from backend.app.language import detect, detect_offline, language_name

CONFIDENT = 0.7

DEV = [
    ("en", "I need fast internet for my shop"),
    ("hi", "मुझे अपनी दुकान के लिए इंटरनेट चाहिए"),
    ("hi", "हमें तेज़ इंटरनेट कनेक्शन की ज़रूरत है"),
    ("mr", "मला माझ्या दुकानासाठी इंटरनेट हवे आहे"),
    ("mr", "आम्हाला वेगवान इंटरनेट पाहिजे"),
    ("ne", "मलाई मेरो पसलको लागि इन्टरनेट चाहिन्छ"),
    ("ta", "எனக்கு என் கடைக்கு இன்டர்நெட் வேண்டும்"),
    ("te", "నా దుకాణానికి ఇంటర్నెట్ కావాలి"),
    ("kn", "ನನ್ನ ಅಂಗಡಿಗೆ ಇಂಟರ್ನೆಟ್ ಬೇಕು"),
    ("ml", "എന്റെ കടയ്ക്ക് ഇന്റർനെറ്റ് വേണം"),
    ("gu", "મારી દુકાન માટે ઇન્ટરનેટ જોઈએ છે"),
    ("pa", "ਮੈਨੂੰ ਆਪਣੀ ਦੁਕਾਨ ਲਈ ਇੰਟਰਨੈੱਟ ਚਾਹੀਦਾ ਹੈ"),
    ("bn", "আমার দোকানের জন্য ইন্টারনেট লাগবে"),
    ("or", "ମୋ ଦୋକାନ ପାଇଁ ଇଣ୍ଟରନେଟ୍ ଦରକାର"),
    ("ur", "مجھے اپنی دکان کے لیے انٹرنیٹ چاہیے"),
    ("hi-Latn", "mujhe apni dukaan ke liye internet chahiye"),
    ("mr-Latn", "mala majhya dukanasathi internet pahije"),
    ("ta-Latn", "enakku en kadaikku internet vendum"),
    ("te-Latn", "naaku na shop ki internet kavali"),
    ("bn-Latn", "amar dokaner jonno internet lagbe"),
]

# Not used to write the marker lists. Some are expected to be missed; none may be missed with confidence.
HELD_OUT = [
    ("hi-Latn", "humein apni dukaan mein wifi lagwana hai"),
    ("mr-Latn", "amhi navin dukan ughadli aahe"),
    ("ta-Latn", "enga kadaiyil wifi poda mudiyuma"),
    ("te-Latn", "mee shop ki fast internet kavali"),
    ("bn-Latn", "ami amar dokane wifi lagate chai"),
    ("gu-Latn", "amari dukan mate wifi joiye"),
    ("en", "We need a leased line for our factory"),
    ("en", "internet"),
    ("hi", "इंटरनेट कनेक्शन"),
    ("mr", "आमची तीन दुकाने आहेत"),
    ("ur", "ہمیں وائی فائی چاہیے"),
    ("ta", "வைஃபை வேண்டும்"),
]


@pytest.mark.parametrize("expected,text", DEV)
def test_dev_sentences(expected, text):
    d = detect_offline(text)
    assert d.language == expected and d.confidence >= CONFIDENT


@pytest.mark.parametrize("expected,text", HELD_OUT)
def test_held_out_sentences_are_never_confidently_wrong(expected, text):
    d = detect_offline(text)
    assert d.language == expected or d.confidence < CONFIDENT, f"confidently wrong: {d}"


def test_unknown_input_asks_instead_of_guessing():
    for text in ["wifi", "ok", "internet connection"]:
        assert detect_offline(text).confidence < CONFIDENT
    assert detect_offline("इंटरनेट").confidence < CONFIDENT


def test_detect_without_an_llm_uses_the_offline_path():
    assert detect("مجھے انٹرنیٹ چاہیے").language == "ur"


def test_llm_failure_falls_back_to_offline_instead_of_a_confident_guess():
    class Broken:
        def json(self, *a, **k):
            raise RuntimeError("network down")
    d = detect("humein apni dukaan ke liye internet chahiye", Broken())
    assert d.language == "hi-Latn" and d.confidence >= CONFIDENT
    assert detect("wifi", Broken()).confidence < CONFIDENT


def test_language_names_for_the_prompt():
    assert language_name("mr") == "Marathi" and language_name("ur") == "Urdu"
    assert language_name("hi-Latn").startswith("Hinglish")
    assert language_name("te-Latn") == "Telugu written in English letters"
    assert language_name("Hindi") == "Hindi" and language_name(None) == "English"


def test_advisor_is_told_which_language_to_reply_in():
    seen = {}

    class Spy:
        def chat(self, messages, tools=None):
            seen["system"] = messages[0]["content"]
            return {"content": "ठीक आहे"}

    ctx = AdvisorContext("lang1", preferred_language="mr")
    run_advisor_turn("नमस्कार", ctx, client=Spy(), catalogue=load_catalogue())
    assert "CUSTOMER LANGUAGE: Marathi" in seen["system"]


def test_chat_endpoint_reports_detected_language_and_asks_when_unsure(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(main, "SETTINGS", dataclasses.replace(main.SETTINGS, groq_api_key=None))
    c = TestClient(main.app)
    r = c.post("/api/chat", json={"session_id": "lang-sess-1", "message": "مجھے اپنی دکان کے لیے انٹرنیٹ چاہیے"}).json()
    assert r["detected_language"] == "ur" and r["ask_language"] is False
    r = c.post("/api/chat", json={"session_id": "lang-sess-2", "message": "wifi"}).json()
    assert r["ask_language"] is True
