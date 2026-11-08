"""Unit tests for the unified LLM interface (backend.app.llm)."""
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch
import pytest

from backend.app.config import Settings
from backend.app.llm import FakeLLM, GroqLLM, LLM


def test_fake_llm_conforms_to_protocol():
    llm = FakeLLM()
    assert isinstance(llm, LLM)


def test_fake_llm_canned_replies():
    llm = FakeLLM(
        json_replies=[{"status": "ok"}],
        text_replies=["Custom text"],
        chat_replies=[{"role": "assistant", "content": "Custom chat", "tool_calls": []}],
    )
    assert llm.json("sys", "usr") == {"status": "ok"}
    assert llm.text("sys", "usr") == "Custom text"
    assert llm.chat([{"role": "user", "content": "hi"}])["content"] == "Custom chat"


def test_fake_llm_language_detection_fallbacks():
    llm = FakeLLM()
    # Hindi Devanagari
    res_hi = llm.json("detect BCP-47 language", "मेरी दुकान लखनऊ में है")
    assert res_hi["language"] == "hi"

    # Hinglish Latin
    res_latn = llm.json("detect BCP-47 language", "Mera business bahut accha hai")
    assert res_latn["language"] == "hi-Latn"

    # Marathi
    res_mr = llm.json("detect BCP-47 language", "आमचे पुण्यात दुकान आहे")
    assert res_mr["language"] == "mr"

    # Tamil
    res_ta = llm.json("detect BCP-47 language", "சென்னையில் உள்ளது")
    assert res_ta["language"] == "ta"

    # Bengali
    res_bn = llm.json("detect BCP-47 language", "মিষ্টির দোকান আছে")
    assert res_bn["language"] == "bn"

    # English default
    res_en = llm.json("detect BCP-47 language", "We run a clinic in Mumbai")
    assert res_en["language"] == "en"


def test_fake_llm_business_profile_extraction():
    llm = FakeLLM()
    res = llm.json("extract BusinessProfileDraft", "I have a kirana shop in Lucknow with 5 staff")
    assert res["business_type"] == "micro"
    assert res["employees"] == 5
    assert res["city"] == "Lucknow"


def test_fake_llm_translation_fallback():
    llm = FakeLLM()
    res = llm.json("translate ui strings to Hindi", json.dumps({"title": "Hello", "cta": "Submit"}))
    assert res == {"title": "[title]", "cta": "[cta]"}


def test_fake_llm_text_fallback():
    llm = FakeLLM()
    res = llm.text("system prompt", "user query")
    assert "rupaye" in res or "plan" in res.lower()


def test_fake_llm_chat_turns():
    llm = FakeLLM()

    # 1. Greeting
    r1 = llm.chat([{"role": "user", "content": "Hello"}])
    assert "Namaste" in r1["content"]
    assert len(r1["tool_calls"]) == 0

    # 2. Segment detection tool call
    r2 = llm.chat([{"role": "user", "content": "I run a kirana shop in Jaipur"}])
    assert len(r2["tool_calls"]) == 1
    assert r2["tool_calls"][0]["function"]["name"] == "tool_detect_and_recommend_template"

    # 3. Quoting tool call
    r3 = llm.chat([{"role": "user", "content": "Please give me a quote for 4 employees"}])
    assert len(r3["tool_calls"]) == 1
    assert r3["tool_calls"][0]["function"]["name"] == "tool_propose_and_quote_bundle"

    # 4. Tool response formatting (quote_12m)
    tool_msg = {
        "role": "tool",
        "content": json.dumps({
            "quote_12m": {"monthly_total": 1499, "one_time_total": 500, "install_days": 3},
            "quote_24m": {"monthly_total": 1299, "saving_vs_baseline_monthly": 200},
        }),
    }
    r4 = llm.chat([{"role": "user", "content": "confirm"}, tool_msg])
    assert "1,499" in r4["content"]
    assert "saving INR 200" in r4["content"]

    # 5. Tool response formatting (template)
    tmpl_msg = {
        "role": "tool",
        "content": json.dumps({
            "template": {"name": "Micro Enterprise Essentials"},
        }),
    }
    r5 = llm.chat([{"role": "user", "content": "shop"}, tmpl_msg])
    assert "Micro Enterprise Essentials" in r5["content"]


def test_groq_llm_raises_when_no_api_key():
    settings = Settings(groq_api_key=None, advisor_model="m", fast_model="m", stt_model="m")
    client = GroqLLM(settings=settings)
    with pytest.raises(RuntimeError, match="Groq client not initialized"):
        client.json("sys", "usr")
    with pytest.raises(RuntimeError, match="Groq client not initialized"):
        client.text("sys", "usr")
    with pytest.raises(RuntimeError, match="Groq client not initialized"):
        client.chat([{"role": "user", "content": "hi"}])


def test_groq_llm_mocked_calls():
    settings = Settings(groq_api_key="mock_key", advisor_model="m-adv", fast_model="m-fast", stt_model="m")
    client = GroqLLM(settings=settings)

    mock_sdk = MagicMock()
    client._client = mock_sdk

    # Test json()
    mock_resp_json = MagicMock()
    mock_resp_json.choices = [MagicMock(message=MagicMock(content='{"detected": "hi"}'))]
    mock_sdk.chat.completions.create.return_value = mock_resp_json

    out_json = client.json("system instruction", "user message")
    assert out_json == {"detected": "hi"}
    mock_sdk.chat.completions.create.assert_called_with(
        model="m-fast",
        temperature=0,
        messages=[{"role": "system", "content": "system instruction"}, {"role": "user", "content": "user message"}],
        response_format={"type": "json_object"},
    )

    # Test text()
    mock_resp_text = MagicMock()
    mock_resp_text.choices = [MagicMock(message=MagicMock(content="Sample response"))]
    mock_sdk.chat.completions.create.return_value = mock_resp_text

    out_text = client.text("sys", "usr")
    assert out_text == "Sample response"

    # Test chat() with tool calls
    mock_tool_call = MagicMock()
    mock_tool_call.id = "tc_1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = "{}"

    mock_resp_chat = MagicMock()
    mock_resp_chat.choices = [MagicMock(message=MagicMock(content="Calling tool", tool_calls=[mock_tool_call]))]
    mock_sdk.chat.completions.create.return_value = mock_resp_chat

    out_chat = client.chat([{"role": "user", "content": "hello"}], tools=[{"type": "function"}])
    assert out_chat["role"] == "assistant"
    assert len(out_chat["tool_calls"]) == 1
    assert out_chat["tool_calls"][0]["function"]["name"] == "test_tool"
