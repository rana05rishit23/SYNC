"""Unified LLM interface for language-agnostic processing and tool-calling advisor.

Consolidates LLM protocols across the system:
- Supports GroqLLM for live production calls using Groq models.
- Supports FakeLLM for deterministic, zero-network unit testing and offline demo mode.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Protocol, runtime_checkable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class LLM(Protocol):
    """Unified LLM protocol supporting structured JSON, free text, and tool calling."""

    def json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        """Return structured JSON output validated against system instructions."""
        ...

    def text(self, system: str, user: str, model: str | None = None) -> str:
        """Return free text output."""
        ...

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Send chat messages and optional tool definitions to the LLM."""
        ...


class GroqLLM:
    """Production Groq LLM client using official Groq SDK."""

    def __init__(self, api_key: str | None = None, model: str | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.api_key = api_key or self.s.groq_api_key
        self.model = model or self.s.advisor_model
        self._client = None
        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Groq SDK: {e}")

    def json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Groq client not initialized (missing GROQ_API_KEY).")
        m = model or self.s.fast_model
        resp = self._client.chat.completions.create(
            model=m,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    def text(self, system: str, user: str, model: str | None = None) -> str:
        if not self._client:
            raise RuntimeError("Groq client not initialized (missing GROQ_API_KEY).")
        m = model or self.model
        resp = self._client.chat.completions.create(
            model=m,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Groq client not initialized (missing API key or offline).")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls = []
        if getattr(choice, "tool_calls", None):
            for tc in choice.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        return {
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": tool_calls,
        }


class FakeLLM:
    """Deterministic mock LLM for zero-network testing and sandbox environments."""

    def __init__(
        self,
        json_replies: list[dict[str, Any]] | None = None,
        text_replies: list[str] | None = None,
        chat_replies: list[dict[str, Any]] | None = None,
        catalogue: Any = None,
    ):
        self.json_replies = list(json_replies or [])
        self.text_replies = list(text_replies or [])
        self.chat_replies = list(chat_replies or [])
        self.cat = catalogue

    def json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        if self.json_replies:
            return self.json_replies.pop(0)
        u = user.lower()
        if "detect" in system.lower() or "bcp-47" in system.lower():
            if any(w in u for w in ["hai", "chahiye", "mera", "meri", "hum", "hamare", "hamari", "hain", "kare", "दुकान", "किराना", "हैं"]):
                if any(w in u for w in ["आहे", "माझे", "आमचे", "आमच्या", "कारखाना", "कामगार", "शाखा", "दुकाने"]):
                    return {"language": "mr", "romanised": False, "confidence": 0.98}
                if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in user):
                    return {"language": "hi", "romanised": False, "confidence": 0.98}
                return {"language": "hi-Latn", "romanised": True, "confidence": 0.95}
            if any(w in u for w in ["आहे", "माझे", "आमचे", "आमच्या", "कारखाना", "कामगार", "शाखा", "दुकाने"]):
                return {"language": "mr", "romanised": False, "confidence": 0.98}
            if any(w in u for w in ["illai", "vendum", "ullathu", "சென்னை", "உள்ளது"]):
                return {"language": "ta", "romanised": False, "confidence": 0.98}
            if any(w in u for w in ["আছেন", "আছে", "দোকান", "মিষ্টি"]):
                return {"language": "bn", "romanised": False, "confidence": 0.98}
            return {"language": "en", "romanised": False, "confidence": 0.99}

        if "extract" in system.lower() or "businessprofile" in system.lower():
            from .templates import detect_segment
            seg = detect_segment(user)
            import re
            m = re.search(r"\b(\d+)\b", user)
            emp = int(m.group(1)) if m else None
            return {
                "business_type": seg,
                "employees": emp,
                "sites": 1,
                "city": "Lucknow" if "lucknow" in u else None,
                "has_gstin": True if "gst" in u else None,
                "products_of_interest": ["P01"],
                "missing_fields": [],
            }

        if "translate" in system.lower() or "ui strings" in system.lower():
            try:
                en_dict = json.loads(user)
                return {k: f"[{k}]" for k in en_dict}
            except Exception:
                return {}

        return {}

    def text(self, system: str, user: str, model: str | None = None) -> str:
        if self.text_replies:
            return self.text_replies.pop(0)
        return "Aapka plan 1,119 rupaye mahine ka hai, 12 mahine ke liye."

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.chat_replies:
            return self.chat_replies.pop(0)

        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1]["content"].lower() if user_msgs else ""
        tool_msgs = [m for m in messages if m.get("role") == "tool"]

        if tool_msgs:
            last_tool = tool_msgs[-1]
            try:
                data = json.loads(last_tool["content"])
            except Exception:
                data = {}

            if "quote_12m" in data:
                q12 = data["quote_12m"]
                q24 = data.get("quote_24m", {})
                saving = q24.get("saving_vs_baseline_monthly", 0)
                msg = (
                    f"I have put together an optimal, verified package for your business! "
                    f"For a 12-month plan, it is INR {q12['monthly_total']:,}/month with INR {q12['one_time_total']:,} setup. "
                    f"If you choose 24 months, your monthly rate drops to INR {q24.get('monthly_total', q12['monthly_total']):,}/month "
                    f"(saving INR {saving:,}/month!). "
                    f"Installation will take approximately {q12['install_days']} business days. "
                    f"Would you like to toggle any add-on or proceed directly to paperless verification?"
                )
                return {"role": "assistant", "content": msg, "tool_calls": []}

            if "template" in data:
                tmpl = data["template"]
                return {
                    "role": "assistant",
                    "content": (
                        f"Great! For {tmpl['name']}, our recommended setup includes high-speed business broadband, "
                        f"management postpaid SIMs, and billing POS. "
                        f"How many team members work at your location, and do you have an active GSTIN?"
                    ),
                    "tool_calls": [],
                }

        if any(w in last_user for w in ["quote", "proceed", "ready", "confirm"]) or any(w in last_user for w in ["staff", "employee", "people"]):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_quote_bundle_1",
                    "type": "function",
                    "function": {
                        "name": "tool_propose_and_quote_bundle",
                        "arguments": json.dumps({
                            "segment": "micro",
                            "city_tier": 2,
                            "employees": 3,
                            "sites": 1,
                            "term_months": 12,
                        }),
                    },
                }],
            }

        if any(w in last_user for w in [
            "kirana", "bakery", "shop", "clinic", "office", "store", "factory", "chain", "branches",
            "किराना", "दुकान", "शाखा", "कारखाना", "தொழிற்சாலை", "உற்பத்தி"
        ]):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_recommend_tmpl_1",
                    "type": "function",
                    "function": {
                        "name": "tool_detect_and_recommend_template",
                        "arguments": json.dumps({"business_type": last_user, "city": "auto", "city_tier": 2}),
                    },
                }],
            }

        return {
            "role": "assistant",
            "content": (
                "Namaste! Welcome to Accel Business. I am your Digital Onboarding Advisor. "
                "I can help get your business connected with high-speed fibre broadband, mobile lines, "
                "and digital tools in under 5 minutes. "
                "To get started, what kind of business do you run?"
            ),
            "tool_calls": [],
        }
