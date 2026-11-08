"""GenAI Advisor module with Groq tool-calling loop and deterministic validation gate.

Implements ADR-001 (Deterministic core, LLM at edges) and ADR-006 (Groq as provider):
- LLM only converses and proposes.
- Rules, eligibility, pricing and registry checks are executed via deterministic tools.
- Every bundle proposal passes validate_bundle() and build_quote() before customer presentation.
- Client abstraction allows running against Groq API in production, or a Mock client
  in sandbox/tests without network calls or API keys.
"""
from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .catalogue import BusinessProfile, Issue, Product, load_catalogue, validate_bundle
from .pricing import Quote, available_addons, build_quote, term_options, unit_price
from .registries import lookup_gstin, verify_gstin_with_repair
from .templates import TEMPLATES, detect_segment, get_segment_template, propose_bundle

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 3


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AdvisorContext:
    session_id: str
    preferred_language: str = "English"  # 'English' | 'Hindi' | 'Hinglish' | 'Marathi' | etc.
    business_type: str | None = None
    segment: str = "micro"               # 'micro' | 'prof' | 'multi' | 'mfg'
    city: str | None = None
    city_tier: int = 2
    employees: int = 3
    sites: int = 1
    has_gstin: bool | None = None
    gstin: str | None = None
    gstin_status: str | None = None      # 'Active' | 'Cancelled' | 'Suspended' | None
    current_bundle: dict[str, int] = field(default_factory=dict)
    term_months: int = 12
    stage: str = "greeting"              # 'greeting' | 'profiling' | 'quoted' | 'ready_for_kyc'
    turn_count: int = 0
    history: list[dict[str, str]] = field(default_factory=list)

    def to_profile(self) -> BusinessProfile:
        return BusinessProfile(
            city_tier=self.city_tier,
            employees=self.employees,
            sites=self.sites,
            gstin_status=self.gstin_status,
        )


@dataclass(frozen=True)
class QuoteSummary:
    term_months: int
    monthly_total: int
    one_time_total: int
    first_payment: int
    contract_value: int
    saving_vs_baseline_monthly: int
    install_days: int
    line_items: list[dict[str, Any]]


@dataclass
class AdvisorTurnResult:
    message: str
    quick_chips: list[str]
    context: AdvisorContext
    bundle: dict[str, int] | None = None
    quote: QuoteSummary | None = None
    quote_24m: QuoteSummary | None = None
    available_addons: list[dict[str, Any]] | None = None
    validation_issues: list[str] = field(default_factory=list)
    next_step: str = "continue_chat"
    grounding: str = "ok"          # ok | retried | fallback (see guard.py)


# ---------------------------------------------------------------------------
# LLM Protocol & Unified Client (from backend.app.llm)
# ---------------------------------------------------------------------------

from .llm import LLM, GroqLLM, FakeLLM
from .language import language_name
from .guard import (MAX_EMPLOYEES, MAX_QTY, MAX_SITES, allowed_numbers, bounded_int,
                    fallback_message, ungrounded_numbers)

# Unified aliases for backwards compatibility
LLMClient = LLM
GroqLLMClient = GroqLLM
MockAdvisorLLMClient = FakeLLM


# ---------------------------------------------------------------------------
# Tool Declarations (Groq / OpenAI Format)
# ---------------------------------------------------------------------------

ADVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_detect_and_recommend_template",
            "description": "Classify SMB business type into a segment and return recommended starter template, quick chips, and curated add-ons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "business_type": {"type": "string", "description": "Description of business, e.g. 'dental clinic', 'kirana store'"},
                    "city": {"type": "string", "description": "City name if known"},
                    "city_tier": {"type": "integer", "enum": [1, 2], "description": "1 for Metro/Tier-1, 2 for Tier-2"},
                },
                "required": ["business_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_eligible_products",
            "description": "Query product catalogue filtered strictly by customer business profile constraints (city tier, employees, sites, GSTIN).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_tier": {"type": "integer", "enum": [1, 2]},
                    "employees": {"type": "integer"},
                    "sites": {"type": "integer"},
                    "gstin_status": {"type": "string", "enum": ["Active", "Cancelled", "Suspended", "None"]},
                },
                "required": ["city_tier", "employees", "sites"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_check_gstin",
            "description": "Verify GSTIN against mock registry with automatic OCR error repair. Returns legal name and Active/Cancelled/Suspended status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gstin": {"type": "string", "description": "15-character GSTIN"},
                },
                "required": ["gstin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_propose_and_quote_bundle",
            "description": "Validate and price a package deterministically. Computes transparent 12m and 24m commitments, savings, and add-ons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {"type": "string", "enum": ["micro", "prof", "multi", "mfg"]},
                    "city_tier": {"type": "integer", "enum": [1, 2]},
                    "employees": {"type": "integer"},
                    "sites": {"type": "integer"},
                    "term_months": {"type": "integer", "enum": [12, 24], "default": 12},
                    "gstin_status": {"type": "string", "enum": ["Active", "Cancelled", "Suspended", "None"]},
                    "custom_items": {"type": "object", "description": "Optional custom {product_id: quantity}"},
                },
                "required": ["segment", "city_tier", "employees", "sites"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Execution Engine
# ---------------------------------------------------------------------------

def _profile_args(args: dict[str, Any], context: AdvisorContext) -> tuple[int, int, int, str | None]:
    """Slot values proposed by the LLM, bounded. GST status is never taken from the LLM (ADR-014):
    only a registry check (tool_check_gstin) sets it, so a model cannot unlock GST-gated products."""
    return (
        bounded_int(args.get("city_tier"), context.city_tier, 1, 2),
        bounded_int(args.get("employees"), context.employees, 1, MAX_EMPLOYEES),
        bounded_int(args.get("sites"), context.sites, 1, MAX_SITES),
        context.gstin_status,
    )


def execute_tool(
    name: str,
    args: dict[str, Any],
    context: AdvisorContext,
    catalogue: dict[str, Product] | None = None,
) -> tuple[dict[str, Any], QuoteSummary | None, QuoteSummary | None, list[dict[str, Any]] | None]:
    """Execute a deterministic core function requested by the LLM."""
    cat = catalogue or load_catalogue()
    q12_summary: QuoteSummary | None = None
    q24_summary: QuoteSummary | None = None
    addons_out: list[dict[str, Any]] | None = None

    if name == "tool_detect_and_recommend_template":
        b_type = args.get("business_type", "")
        seg = detect_segment(b_type)
        context.business_type = b_type
        context.segment = seg
        context.stage = "profiling"
        if "city_tier" in args:
            context.city_tier = bounded_int(args["city_tier"], context.city_tier, 1, 2)
        if "city" in args and args["city"] != "auto":
            context.city = args["city"]
        tmpl = get_segment_template(seg)
        result = {
            "segment": seg,
            "template": {
                "name": tmpl.name,
                "tagline": tmpl.tagline,
                "quick_chips": list(tmpl.quick_chips),
                "curated_addons": list(tmpl.curated_addons),
            },
        }
        return result, None, None, None

    elif name == "tool_get_eligible_products":
        tier, emp, sites, gst = _profile_args(args, context)
        prof = BusinessProfile(city_tier=tier, employees=emp, sites=sites, gstin_status=gst)
        prods = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "category": p.category,
                "unit": p.unit,
                "price_range": f"INR {p.price_min}-{p.price_max}",
                "setup_fee": p.setup_fee,
                "install_days": p.install_days,
            }
            for p in cat.values()
            if not validate_bundle(prof, {p.product_id: 1}, cat) or p.category != "Connectivity"
        ]
        return {"eligible_products": prods}, None, None, None

    elif name == "tool_check_gstin":
        raw = args.get("gstin", "")
        rec, notes = verify_gstin_with_repair(raw)
        if rec:
            context.gstin = rec.gstin
            context.has_gstin = True
            context.gstin_status = rec.status
            return {
                "valid": True,
                "gstin": rec.masked_gstin,
                "legal_name": rec.legal_name,
                "trade_name": rec.trade_name,
                "status": rec.status,
                "state": rec.state,
                "notes": notes,
            }, None, None, None
        else:
            return {"valid": False, "notes": notes}, None, None, None

    elif name == "tool_propose_and_quote_bundle":
        seg = args.get("segment") if args.get("segment") in ("micro", "prof", "multi", "mfg") else context.segment
        tier, emp, sites, gst = _profile_args(args, context)
        term = bounded_int(args.get("term_months"), context.term_months, 12, 24)
        term = term if term in (12, 24) else context.term_months

        context.segment = seg
        context.city_tier = tier
        context.employees = emp
        context.sites = sites
        context.gstin_status = gst
        context.term_months = term

        prof = BusinessProfile(city_tier=tier, employees=emp, sites=sites, gstin_status=gst)

        # Get items: custom or auto-proposed
        custom = args.get("custom_items")
        if custom and isinstance(custom, dict):
            items = {}
            for k, v in custom.items():
                q = bounded_int(v, 0, 1, MAX_QTY)
                if q == 0 or k not in cat:
                    return {"valid": False, "issues": [f"Unknown product or invalid quantity for {k}."]}, None, None, None
                items[str(k)] = q
        else:
            items = propose_bundle(seg, prof, catalogue=cat)

        context.current_bundle = items
        context.stage = "quoted"

        # Validate deterministically
        issues = validate_bundle(prof, items, cat)
        if issues:
            return {
                "valid": False,
                "issues": [i.message for i in issues],
            }, None, None, None

        # Build Quotes (12m and 24m)
        q12 = build_quote(prof, items, 12, cat)
        q24 = build_quote(prof, items, 24, cat)

        def _to_summary(q: Quote) -> QuoteSummary:
            lines = [
                {
                    "product_id": l.product_id,
                    "name": l.name,
                    "quantity": l.quantity,
                    "unit": l.unit,
                    "unit_price": l.unit_price,
                    "monthly": l.monthly,
                    "setup": l.setup,
                }
                for l in q.lines
            ]
            return QuoteSummary(
                term_months=q.term_months,
                monthly_total=q.monthly_total,
                one_time_total=q.one_time_total,
                first_payment=q.first_payment,
                contract_value=q.contract_value,
                saving_vs_baseline_monthly=q.saving_vs_baseline_monthly,
                install_days=q.install_days,
                line_items=lines,
            )

        q12_summary = _to_summary(q12)
        q24_summary = _to_summary(q24)

        # Curated Add-ons
        addons_list = available_addons(prof, items, cat)
        addons_out = [
            {
                "product_id": a.product_id,
                "name": a.name,
                "category": a.category,
                "unit": a.unit,
                "unit_price_12m": unit_price(a, 12),
                "unit_price_24m": unit_price(a, 24),
                "setup_fee": a.setup_fee,
            }
            for a in addons_list
        ]

        result = {
            "valid": True,
            "bundle": items,
            "quote_12m": asdict(q12_summary),
            "quote_24m": asdict(q24_summary),
            "available_addons_count": len(addons_out),
        }
        return result, q12_summary, q24_summary, addons_out

    return {"error": f"Unknown tool: {name}"}, None, None, None


# ---------------------------------------------------------------------------
# Conversational Agent Loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Accel Business Digital Onboarding Advisor for Indian MSMEs.
Your role is to converse warmly, understand the customer's business needs, and recommend an optimal connectivity and digital package.

CORE RULES:
1. Grounding: You do NOT invent pricing, discounts, installation days, or product rules.
   You MUST call tools to calculate quotes, verify GSTINs, and fetch templates.
2. Tone: Helpful, concise, respectful, transparent. Never pushy.
3. Multi-turn flow:
   - Turn 1: Warm greeting, ask business type and city.
   - Turn 2: Call tool_detect_and_recommend_template, ask for employee count and GSTIN status.
   - Turn 3: Call tool_propose_and_quote_bundle, present the verified quote with 12m and 24m term pricing clearly showing monthly savings.
4. Always highlight the transparent monthly price and one-time setup fee returned by the tools.
5. If the customer mentions a GSTIN, always verify it with tool_check_gstin.
"""


def run_advisor_turn(
    user_message: str,
    context: AdvisorContext,
    client: LLMClient | None = None,
    catalogue: dict[str, Product] | None = None,
) -> AdvisorTurnResult:
    """Run one conversational turn of the advisor agent loop."""
    cat = catalogue or load_catalogue()
    llm = client or MockAdvisorLLMClient(cat)

    context.turn_count += 1
    context.history.append({"role": "user", "content": user_message})

    # Prepare message history for LLM
    system = SYSTEM_PROMPT + (
        f"\n\nCUSTOMER LANGUAGE: {language_name(context.preferred_language)}. Reply in this language and script; if the "
        "customer mixes languages, mirror them. Always write prices and numbers with digits 0-9 and the \u20b9 symbol."
    )
    llm_messages = [{"role": "system", "content": system}]
    # Add recent history (up to last 10 messages)
    llm_messages.extend(context.history[-10:])

    response = llm.chat(llm_messages, tools=ADVISOR_TOOLS)

    final_quote_12m: QuoteSummary | None = None
    final_quote_24m: QuoteSummary | None = None
    final_addons: list[dict[str, Any]] | None = None
    validation_issues: list[str] = []
    tool_results: list[dict[str, Any]] = []

    rounds = 0
    while response.get("tool_calls") and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        tool_calls = response["tool_calls"]
        llm_messages.append({
            "role": "assistant",
            "content": response.get("content") or "",
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
            except (json.JSONDecodeError, TypeError):
                fn_args = {}
            if not isinstance(fn_args, dict):
                fn_args = {}

            tool_result, q12, q24, addons = execute_tool(fn_name, fn_args, context, cat)
            tool_results.append(tool_result)
            if q12:
                final_quote_12m = q12
                final_quote_24m = q24
                final_addons = addons
            if not tool_result.get("valid", True) and "issues" in tool_result:
                validation_issues.extend(tool_result["issues"])

            llm_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": json.dumps(tool_result),
            })
        response = llm.chat(llm_messages, tools=ADVISOR_TOOLS)
    assistant_text = (response.get("content") or "").strip()

    # Grounding gate: the text the customer reads may not contain numbers the tools did not produce.
    allowed = allowed_numbers(
        tool_results, cat, [m["content"] for m in context.history if m["role"] == "user"],
        {context.employees, context.sites, context.city_tier, context.term_months},
    )
    grounding = "ok"
    bad = ungrounded_numbers(assistant_text, allowed)
    if bad or not assistant_text:
        logger.warning("Advisor reply failed grounding (numbers=%s); retrying once", sorted(bad))
        llm_messages.append({"role": "assistant", "content": assistant_text})
        llm_messages.append({"role": "user", "content": (
            f"Your reply contained numbers that are not in the tool results: {sorted(bad)}. Rewrite it using only "
            "numbers from the tool results. Do not mention any price, discount or speed the tools did not return."
        )})
        retry = (llm.chat(llm_messages, tools=None).get("content") or "").strip()
        if retry and not ungrounded_numbers(retry, allowed):
            assistant_text, grounding = retry, "retried"
        else:
            assistant_text, grounding = fallback_message(final_quote_12m, final_quote_24m), "fallback"

    context.history.append({"role": "assistant", "content": assistant_text})

    # Determine quick chips and next step
    tmpl = get_segment_template(context.segment)
    quick_chips: list[str] = []
    if context.stage == "quoted":
        quick_chips = ["Confirm 12-Month Plan", "Switch to 24-Month Plan (Save more!)", "Add Cloud CCTV", "Proceed to KYC"]
        next_step = "proceed_to_kyc"
    elif context.stage == "profiling":
        quick_chips = list(tmpl.quick_chips[:4])
        next_step = "provide_profile"
    elif context.stage == "greeting" or context.turn_count <= 1:
        quick_chips = [
            "Kirana / Grocery Shop",
            "Clinic / Medical Office",
            "Multi-site Retail Chain",
            "Manufacturing / Factory Unit",
        ]
        next_step = "select_business_type"
    else:
        quick_chips = ["Proceed to Paperless KYC", "Modify Bundle", "Speak to Human Agent"]
        next_step = "kyc"

    return AdvisorTurnResult(
        message=assistant_text,
        quick_chips=quick_chips,
        context=context,
        bundle=context.current_bundle or None,
        quote=final_quote_12m,
        quote_24m=final_quote_24m,
        available_addons=final_addons,
        validation_issues=validation_issues,
        next_step=next_step,
        grounding=grounding,
    )
