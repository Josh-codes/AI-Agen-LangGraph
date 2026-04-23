"""RAG utilities for retrieving AutoStream product knowledge."""

from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

PRICING_KEYWORDS = {
    "price",
    "pricing",
    "cost",
    "plan",
    "plans",
    "subscription",
    "monthly",
    "yearly",
    "usd",
    "dollar",
    "how much",
}
FEATURE_KEYWORDS = {
    "feature",
    "features",
    "include",
    "included",
    "resolution",
    "captions",
    "videos",
    "support",
    "4k",
    "720p",
}
POLICY_KEYWORDS = {
    "refund",
    "policy",
    "policies",
    "cancel",
    "cancellation",
    "terms",
    "support",
    "24/7",
}


def load_knowledge_base() -> Dict[str, Any]:
    """Load the local JSON knowledge base from disk."""
    try:
        with KB_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        logger.error("knowledge_base.json not found at %s", KB_PATH)
        raise FileNotFoundError("knowledge_base.json is missing") from exc
    except json.JSONDecodeError as exc:
        logger.error("knowledge_base.json contains invalid JSON")
        raise ValueError("knowledge_base.json is invalid") from exc


def _match_sections(query: str) -> List[str]:
    """Return relevant knowledge sections using keyword heuristics."""
    normalized = query.lower()
    sections: List[str] = []

    if any(keyword in normalized for keyword in PRICING_KEYWORDS):
        sections.append("pricing")

    if any(keyword in normalized for keyword in FEATURE_KEYWORDS):
        sections.append("features")

    if any(keyword in normalized for keyword in POLICY_KEYWORDS):
        sections.append("policies")

    if "pro" in normalized and "pricing" not in sections:
        sections.append("pricing")
    if "basic" in normalized and "pricing" not in sections:
        sections.append("pricing")

    if not sections:
        sections = ["pricing", "features", "policies"]

    return sections


def format_context_for_prompt(context: Dict[str, Any]) -> str:
    """Convert selected KB content into concise prompt-ready text."""
    lines: List[str] = []

    pricing = context.get("pricing")
    if pricing:
        lines.append("Pricing:")
        basic = pricing.get("basic", {})
        pro = pricing.get("pro", {})

        if basic:
            lines.append(
                "- Basic: $"
                f"{basic.get('price')}/{basic.get('billing')} | "
                f"videos: {basic.get('videos_per_month')} | "
                f"resolution: {basic.get('max_resolution')} | "
                f"AI captions: {basic.get('ai_captions')} | "
                f"24/7 support: {basic.get('support_24_7')}"
            )
        if pro:
            lines.append(
                "- Pro: $"
                f"{pro.get('price')}/{pro.get('billing')} | "
                f"videos: {pro.get('videos_per_month')} | "
                f"resolution: {pro.get('max_resolution')} | "
                f"AI captions: {pro.get('ai_captions')} | "
                f"24/7 support: {pro.get('support_24_7')}"
            )

    features = context.get("features")
    if features:
        lines.append("Features:")
        for tier, items in features.items():
            joined = ", ".join(items)
            lines.append(f"- {tier.title()}: {joined}")

    policies = context.get("policies")
    if policies:
        lines.append("Policies:")
        lines.append(
            f"- Refund period: {policies.get('refund_period_days')} days"
        )
        lines.append(f"- Refund policy: {policies.get('refund_policy')}")
        lines.append(
            f"- 24/7 support available on: {policies.get('support_24_7_plan')}"
        )

    return "\n".join(lines).strip()


def retrieve_context(query: str, kb: Dict[str, Any]) -> str:
    """Retrieve and format relevant context from the knowledge base.

    Args:
        query: The user message to match against KB topics.
        kb: Parsed knowledge base dictionary.

    Returns:
        Prompt-ready context text for downstream LLM response generation.
    """
    sections = _match_sections(query)
    selected_context: Dict[str, Any] = {
        section: kb.get(section) for section in sections if section in kb
    }

    if not selected_context:
        logger.warning("No relevant context found for query: %s", query)
        return "I don't have that information in my knowledge base."

    return format_context_for_prompt(selected_context)
