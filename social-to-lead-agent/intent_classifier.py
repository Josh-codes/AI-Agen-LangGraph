"""Intent classification for AutoStream social-to-lead conversations."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are an expert intent classifier for a sales conversation.

Classify the user's message into EXACTLY ONE of these categories:

1. "greeting" - Simple greetings, small talk, casual conversation starters
   Examples: "hi", "hello", "how are you", "hey there"

2. "inquiry" - Questions about product features, pricing, plans, capabilities
   Examples: "tell me about pricing", "what features do you have", "how does it work"

3. "high_intent" - User is ready to purchase, sign up, or start a trial
   Examples: "I want to try", "sign me up", "let's get started", "I'm interested in the Pro plan"

Respond with ONLY the category name, nothing else.
"""

VALID_INTENTS = {"greeting", "inquiry", "high_intent"}

GREETING_PATTERN = re.compile(
    r"\b(hi|hello|hey|how are you|good morning|good afternoon|good evening)\b",
    flags=re.IGNORECASE,
)
INQUIRY_PATTERN = re.compile(
    r"\b(price|pricing|cost|feature|features|plan|plans|support|refund|how much|what|tell me|can you)\b",
    flags=re.IGNORECASE,
)
HIGH_INTENT_PATTERN = re.compile(
    r"\b(sign me up|sign me in|i want to try|i want to buy|i want the pro plan|"
    r"i want pro|want the pro plan|i want the pro channel|i want pro channel|"
    r"want pro|go pro|choose pro|i'll take pro|ill take pro|"
    r"let's get started|lets get started|"
    r"start trial|start a trial|ready to buy|interested in|i am interested in|"
    r"take pro|go with pro|"
    r"book demo|contact sales)\b",
    flags=re.IGNORECASE,
)


def _extract_intent_from_text(text: str) -> Optional[str]:
    """Extract a valid intent label from text when possible."""
    cleaned = text.strip().lower().strip("`\"' ")
    if cleaned in VALID_INTENTS:
        return cleaned

    # Prefer explicit labeled formats first (e.g., "intent: inquiry").
    labeled_match = re.search(
        r"(?:intent|label|category)\s*[:=-]\s*(greeting|inquiry|high_intent)",
        cleaned,
    )
    if labeled_match:
        return labeled_match.group(1)

    # Accept short natural-language replies like "the intent is inquiry".
    if len(cleaned) <= 60:
        short_match = re.search(r"\b(greeting|inquiry|high_intent)\b", cleaned)
        if short_match:
            return short_match.group(1)

    return None


def _extract_intent_from_model_content(content: Any) -> Optional[str]:
    """Extract intent from model content across string/list/dict payloads."""
    if content is None:
        return None

    if isinstance(content, str):
        return _extract_intent_from_text(content)

    if isinstance(content, list):
        for item in content:
            extracted = _extract_intent_from_model_content(item)
            if extracted:
                return extracted
        return None

    if isinstance(content, dict):
        for key in ("intent", "label", "text", "content", "output_text"):
            if key in content:
                extracted = _extract_intent_from_model_content(content[key])
                if extracted:
                    return extracted
        for value in content.values():
            extracted = _extract_intent_from_model_content(value)
            if extracted:
                return extracted
        return None

    return _extract_intent_from_text(str(content))


def _is_empty_model_content(content: Any) -> bool:
    """Return True when model content is empty (e.g., [], {}, or blank text)."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip() or content.strip() in {"[]", "{}", "''", '""'}
    if isinstance(content, list):
        return len(content) == 0
    if isinstance(content, dict):
        return len(content) == 0
    return False


def _heuristic_intent(message: str) -> str:
    """Fast fallback intent detector used when model calls fail or are ambiguous."""
    lowered = message.strip().lower()

    if HIGH_INTENT_PATTERN.search(lowered):
        return "high_intent"

    if INQUIRY_PATTERN.search(lowered) or "?" in lowered:
        return "inquiry"

    if GREETING_PATTERN.search(lowered) and len(lowered.split()) <= 6:
        return "greeting"

    if re.search(r"\b\w+@\w+\.\w+\b", lowered):
        return "high_intent"

    return "inquiry"


def _build_classifier() -> ChatGoogleGenerativeAI:
    """Create the Gemini classifier instance from environment settings."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_tokens=8,
        timeout=10,
        max_retries=1,
        google_api_key=api_key,
    )


def classify_intent(message: str, conversation_history: List[Any]) -> str:
    """Classify a user message as greeting, inquiry, or high_intent.

    Args:
        message: Latest user utterance.
        conversation_history: Recent conversation history for context.

    Returns:
        One of: greeting, inquiry, high_intent.
    """
    stripped_message = message.strip()
    if not stripped_message:
        return "inquiry"

    heuristic = _heuristic_intent(stripped_message)

    # Strong purchase/signup signals should be deterministic and not model-dependent.
    if heuristic == "high_intent":
        return "high_intent"

    try:
        classifier = _build_classifier()

        recent_history = conversation_history[-6:] if conversation_history else []
        history_text = "\n".join(str(item) for item in recent_history)

        response = classifier.invoke(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "Conversation history:\n"
                        f"{history_text}\n\n"
                        f"User message:\n{stripped_message}"
                    )
                ),
            ]
        )

        extracted_intent = _extract_intent_from_model_content(response.content)
        if extracted_intent in VALID_INTENTS:
            return extracted_intent

        if _is_empty_model_content(response.content):
            logger.info("Classifier returned empty content; using heuristic '%s'", heuristic)
            return heuristic

        logger.warning(
            "Classifier returned unparseable content '%s'; using heuristic '%s'",
            str(response.content)[:200],
            heuristic,
        )
        return heuristic
    except Exception as exc:
        logger.warning("Intent classification fallback triggered: %s", exc)
        return heuristic
