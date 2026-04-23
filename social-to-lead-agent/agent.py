"""Main LangGraph agent for AutoStream social-to-lead conversations."""

from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from intent_classifier import classify_intent
from rag import load_knowledge_base, retrieve_context
from tools import validate_lead_data, mock_lead_capture

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["name", "email", "platform"]
NON_NAME_PHRASES = {
    "sign me up",
    "sign me in",
    "lets get started",
    "let's get started",
    "start trial",
    "start a trial",
    "book demo",
    "contact sales",
    "i want pro",
    "i want the pro channel",
    "i want pro channel",
    "want pro",
    "go pro",
    "choose pro",
    "take pro",
}
NON_NAME_WORDS = {
    "i",
    "im",
    "i'm",
    "sign",
    "me",
    "up",
    "in",
    "actually",
    "yes",
    "no",
    "never",
    "mind",
    "want",
    "the",
    "plan",
    "for",
    "my",
    "channel",
    "please",
    "start",
    "trial",
    "get",
    "started",
    "interested",
    "pro",
    "basic",
    "choose",
    "take",
    "go",
    "with",
}
PLATFORM_ALIASES = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "twitch": "Twitch",
    "facebook": "Facebook",
    "x": "X",
    "twitter": "Twitter",
    "linkedin": "LinkedIn",
    "snapchat": "Snapchat",
}

RAG_RETRIEVAL_PROMPT = """You are a knowledgeable sales assistant for AutoStream, a SaaS video editing platform.

Use the following context from our knowledge base to answer the user's question:

{retrieved_context}

Provide a helpful, accurate response based ONLY on the information in the context.
If the context doesn't contain the answer, say "I don't have that information in my knowledge base."

Be friendly, concise, and persuasive without being pushy.
"""

LEAD_CAPTURE_PROMPT = """You are collecting lead information for AutoStream.

You need to collect:
1. Name
2. Email
3. Creator Platform (YouTube, Instagram, TikTok, etc.)

Currently collected: {collected_fields}
Still needed: {missing_fields}

Ask for the next missing field in a friendly, conversational way.
Once all fields are collected, confirm the information and express excitement about getting them started.
"""


class AgentState(TypedDict):
    """Persistent state shared across LangGraph nodes."""

    messages: List[BaseMessage]
    intent: Optional[str]
    lead_data: Dict[str, Optional[str]]
    collected_fields: List[str]
    rag_context: Optional[str]


def _initialize_state() -> AgentState:
    """Return a fresh default state object."""
    return AgentState(
        messages=[],
        intent=None,
        lead_data={"name": None, "email": None, "platform": None},
        collected_fields=[],
        rag_context=None,
    )


def create_initial_state() -> AgentState:
    """Public helper for creating a new agent state."""
    return _initialize_state()


def _get_latest_user_message(state: AgentState) -> str:
    """Extract the latest user message from history."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _render_history(messages: List[BaseMessage], turns: int = 6) -> List[str]:
    """Render compact text history for classifier context."""
    sliced = messages[-(turns * 2) :]
    rendered: List[str] = []
    for item in sliced:
        role = "assistant"
        if isinstance(item, HumanMessage):
            role = "user"
        rendered.append(f"{role}: {item.content}")
    return rendered


def _is_valid_email(email: str) -> bool:
    """Validate email with a practical regex."""
    return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email.strip()))


def _extract_email(text: str) -> Optional[str]:
    """Extract an email address from free text."""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if not email_match:
        return None
    email = email_match.group(0)
    return email if _is_valid_email(email) else None


def _extract_name(text: str) -> Optional[str]:
    """Extract a likely full name from user input."""
    cleaned = text.strip()
    lowered = cleaned.lower()

    if not cleaned:
        return None
    if lowered in PLATFORM_ALIASES:
        return None
    if any(token in lowered for token in ["platform", "channel"]):
        return None
    if "?" in cleaned:
        return None
    if any(phrase in lowered for phrase in NON_NAME_PHRASES):
        return None
    if re.search(r"\b(want|choose|take|go)\b", lowered) and re.search(
        r"\b(pro|basic|plan|channel)\b", lowered
    ):
        return None

    explicit_patterns = [
        r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s'-]{1,60})$",
        r"name[:\-]\s*([A-Za-z][A-Za-z\s'-]{1,60})$",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .,")
            if len(candidate) >= 2:
                return candidate.title()

    if "@" in cleaned:
        return None

    if 1 <= len(cleaned.split()) <= 4 and len(cleaned) <= 60 and not cleaned.endswith("?"):
        filtered = re.sub(r"[^A-Za-z\s'-]", "", cleaned).strip()
        words = [word for word in re.split(r"\s+", filtered.lower()) if word]
        if words and all(word in NON_NAME_WORDS for word in words):
            return None
        if len(filtered) >= 2:
            return filtered.title()

    return None


def _extract_platform(text: str) -> Optional[str]:
    """Extract supported creator platform from text."""
    lowered = text.lower()
    for alias, canonical in PLATFORM_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical

    generic_match = re.search(
        r"(?:platform is|on|channel is|creator platform)\s+([A-Za-z][A-Za-z\s]{1,30})",
        text,
        flags=re.IGNORECASE,
    )
    if generic_match:
        return generic_match.group(1).strip().title()

    stripped = text.strip()
    if (
        len(stripped.split()) == 1
        and stripped.isalpha()
        and len(stripped) >= 3
        and stripped.lower() not in {"yes", "no", "maybe", "thanks", "okay"}
    ):
        return stripped.title()

    return None


def _contains_platform_alias(text: str) -> bool:
    """Return True when text contains any known platform alias."""
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in PLATFORM_ALIASES)


def _refresh_collected_fields(state: AgentState) -> None:
    """Synchronize the collected_fields list with current lead_data."""
    collected: List[str] = []
    for field in REQUIRED_FIELDS:
        value = state["lead_data"].get(field)
        if value and str(value).strip():
            collected.append(field)
    state["collected_fields"] = collected


def _missing_fields(state: AgentState) -> List[str]:
    """Return required lead fields still missing in strict order."""
    return [field for field in REQUIRED_FIELDS if field not in state["collected_fields"]]


def _lead_capture_in_progress(state: AgentState) -> bool:
    """Check whether lead collection has started but is not complete."""
    _refresh_collected_fields(state)
    return 0 < len(state["collected_fields"]) < len(REQUIRED_FIELDS)


def _looks_like_lead_data_input(text: str) -> bool:
    """Detect whether a message likely contains lead details."""
    lowered = text.lower()
    return bool(
        _extract_email(text)
        or _extract_platform(text)
        or _extract_name(text)
        or any(token in lowered for token in ["name", "email", "youtube", "instagram", "tiktok"])
    )


def _is_explicit_product_inquiry(text: str) -> bool:
    """Detect explicit product questions that should interrupt lead capture."""
    lowered = text.lower()
    inquiry_terms = [
        "price",
        "pricing",
        "cost",
        "feature",
        "features",
        "plan",
        "plans",
        "support",
        "refund",
        "policy",
        "what",
        "how",
    ]
    return "?" in lowered or any(term in lowered for term in inquiry_terms)


def _ask_for_next_field(state: AgentState, invalid_email: bool = False) -> str:
    """Generate a friendly prompt for the next missing lead field."""
    missing = _missing_fields(state)
    collected = state["collected_fields"]

    if not missing:
        return "Great, I have everything I need. I am capturing your details now."

    lead_prompt = LEAD_CAPTURE_PROMPT.format(
        collected_fields=collected,
        missing_fields=missing,
    )
    logger.debug("Lead collection prompt context: %s", lead_prompt)

    next_field = missing[0]
    if invalid_email:
        return "That email format looks invalid. Please share a valid email address."

    if next_field == "name":
        return "Excellent choice. What is your name?"
    if next_field == "email":
        name = state["lead_data"].get("name") or "there"
        return f"Great to meet you, {name}. What is your email address?"
    return "Perfect. Which creator platform do you primarily use (YouTube, Instagram, TikTok, etc.)?"


def _build_llm() -> Optional[ChatGoogleGenerativeAI]:
    """Build Gemini client for response generation."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    if not api_key:
        logger.warning("GOOGLE_API_KEY is not set. Falling back to template responses.")
        return None

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        max_tokens=256,
        timeout=12,
        max_retries=1,
        google_api_key=api_key,
    )


def _extract_text_from_model_content(content: Any) -> str:
    """Extract readable plain text from Gemini model content payloads."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = _extract_text_from_model_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        # Gemini payloads commonly expose text in this order.
        for key in ("text", "output_text", "content"):
            if key in content:
                text = _extract_text_from_model_content(content[key])
                if text:
                    return text

        parts = content.get("parts")
        if parts is not None:
            text = _extract_text_from_model_content(parts)
            if text:
                return text

        # Fallback: recursively scan values.
        for value in content.values():
            text = _extract_text_from_model_content(value)
            if text:
                return text

        return ""

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr.strip()

    return str(content).strip()


def detect_intent_node(state: AgentState) -> AgentState:
    """Detect conversation intent for current user message."""
    user_text = _get_latest_user_message(state)
    history = _render_history(state["messages"])
    intent = classify_intent(user_text, history)

    previously_high_intent = state.get("intent") == "high_intent"
    lead_complete = validate_lead_data(state["lead_data"])

    if previously_high_intent and not lead_complete and not _is_explicit_product_inquiry(user_text):
        intent = "high_intent"

    if _lead_capture_in_progress(state) and _looks_like_lead_data_input(user_text):
        intent = "high_intent"

    state["intent"] = intent
    logger.info("Detected intent=%s for message='%s'", intent, user_text)
    return state


def retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve RAG context for inquiry messages."""
    user_text = _get_latest_user_message(state)
    kb = load_knowledge_base()
    state["rag_context"] = retrieve_context(user_text, kb)
    return state


def chat_node(state: AgentState) -> AgentState:
    """Generate conversational response for greetings and inquiries."""
    intent = state.get("intent")

    if intent == "greeting":
        state["messages"].append(
            AIMessage(
                content=(
                    "Hello! Welcome to AutoStream. I can help with pricing, features, "
                    "and getting you started on the right plan for your creator workflow."
                )
            )
        )
        return state

    if intent == "inquiry":
        context = state.get("rag_context") or "I don't have that information in my knowledge base."
        response_text = (
            "Here is what I found in the AutoStream knowledge base:\n"
            f"{context}\n\n"
            "If you want, I can also help you pick the best plan for your channel."
        )
        state["messages"].append(AIMessage(content=response_text))
        return state

    state["messages"].append(
        AIMessage(
            content=(
                "I can help with pricing, features, and getting you started. "
                "What would you like to know?"
            )
        )
    )
    return state


def collect_lead_node(state: AgentState) -> AgentState:
    """Collect and validate lead fields in strict sequence."""
    user_text = _get_latest_user_message(state).strip()
    missing_before = _missing_fields(state)

    invalid_email = False

    if missing_before:
        next_field = missing_before[0]

        if next_field == "name" and not state["lead_data"].get("name"):
            name = _extract_name(user_text)
            if name:
                state["lead_data"]["name"] = name
                logger.info("Collected lead field: name")

        elif next_field == "email" and not state["lead_data"].get("email"):
            email = _extract_email(user_text)
            if email:
                state["lead_data"]["email"] = email
                logger.info("Collected lead field: email")
            elif "@" in user_text or "email" in user_text.lower():
                invalid_email = True

        elif next_field == "platform" and not state["lead_data"].get("platform"):
            platform = _extract_platform(user_text)
            if platform:
                state["lead_data"]["platform"] = platform
                logger.info("Collected lead field: platform")

    _refresh_collected_fields(state)

    if validate_lead_data(state["lead_data"]):
        state["messages"].append(
            AIMessage(content="Awesome, thanks. I have all details and will register your interest now.")
        )
        return state

    prompt = _ask_for_next_field(state, invalid_email=invalid_email)
    state["messages"].append(AIMessage(content=prompt))
    return state


def capture_lead_node(state: AgentState) -> AgentState:
    """Execute mock lead capture after all required fields are valid."""
    lead_data = state["lead_data"]

    if not validate_lead_data(lead_data):
        _refresh_collected_fields(state)
        prompt = _ask_for_next_field(state)
        state["messages"].append(AIMessage(content=prompt))
        return state

    try:
        result = mock_lead_capture(
            cast(str, lead_data["name"]),
            cast(str, lead_data["email"]),
            cast(str, lead_data["platform"]),
        )
        state["messages"].append(
            AIMessage(
                content=(
                    "Great news. You are all set. "
                    f"I have registered your interest for {lead_data['platform']}. "
                    f"Our team will reach out to {lead_data['email']} shortly. "
                    f"Reference ID: {result['lead_id']}"
                )
            )
        )
        logger.info("Lead capture completed successfully")
    except Exception as exc:
        logger.exception("Lead capture failed: %s", exc)
        state["messages"].append(
            AIMessage(
                content=(
                    "I hit an issue while saving your details. Please share your "
                    "name, email, and platform one more time."
                )
            )
        )

    state["lead_data"] = {"name": None, "email": None, "platform": None}
    state["collected_fields"] = []
    return state


def _route_from_intent(state: AgentState) -> str:
    """Route from detect_intent node to the next node."""
    intent = state.get("intent")
    if intent == "high_intent":
        return "collect_lead"
    if intent == "inquiry":
        return "retrieve_context"
    return "chat"


def _route_after_collect(state: AgentState) -> str:
    """Route to capture when all lead fields are complete and valid."""
    if validate_lead_data(state["lead_data"]):
        return "capture_lead"
    return "end"


def setup_agent():
    """Create and compile the LangGraph state machine."""
    workflow = StateGraph(AgentState)

    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("collect_lead", collect_lead_node)
    workflow.add_node("capture_lead", capture_lead_node)

    workflow.set_entry_point("detect_intent")

    workflow.add_conditional_edges(
        "detect_intent",
        _route_from_intent,
        {
            "chat": "chat",
            "retrieve_context": "retrieve_context",
            "collect_lead": "collect_lead",
        },
    )

    workflow.add_edge("retrieve_context", "chat")
    workflow.add_edge("chat", END)

    workflow.add_conditional_edges(
        "collect_lead",
        _route_after_collect,
        {
            "capture_lead": "capture_lead",
            "end": END,
        },
    )
    workflow.add_edge("capture_lead", END)

    return workflow.compile()


_AGENT_GRAPH = setup_agent()


def process_message(state: AgentState, user_input: str) -> AgentState:
    """Run one conversational turn through the graph.

    Args:
        state: Current agent state.
        user_input: Latest user message.

    Returns:
        Updated agent state after graph execution.
    """
    next_state = copy.deepcopy(state)
    next_state["messages"].append(HumanMessage(content=user_input))

    updated_state = cast(AgentState, _AGENT_GRAPH.invoke(next_state))

    max_turns = int(os.getenv("MAX_CONVERSATION_TURNS", "6"))
    max_messages = max_turns * 2 + 4
    if len(updated_state["messages"]) > max_messages:
        updated_state["messages"] = updated_state["messages"][-max_messages:]

    return updated_state


def run_conversation() -> None:
    """Run the local CLI chat loop for manual testing."""
    load_dotenv()

    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("AutoStream Lead Capture Agent")
    print("Type 'exit' to end the conversation.")

    state = _initialize_state()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Thanks for chatting. Goodbye.")
            break

        if not user_input:
            print("Agent: Please enter a message so I can help.")
            continue

        state = process_message(state, user_input)

        assistant_message = ""
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                assistant_message = str(message.content)
                break

        print(f"Agent: {assistant_message}")


if __name__ == "__main__":
    run_conversation()
