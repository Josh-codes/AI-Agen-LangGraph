"""Streamlit UI for the AutoStream social-to-lead agent."""

from __future__ import annotations

import logging
from typing import Iterable

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import AgentState, create_initial_state, process_message

logger = logging.getLogger(__name__)


def _ensure_session_state() -> None:
    """Initialize Streamlit session state keys used by the app."""
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = create_initial_state()


def _render_messages(messages: Iterable[BaseMessage]) -> None:
    """Render conversation history in Streamlit chat bubbles."""
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(str(message.content))
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(str(message.content))


def _render_sidebar(state: AgentState) -> None:
    """Render helper diagnostics and controls."""
    with st.sidebar:
        st.header("Session")

        if st.button("Reset conversation", type="primary", use_container_width=True):
            st.session_state.agent_state = create_initial_state()
            st.rerun()

        st.divider()
        st.subheader("Live agent state")
        st.write(f"Intent: {state.get('intent') or 'None'}")
        collected = state.get("collected_fields") or []
        st.write(f"Collected fields: {', '.join(collected) if collected else 'None'}")

        with st.expander("Lead data", expanded=False):
            st.json(state.get("lead_data", {}))

        with st.expander("RAG context", expanded=False):
            st.write(state.get("rag_context") or "No context retrieved yet.")


st.set_page_config(
    page_title="AutoStream Lead Agent",
    page_icon=":speech_balloon:",
    layout="wide",
)

st.title("AutoStream Social-to-Lead Agent")
st.caption("Ask about pricing and features, or tell the assistant you are ready to start.")

_ensure_session_state()
state: AgentState = st.session_state.agent_state

_render_sidebar(state)
_render_messages(state["messages"])

user_message = st.chat_input("Type your message...")

if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                st.session_state.agent_state = process_message(state, user_message)
            except Exception as exc:
                logger.exception("Streamlit turn execution failed: %s", exc)
                st.error("Something went wrong while processing your message. Please try again.")
            else:
                st.rerun()
