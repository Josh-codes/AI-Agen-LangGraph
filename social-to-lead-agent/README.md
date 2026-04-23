# AutoStream Social-to-Lead Agent

Conversational AI agent that identifies qualified leads from social conversations, answers product questions with RAG, and captures validated lead details when buying intent is detected.

## What This Agent Does

- Detects intent per user turn: greeting, inquiry, or high_intent.
- Answers product questions from knowledge_base.json (pricing, features, policies).
- Collects lead fields in order: name -> email -> creator platform.
- Validates lead data before calling the capture tool.
- Runs in both CLI mode and Streamlit web UI mode.
- Uses deterministic greeting/inquiry outputs to avoid off-domain hallucinations.

## Project Structure

- agent.py: LangGraph state machine and conversation orchestration.
- intent_classifier.py: Gemini-based 3-class intent classification.
- rag.py: JSON knowledge retrieval and context formatting.
- tools.py: Lead validation and mock lead capture execution.
- knowledge_base.json: Product pricing, features, and policies.
- streamlit_app.py: Streamlit chat interface.
- requirements.txt: Python dependencies.
- demo/demo_video.md: Demo flow notes.

## Prerequisites

- Python 3.9+
- A Gemini API key from Google AI Studio

## Setup Instructions

1. Clone or open the project in your workspace.
2. Create a virtual environment:
   - Windows PowerShell:
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1
3. Install dependencies:
   - pip install -r requirements.txt
4. Configure environment variables:
   - Create a .env file in social-to-lead-agent
   - Add GOOGLE_API_KEY=your_api_key_here
   - Optional: GEMINI_MODEL=gemini-flash-latest
5. Run the CLI agent:
   - python agent.py

## Streamlit UI

Run a web chat interface for the same LangGraph agent:

1. Ensure dependencies are installed:
   - pip install -r requirements.txt
2. Start the Streamlit app from the social-to-lead-agent folder:
   - streamlit run streamlit_app.py

The UI includes:

- Session-persistent conversation state.
- Sidebar diagnostics for intent, collected fields, lead data, and RAG context.
- Immediate user-message rendering after pressing Enter.
- Assistant "Thinking..." loader while processing each turn.

## How It Works

The agent classifies each incoming message into greeting, inquiry, or high_intent.

- Inquiry route: retrieve_context_node loads relevant facts and returns a grounded response from the knowledge base context.
- High-intent route: collect_lead_node gathers missing lead fields step-by-step.
- Capture route: capture_lead_node executes only after validate_lead_data() returns True.

Conversation state persists in a LangGraph state dictionary with message history, current intent, lead data, collected fields, and latest RAG context.

LangGraph is used because this workflow needs explicit state transitions, deterministic routing, and strict control over tool execution.

## Reliability Guards

- Strong purchase signals (for example, "i want pro") are forced to high_intent.
- Plan-selection phrases are blocked from being treated as person names.
- Greeting and inquiry responses are deterministic and product-specific to prevent off-topic outputs.
- Tool execution is gated by strict lead validation.

## WhatsApp Deployment Strategy

1. Integrate WhatsApp Business API via a webhook endpoint (for example, FastAPI or Flask).
2. On each inbound webhook event:
   - Extract user phone number as session key.
   - Load persisted AgentState from Redis, DynamoDB, or PostgreSQL.
   - Pass user message into process_message(state, user_input).
   - Save updated state back to storage.
   - Send assistant response through WhatsApp send-message API.
3. Use asynchronous processing (queue worker) to handle webhook bursts and retries.
4. Add idempotency keys per message ID to prevent duplicate processing.
5. Store observability signals (intent, routing path, tool calls, latency) for monitoring.

## Testing Scenarios

1. Full lead capture flow:
   - Greeting -> pricing inquiry -> high intent -> collect name/email/platform -> capture lead.
2. Interrupted flow:
   - Start sign-up -> switch to feature question -> return to sign-up.
3. Validation flow:
   - Provide invalid email -> agent reprompts -> valid email accepted.
4. State retention:
   - 5-6 turns of mixed intent with accurate contextual responses.
5. Pro-intent parsing:
   - "i want pro" and "i want the pro channel" should enter lead collection and ask for name.

## Notes

- If GOOGLE_API_KEY is missing, intent classification falls back to heuristics.
- Logging is enabled for intent decisions, retrieval routing, and lead capture events.
- mock_lead_capture() is a stub and should be replaced with real CRM integration for production.
