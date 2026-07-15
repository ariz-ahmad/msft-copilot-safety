# MSFT Copilot Safety

Safety middleware for a Microsoft 365 Copilot–style chat API: screens both user input and model output through Azure AI Content Safety plus a custom prompt-injection blocklist, optionally checks RAG responses for groundedness with an LLM judge, and logs every safety decision. Includes an evaluation harness that measures precision/recall/false-positive rate on a labeled safe/harmful prompt set.

## Overview

```
user_message → SafetyScreener (input) → GPT-4o → SafetyScreener (output) → groundedness_checker (optional) → response
                     │                                    │
                     ▼                                    ▼
              block on hate/violence/                block if the
              sexual/self-harm severity,              output itself
              or a blocklist match                    fails screening
              (prompt-injection phrases)
```

Every request — allowed or blocked — is logged with its safety scores and the action taken (`ALLOWED`, `BLOCKED_INPUT`, `BLOCKED_OUTPUT`, `GROUNDING_WARNING`).

## Project structure

```
.
├── src/
│   ├── safety_screener.py       # Azure AI Content Safety client + custom blocklist (prompt injection)
│   ├── groundedness_checker.py  # GPT-4o judge: flags response claims not supported by RAG context
│   ├── safe_copilot_api.py      # FastAPI app: /chat wraps input screening → LLM call → output screening
│   └── evaluate.py              # Precision/recall/FPR eval on 30 safe + 30 harmful labeled prompts
├── data/
│   └── evaluation_results.json  # Example output of src/evaluate.py
└── requirements.txt
```

## How it works

- **Input/output screening** (`safety_screener.py`): calls Azure AI Content Safety's `analyze_text` for hate, violence, sexual, and self-harm categories. Blocks at severity ≥ 4 (medium) for most categories, and ≥ 2 for self-harm (stricter threshold). Also maintains a custom text blocklist (`copilot-policy-blocklist`) seeded with known prompt-injection phrases (`"IGNORE PREVIOUS INSTRUCTIONS"`, `"jailbreak"`, `"DAN mode"`, etc.). Fails closed — a Content Safety API error blocks the request rather than allowing it through.
- **Groundedness check** (`groundedness_checker.py`): for RAG-style requests, asks GPT-4o to identify any claims in the response not supported by the retrieved context documents, returning a grounded/ungrounded verdict, confidence, and the specific ungrounded claims. Currently used as a warning signal (logged, not blocking).
- **API** (`safe_copilot_api.py`): `POST /chat` runs the full pipeline — screen input, call the model, screen output, optionally check groundedness — and returns either a `CopilotResponse` or a `BlockedResponse` with the block reason. `GET /health` reports readiness.
- **Evaluation** (`evaluate.py`): runs the screener against 30 benign prompts and 30 explicitly harmful prompts (violence, hate speech, self-harm, sexual exploitation — used only as negative test cases for the classifier) and reports precision, recall, false-positive rate, F1, and accuracy.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Environment variables required:

```bash
export CONTENT_SAFETY_ENDPOINT=<your-azure-ai-content-safety-endpoint>
export CONTENT_SAFETY_KEY=<your-azure-ai-content-safety-key>
export OPENAI_API_KEY=<your-openai-key>
```

## Usage

```bash
# Run the safety evaluation harness
python -m src.evaluate

# Serve the safety-wrapped Copilot API
uvicorn src.safe_copilot_api:app --reload
# POST /chat  {"user_message": "...", "context_docs": [...], "check_groundedness": true}
```

## Tech stack

Azure AI Content Safety, OpenAI (GPT-4o), FastAPI/uvicorn, pydantic.

## Notes

- `HARMFUL_PROMPTS` in `src/evaluate.py` contains explicit, graphic test prompts (violence, hate speech, self-harm, sexual exploitation). These exist solely as labeled negative test cases to measure the safety screener's recall — the same red-teaming pattern used in published content-safety benchmarks — and are never sent to the LLM, only to the classifier.
- `requirements.txt` lists unpinned package names; pin versions before deploying to production.
