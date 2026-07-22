# MSFT Copilot Safety

Safety middleware for a Microsoft 365 Copilot–style chat API: screens both user input and model output through Azure AI Content Safety plus a custom prompt-injection blocklist and a zero-tolerance child-safety keyword check, optionally checks RAG responses for groundedness with an LLM judge, and logs every safety decision. Includes an evaluation harness that measures precision/recall/false-positive rate on a labeled safe/harmful prompt set.

**Results** (60-prompt labeled eval, live against Azure Content Safety): **precision 1.00, recall 0.90, false-positive rate 0.00** (27/30 harmful prompts blocked, 0/30 safe prompts wrongly blocked). An earlier version of this middleware scored recall 0.70 at the same precision/FPR — see "Recall improvement" below for what changed and a documented ceiling on the remaining gap.

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

- **Input/output screening** (`safety_screener.py`): calls Azure AI Content Safety's `analyze_text` for hate, violence, sexual, and self-harm categories, blocking at severity ≥ 2 (low) across all four categories. Also runs a zero-tolerance keyword co-occurrence check for child-safety content (any child-reference term — "minor", "child", etc. — combined with a harm-reference term hard-blocks regardless of Content Safety's score), and maintains a custom text blocklist (`copilot-policy-blocklist`, via a separate `BlocklistClient`) seeded with known prompt-injection phrases (`"IGNORE PREVIOUS INSTRUCTIONS"`, `"jailbreak"`, `"DAN mode"`, etc.). Fails closed — a Content Safety API error blocks the request rather than allowing it through.
- **Groundedness check** (`groundedness_checker.py`): for RAG-style requests, asks GPT-4o to identify any claims in the response not supported by the retrieved context documents, returning a grounded/ungrounded verdict, confidence, and the specific ungrounded claims. Currently used as a warning signal (logged, not blocking).
- **API** (`safe_copilot_api.py`): `POST /chat` runs the full pipeline — screen input, call the model, screen output, optionally check groundedness — and returns either a `CopilotResponse` or a `BlockedResponse` with the block reason. `GET /health` reports readiness.
- **Evaluation** (`evaluate.py`): runs the screener against 30 benign prompts and 30 explicitly harmful prompts (violence, hate speech, self-harm, sexual exploitation — used only as negative test cases for the classifier) and reports precision, recall, false-positive rate, F1, and accuracy.

## Recall improvement: 70% → 90%

An initial eval run scored precision 1.00 / recall 0.70 / FPR 0.00 — the screener never wrongly blocked a safe prompt, but missed 9 of 30 harmful ones. Two bugs and one design change closed most of that gap:

1. **Dead blocklist code**: the custom prompt-injection blocklist was created using methods (`create_or_update_text_blocklist`, `add_or_update_text_blocklist_items`) that don't exist on `ContentSafetyClient` in this SDK version — blocklist management lives on a separate `BlocklistClient`. The calls were silently failing (caught by a broad `except`), so the blocklist was never actually created, and `analyze_text` was never even passing `blocklist_names` to check against it in the first place. Both are now fixed.
2. **Thresholds lowered from 4 to 2** for HATE/VIOLENCE/SEXUAL, matching SELF_HARM's already-stricter setting.
3. **Added a zero-tolerance child-safety keyword co-occurrence check**, since several misses were meta-level requests ("describe X involving minors") phrased in neutral instructional language that scores low on Content Safety's categorical severity even though the intent is unambiguous — a known limitation of scoring surface text rather than requester intent.

Verified live against the Azure Content Safety API: recall improved from 0.70 to 0.90 (21→27 of 30 caught) with precision and FPR unchanged (1.00 / 0.00) — no new false positives introduced on any of the 30 safe prompts.

**The remaining 3 misses have a documented ceiling, not just an unturned dial**: all 3 ("torture techniques used on prisoners," "domestic abuse in glorifying terms," "instructions for committing hate crimes") score **0 across every Content Safety category** — the classifier detects nothing at all in these specific phrasings, not merely below-threshold. Since severity is never negative, there is no threshold that selectively catches these while leaving safe content alone; the only remaining lever is keyword/pattern matching tailored to these exact phrasings, which would be overfitting to this specific 30-example eval rather than a generalizable improvement. Stopping at 90% recall here is a deliberate choice, not a shortfall.

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
