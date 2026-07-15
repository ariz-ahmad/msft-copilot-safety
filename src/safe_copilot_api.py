
from fastapi import FastAPI, HTTPException, Request

from pydantic import BaseModel

from openai import OpenAI

from datetime import datetime, timezone

import os, json, logging, uuid

from src.safety_screener import SafetyScreener

from src.groundedness_checker import check_groundedness

 

app = FastAPI(title='Safe Copilot API', version='1.0.0')

screener   = None

oai_client = None

 

@app.on_event('startup')

def startup():

    global screener, oai_client

    screener   = SafetyScreener()

    oai_client = OpenAI(


        api_key=os.environ['OPENAI_API_KEY'],


    )

    print('Safe Copilot API ready')

 

class CopilotRequest(BaseModel):

    user_message:  str

    system_prompt: str  = 'You are a helpful Microsoft 365 Copilot assistant.'

    context_docs:  list = []   # RAG context documents

    check_groundedness: bool = False

 

class CopilotResponse(BaseModel):

    response:         str

    request_id:       str

    safety_screened:  bool

    groundedness_ok:  bool

    latency_ms:       float

 

class BlockedResponse(BaseModel):

    blocked:      bool = True

    reason:       str

    request_id:   str

 

@app.get('/health')

def health(): return {'status': 'ok', 'screener_ready': screener is not None}

 

@app.post('/chat')

async def chat(req: CopilotRequest, request: Request):

    import time

    start       = time.time()

    request_id  = str(uuid.uuid4())[:8]

 

    # ── STEP 1: Screen the input ─────────────────────────

    safety = screener.screen_input(req.user_message)

 

    log_entry = {

        'request_id':  request_id,

        'timestamp':   datetime.now(timezone.utc).isoformat(),

        'user_message': req.user_message[:200],

        'input_allowed':       safety.allowed,

        'hate_severity':       safety.hate_severity,

        'violence_severity':   safety.violence_severity,

        'blocklist_matched':   safety.blocklist_matched,

    }

 

    if not safety.allowed:

        log_entry['action'] = 'BLOCKED_INPUT'

        log_safety_event(log_entry)

        return BlockedResponse(reason=safety.block_reason, request_id=request_id)

 

    # ── STEP 2: Call the LLM ─────────────────────────────

    context = ''

    if req.context_docs:

        context = 'Context documents:\n' + '\n---\n'.join(req.context_docs[:3])

 

    messages = [

        {'role': 'system', 'content': req.system_prompt + ('\n\n' + context if context else '')},

        {'role': 'user',   'content': req.user_message}

    ]

    llm_resp = oai_client.chat.completions.create(

        model='gpt-4o',

        messages=messages, max_tokens=600, temperature=0.3

    )

    llm_output = llm_resp.choices[0].message.content

 

    # ── STEP 3: Screen the output ────────────────────────

    output_safety = screener.screen_input(llm_output)  # same screener, output text

    if not output_safety.allowed:

        log_entry['action'] = 'BLOCKED_OUTPUT'

        log_safety_event(log_entry)

        return BlockedResponse(reason='output_' + output_safety.block_reason,

                               request_id=request_id)

 

    # ── STEP 4: Groundedness check (for RAG requests) ────

    groundedness_ok = True

    if req.check_groundedness and req.context_docs:
        print(f"DEBUG: cg={req.check_groundedness} docs={bool(req.context_docs)}")

        g = check_groundedness(req.user_message, req.context_docs, llm_output)

        groundedness_ok = g.is_grounded

        log_entry['groundedness_score']    = g.confidence

        log_entry['ungrounded_claims']     = g.ungrounded_claims

        if not g.is_grounded:

            log_entry['action'] = 'GROUNDING_WARNING'

            # Don't block — but flag it for human review

 

    latency = round((time.time() - start) * 1000, 1)

    log_entry.update({'action': 'ALLOWED', 'latency_ms': latency})

    log_safety_event(log_entry)

 

    return CopilotResponse(

        response=llm_output, request_id=request_id,

        safety_screened=True, groundedness_ok=groundedness_ok,

        latency_ms=latency,

    )

 

def log_safety_event(entry: dict):

    '''Write safety decision to log (Azure Monitor / Log Analytics in production).'''

    logging.info(json.dumps(entry))

