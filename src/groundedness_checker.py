
from azure.ai.contentsafety import ContentSafetyClient

from azure.ai.contentsafety.models import AnalyzeTextOptions

from azure.core.credentials import AzureKeyCredential

from openai import OpenAI

from dataclasses import dataclass

from typing import Optional

import os

 

@dataclass

class GroundednessResult:

    is_grounded:      bool

    confidence:       float   # 0.0 to 1.0

    ungrounded_claims: list   # list of strings from response not supported by context

 

def check_groundedness(query: str, context_docs: list[str],

                        llm_response: str) -> GroundednessResult:

    '''

    Check whether the LLM response is grounded in the provided context.

    Uses GPT-4o as the grounding judge (Azure Content Safety also has a

    groundedness API in preview — use that in production).

    '''
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
 

    context_text = '\n---\n'.join(context_docs[:3])  # use top 3 docs

 

    grounding_prompt = f'''

You are a fact-checking system. Given source documents and an AI response,

identify any claims in the response that are NOT supported by the source documents.

 

Source documents:

{context_text}

 

AI Response to check:

{llm_response}

 

Respond ONLY with a JSON object:

{{

  "is_grounded": true/false,

  "confidence": 0.0-1.0,

  "ungrounded_claims": ["claim1", "claim2"]

}}

If all claims are supported: is_grounded=true, ungrounded_claims=[]

'''

 

    resp = client.chat.completions.create(

        model='gpt-4o',

        messages=[{'role': 'user', 'content': grounding_prompt}],

        max_tokens=300,

        temperature=0.0,

        response_format={'type': 'json_object'}

    )

 

    import json

    result = json.loads(resp.choices[0].message.content)

    return GroundednessResult(

        is_grounded=result.get('is_grounded', True),

        confidence=result.get('confidence', 0.5),

        ungrounded_claims=result.get('ungrounded_claims', []),

    )

