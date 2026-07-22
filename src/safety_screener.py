import re

from azure.ai.contentsafety import ContentSafetyClient, BlocklistClient
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions, TextCategory, AddOrUpdateTextBlocklistItemsOptions,
    TextBlocklistItem, TextBlocklist
)
from azure.core.credentials import AzureKeyCredential
from dataclasses import dataclass
from typing import Optional
import os, logging

logger = logging.getLogger(__name__)

@dataclass
class SafetyResult:
    allowed:       bool
    block_reason:  Optional[str]
    hate_severity:      int
    violence_severity:  int
    sexual_severity:    int
    selfharm_severity:  int
    blocklist_matched:  bool
    prompt_injection:   bool

# Severity thresholds: 0=safe 2=low 4=medium 6=high
# Block at low or above (severity >= 2). Originally 4 for HATE/VIOLENCE/SEXUAL;
# lowered to match SELF_HARM's stricter setting after an eval run showed 70%
# recall on harmful content (9/30 misses) at the threshold-4 setting.
THRESHOLDS = {
    TextCategory.HATE:        2,
    TextCategory.VIOLENCE:    2,
    TextCategory.SEXUAL:      2,
    TextCategory.SELF_HARM:   2,
}

# Zero-tolerance keyword co-occurrence check for child-safety content. Several
# of the eval misses were meta-level requests ("describe X involving minors",
# "how to kidnap and hurt children") phrased in neutral instructional language
# that scores low on Content Safety's categorical severity scale even though
# the intent is unambiguous — this is a known limitation of scoring surface
# text rather than requester intent. Any co-occurrence of a child-reference
# term with a harm-reference term hard-blocks regardless of category severity.
_CHILD_TERMS = re.compile(r'\b(minor|minors|underage|child|children|kids?)\b', re.IGNORECASE)
_HARM_TERMS = re.compile(
    r'\b(sex(ual)?|exploit\w*|abuse|assault|kidnap\w*|kill\w*|hurt\w*|torture\w*|'
    r'violen\w*|molest\w*|traffic\w*|victim\w*)\b',
    re.IGNORECASE,
)


def _child_safety_hard_block(text: str) -> bool:
    return bool(_CHILD_TERMS.search(text)) and bool(_HARM_TERMS.search(text))
 
class SafetyScreener:
    def __init__(self):
        self.client = ContentSafetyClient(
            endpoint=os.environ['CONTENT_SAFETY_ENDPOINT'],
            credential=AzureKeyCredential(os.environ['CONTENT_SAFETY_KEY'])
        )
        # Blocklist management lives on a separate BlocklistClient in this SDK
        # version, not on ContentSafetyClient. The original code called
        # blocklist methods on self.client (ContentSafetyClient), which don't
        # exist there — this silently failed (caught by the broad except
        # below) so the blocklist was never actually created server-side.
        self.blocklist_client = BlocklistClient(
            endpoint=os.environ['CONTENT_SAFETY_ENDPOINT'],
            credential=AzureKeyCredential(os.environ['CONTENT_SAFETY_KEY'])
        )
        self.blocklist_name = 'copilot-policy-blocklist'
        self.blocklist_ready = self._ensure_blocklist_exists()

    def _ensure_blocklist_exists(self) -> bool:
        '''Create custom blocklist with policy-specific terms. Returns True on success.'''
        try:
            self.blocklist_client.create_or_update_text_blocklist(
                blocklist_name=self.blocklist_name,
                options=TextBlocklist(blocklist_name=self.blocklist_name, description="Copilot policy")
            )

            # Add terms that should always be blocked in a Microsoft Copilot context
            custom_terms = [
                'IGNORE PREVIOUS INSTRUCTIONS',  # prompt injection pattern
                'ignore your system prompt',
                'jailbreak',
                'DAN mode',
            ]
            items = [TextBlocklistItem(text=t) for t in custom_terms]
            self.blocklist_client.add_or_update_blocklist_items(
                blocklist_name=self.blocklist_name,
                options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=items)
            )
            return True
        except Exception as e:
            logger.warning(f'Blocklist setup failed, proceeding without blocklist checking: {e}')
            return False
 
    def screen_input(self, text: str) -> SafetyResult:
        '''Screen user input. Returns SafetyResult with allow/block decision.'''
        if _child_safety_hard_block(text):
            return SafetyResult(allowed=False, block_reason='child_safety_hard_block',
                                hate_severity=0, violence_severity=0,
                                sexual_severity=0, selfharm_severity=0,
                                blocklist_matched=False, prompt_injection=False)

        try:
            options = dict(
                text=text,
                categories=[TextCategory.HATE, TextCategory.VIOLENCE,
                             TextCategory.SEXUAL, TextCategory.SELF_HARM],
            )
            if self.blocklist_ready:
                options['blocklist_names'] = [self.blocklist_name]
                options['halt_on_blocklist_hit'] = False
            response = self.client.analyze_text(AnalyzeTextOptions(**options))
        except Exception as e:
            logger.error(f'Content Safety API error: {e}')
            # Fail safe: block on API error
            return SafetyResult(allowed=False, block_reason='safety_api_error',
                                hate_severity=0, violence_severity=0,
                                sexual_severity=0, selfharm_severity=0,
                                blocklist_matched=False, prompt_injection=False)

        # The custom blocklist (prompt-injection phrases) was previously
        # created in Azure but never actually passed to analyze_text, so it
        # was never consulted. Fixed above via blocklist_names=[...]; check
        # the match here.
        if response.blocklists_match:
            return SafetyResult(
                allowed=False,
                block_reason=f'blocklist_match_{response.blocklists_match[0].blocklist_item_id}',
                hate_severity=0, violence_severity=0,
                sexual_severity=0, selfharm_severity=0,
                blocklist_matched=True, prompt_injection=True,
            )

        scores = {cat.category: cat.severity for cat in response.categories_analysis}

        # Check each threshold
        for category, threshold in THRESHOLDS.items():
            severity = scores.get(category, 0)
            if severity >= threshold:
                return SafetyResult(
                    allowed=False,
                    block_reason=f'{category}_severity_{severity}',
                    hate_severity=scores.get(TextCategory.HATE, 0),
                    violence_severity=scores.get(TextCategory.VIOLENCE, 0),
                    sexual_severity=scores.get(TextCategory.SEXUAL, 0),
                    selfharm_severity=scores.get(TextCategory.SELF_HARM, 0),
                    blocklist_matched=False,
                    prompt_injection=False,
                )
 
        return SafetyResult(
            allowed=True, block_reason=None,
            hate_severity=scores.get(TextCategory.HATE, 0),
            violence_severity=scores.get(TextCategory.VIOLENCE, 0),
            sexual_severity=scores.get(TextCategory.SEXUAL, 0),
            selfharm_severity=scores.get(TextCategory.SELF_HARM, 0),
            blocklist_matched=False, prompt_injection=False,
        )
