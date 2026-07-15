from azure.ai.contentsafety import ContentSafetyClient
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
# Block at medium or above (severity >= 4)
THRESHOLDS = {
    TextCategory.HATE:        4,
    TextCategory.VIOLENCE:    4,
    TextCategory.SEXUAL:      4,
    TextCategory.SELF_HARM:   2,  # stricter for self-harm
}
 
class SafetyScreener:
    def __init__(self):
        self.client = ContentSafetyClient(
            endpoint=os.environ['CONTENT_SAFETY_ENDPOINT'],
            credential=AzureKeyCredential(os.environ['CONTENT_SAFETY_KEY'])
        )
        self.blocklist_name = 'copilot-policy-blocklist'
        self._ensure_blocklist_exists()
 
    def _ensure_blocklist_exists(self):
        '''Create custom blocklist with policy-specific terms.'''
        try:
            from azure.ai.contentsafety.models import TextBlocklist
            try:
                self.client.create_or_update_text_blocklist(
                    blocklist_name=self.blocklist_name,
                    options=TextBlocklist(blocklist_name=self.blocklist_name, description="Copilot policy")
                )
            except Exception:
                pass

            # Add terms that should always be blocked in a Microsoft Copilot context
            custom_terms = [
                'IGNORE PREVIOUS INSTRUCTIONS',  # prompt injection pattern
                'ignore your system prompt',
                'jailbreak',
                'DAN mode',
            ]
            items = [TextBlocklistItem(text=t) for t in custom_terms]
            self.client.add_or_update_text_blocklist_items(
                blocklist_name=self.blocklist_name,
                options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=items)
            )
        except Exception as e:
            logger.warning(f'Blocklist setup: {e}')
 
    def screen_input(self, text: str) -> SafetyResult:
        '''Screen user input. Returns SafetyResult with allow/block decision.'''
        try:
            response = self.client.analyze_text(AnalyzeTextOptions(
                text=text,
                categories=[TextCategory.HATE, TextCategory.VIOLENCE,
                             TextCategory.SEXUAL, TextCategory.SELF_HARM],
            ))
        except Exception as e:
            logger.error(f'Content Safety API error: {e}')
            # Fail safe: block on API error
            return SafetyResult(allowed=False, block_reason='safety_api_error',
                                hate_severity=0, violence_severity=0,
                                sexual_severity=0, selfharm_severity=0,
                                blocklist_matched=False, prompt_injection=False)
 
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
