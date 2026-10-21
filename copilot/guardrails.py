"""Input and output guardrails.

1. PII masking: card numbers, NRIC/FIN, phone numbers and emails are masked
   before anything reaches the model or the conversation store.
2. Prompt injection and scope screening: requests that try to override the
   system prompt, reach other customers' data, or ask for actions the copilot
   is not allowed to take are refused before planning.
3. Human approval: tools that change account state are paused until the
   customer explicitly confirms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Tools that change state. The agent graph pauses before running these.
REQUIRES_APPROVAL = {"freeze_card", "open_dispute"}

_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_NRIC = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.IGNORECASE)
_PHONE = re.compile(r"(?:\+65[ -]?)?\b[689]\d{3}[ -]?\d{4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


@dataclass
class MaskResult:
    text: str
    found: list[str] = field(default_factory=list)


def mask_pii(text: str) -> MaskResult:
    found: list[str] = []

    def card(m: re.Match) -> str:
        found.append("card_number")
        digits = re.sub(r"\D", "", m.group())
        return f"[CARD ****{digits[-4:]}]"

    def tag(kind: str):
        def _sub(_: re.Match) -> str:
            found.append(kind)
            return f"[{kind.upper()}]"
        return _sub

    text = _CARD.sub(card, text)
    text = _NRIC.sub(tag("nric"), text)
    text = _EMAIL.sub(tag("email"), text)
    text = _PHONE.sub(tag("phone"), text)
    return MaskResult(text, found)


_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (your|the|all) (instructions|rules|guidelines)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"\bsystem prompt\b",
    r"you are now\b",
    r"\b(developer|god|admin|debug) mode\b",
    r"\bjailbreak\b",
    r"pretend (you are|to be)",
    r"act as (an? )?(admin|administrator|bank officer|supervisor)",
]
_OTHER_CUSTOMERS = [
    r"(another|other|someone else'?s?|a different) (customer|person|user)'?s?",
    r"\ball (customers|users|accounts in the bank)\b",
    r"\bcustomer (id )?C\d{3}\b",
]
_UNSUPPORTED = [
    r"\b(transfer|send|wire|paynow|pay now|remit)\b.*\b(money|\$|sgd|funds|to)\b",
    r"\b(approve|increase|raise)\b.*\b(loan|credit limit|limit)\b",
    r"\bclose (my )?account\b",
    r"^(please )?(unfreeze|unblock|reactivate)\b",
    r"\b(can you|could you|please) (unfreeze|unblock|reactivate)\b",
]

REFUSALS = {
    "prompt_injection": "I can't change how I operate or share my instructions. I'm happy to help with your accounts, cards, disputes or bank policies.",
    "other_customer": "I can only access the accounts of the customer I'm signed in for, so I can't help with anyone else's information.",
    "unsupported_action": "That isn't something I can do in this chat. Please use the bank app or speak to a branch officer, and I can help with balances, transactions, card freezes, disputes or policy questions.",
}


@dataclass
class ScreenResult:
    allowed: bool
    category: str | None = None

    @property
    def message(self) -> str:
        return REFUSALS.get(self.category or "", "")


def screen_input(text: str) -> ScreenResult:
    lowered = text.lower()
    for category, patterns in (
        ("prompt_injection", _INJECTION),
        ("other_customer", _OTHER_CUSTOMERS),
        ("unsupported_action", _UNSUPPORTED),
    ):
        if any(re.search(p, lowered, re.IGNORECASE) for p in patterns):
            return ScreenResult(False, category)
    return ScreenResult(True)
