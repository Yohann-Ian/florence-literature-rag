"""
agent/guardrails.py

Florence — Literature RAG Agent
Security guardrails:
  - check_prompt_injection: screens incoming queries
  - check_output_safety: screens outgoing answers

These are deliberately simple pattern-based checks.
They are the first line of defence, not the only one —
the domain router and Claude's own safety training provide additional layers.
"""

import re


# ── Prompt injection patterns ──────────────────────────────────────────────────
# Common injection phrasings. Lowercase — input is lowercased before matching.

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore your previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard your instructions",
    "disregard previous",
    "forget your instructions",
    "forget everything above",
    "system prompt",
    "your instructions are",
    "you are now",
    "act as if",
    "pretend you are",
    "jailbreak",
    "developer mode",
    "dan mode",
    "output your prompt",
    "reveal your prompt",
    "show me your instructions",
    "what are your instructions",
    "repeat your system message",
    "print your system message",
]


def check_prompt_injection(query: str) -> bool:
    """
    Returns True if the query appears to contain a prompt injection attempt.

    Checks against known injection patterns. Simple but effective
    against the most common attacks. Claude's own training handles
    more sophisticated attempts.
    """
    query_lower = query.lower()

    for pattern in INJECTION_PATTERNS:
        if pattern in query_lower:
            return True

    # Check for attempts to use special formatting to confuse the model
    suspicious_formats = [
        r"<\s*system\s*>",        # <system> tags
        r"\[\s*system\s*\]",      # [system] tags
        r"#{3,}\s*instruction",   # ### instruction headers
    ]

    for pattern in suspicious_formats:
        if re.search(pattern, query_lower):
            return True

    return False


# ── Output safety ──────────────────────────────────────────────────────────────

UNSAFE_OUTPUT_PATTERNS = [
    # Florence should never output her own system internals
    "my system prompt",
    "my instructions are",
    # Florence should never generate content impersonating system messages
    "<system>",
    "[system]",
]


def check_output_safety(answer: str) -> bool:
    """
    Returns True if the answer is safe to send to the user.
    Returns False if the answer contains unsafe content.

    Checks that the answer doesn't leak system internals
    or contain obviously malformed content.
    """
    if not answer or len(answer.strip()) == 0:
        return False

    answer_lower = answer.lower()

    for pattern in UNSAFE_OUTPUT_PATTERNS:
        if pattern in answer_lower:
            return False

    return True