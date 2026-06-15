"""
api/schemas.py

Florence — Literature RAG Agent
Pydantic models defining the request and response shapes for the API.

Pydantic validates incoming requests automatically: if a request doesn't
match these shapes, FastAPI rejects it with a clear error before the
request ever reaches Florence's agent logic.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request shapes ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """A question coming in to Florence."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The literary question to ask Florence.",
        examples=["How does the Party in Nineteen Eighty-Four control thought?"],
    )


class ResumeRequest(BaseModel):
    """A human decision resuming a paused (low-confidence) answer."""
    thread_id: str = Field(
        ...,
        description="The thread ID returned when the query was paused.",
    )
    decision: str = Field(
        ...,
        description="'approve' to accept Florence's answer, or a corrected answer to use instead.",
    )


# ── Response shapes ──────────────────────────────────────────────────────────

class Source(BaseModel):
    """A single cited source in an answer."""
    document: str
    author: str
    excerpt: str


class QueryResponse(BaseModel):
    """Florence's answer to a question."""
    status: str = Field(description="'complete', 'paused', or 'rejected'.")
    answer: Optional[str] = None
    sources: list[Source] = []
    confidence: float = 0.0
    faithfulness: float = 0.0
    # When status is 'paused', this lets the caller resume after human review
    thread_id: Optional[str] = None
    # When status is 'rejected', this explains why
    reason: Optional[str] = None