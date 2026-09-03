"""
api/main.py

Florence — Literature RAG Agent
The FastAPI application. Turns the agent graph into an HTTP service.

Is all this necessary? Yes. No one is gonna run my github on their computer
just to screen me.

ENDPOINTS:
==========
  GET  /            → health check (is Florence alive?)
  POST /query       → ask Florence a question
  POST /resume      → resume a paused (low-confidence) answer after human review

HOW A REQUEST FLOWS:
====================
  1. Browser sends POST /query with a question
  2. FastAPI validates it against QueryRequest (schemas.py)
  3. We build the agent graph and invoke it
  4. If the graph completes → return the answer (status: complete)
  5. If the graph pauses for human review → save state, return thread_id (status: paused)
  6. If the domain router rejected it → return the rejection (status: rejected)
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from agent.graph import build_graph
from api.schemas import QueryRequest, ResumeRequest, QueryResponse, Source


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os as _os_for_static


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Florence — Literature RAG Agent",
    description="A grounded question-answering agent over a comparative-literature corpus.",
    version="1.0.0",
)

# CORS lets a browser frontend on a different origin call this API.
# For a public demo we allow all origins; tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the graph once at startup, not on every request.
# Loading the agent (embedding model, ChromaDB) is expensive — do it once.
print("[api] Building Florence agent graph...")
_graph = build_graph()
print("[api] Florence is open to visitors.")



# ── Helper: turn final graph state into a response ───────────────────────────

def _state_to_response(state: dict, thread_id: str) -> QueryResponse:
    """Convert the agent's final state into an API response."""
    # Domain router rejected the query
    if not state.get("domain_valid", True):
        return QueryResponse(
            status="rejected",
            reason=state.get("error", "Query is outside Florence's domain."),
        )

    sources = [
        Source(
            document=s.get("document", "Unknown"),
            author=s.get("author", "Unknown"),
            excerpt=s.get("excerpt", ""),
        )
        for s in state.get("sources", [])
    ]

    return QueryResponse(
        status="complete",
        answer=state.get("answer"),
        sources=sources,
        confidence=state.get("confidence", 0.0),
        faithfulness=state.get("hallucination_score", 0.0),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

_STATIC_DIR = _os_for_static.path.join(_os_for_static.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
def index():
    """Serve the frontend."""
    return FileResponse(_os_for_static.path.join(_STATIC_DIR, "Indexproper.html"))

@app.get("/health")
def health_check():
    """Is Florence alive? Used by load balancers and uptime checks."""
    return {"status": "ok", "service": "Florence — Literature RAG Agent"}




@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask Florence a question. Returns one of:
      - complete  → answer ready
      - paused    → low confidence, awaiting human review (use /resume with thread_id)
      - rejected  → query is outside Florence's domain
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"query": request.query}
    result = _graph.invoke(initial_state, config=config)

    # Did the graph pause for human review?
    snapshot = _graph.get_state(config)
    if snapshot.next:  # pending nodes = paused
        return QueryResponse(
            status="paused",
            answer=result.get("answer"),
            confidence=result.get("confidence", 0.0),
            faithfulness=result.get("hallucination_score", 0.0),
            thread_id=thread_id,
            reason=result.get("hitl_reason", "Low-confidence answer — human review requested."),
        )

    return _state_to_response(result, thread_id)



@app.post("/resume", response_model=QueryResponse)
def resume(request: ResumeRequest):
    """
    Resume a paused answer after human review.
    'approve' accepts Florence's answer; any other text replaces it.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    # Resume the graph from where it paused, feeding in the human decision.
    result = _graph.invoke(Command(resume=request.decision), config=config)

    return _state_to_response(result, request.thread_id)