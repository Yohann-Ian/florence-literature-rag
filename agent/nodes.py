"""
agent/nodes.py

Florence — Literature RAG Agent
Node functions for the LangGraph state machine.
Each function takes AgentState and returns updated AgentState.
Stubs only — full logic added in next stage.
"""

from agent.graph import AgentState


def domain_router(state: AgentState) -> AgentState:
    """Check if query is within the literary corpus domain."""
    return state


def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from ChromaDB using hybrid search."""
    return state


def grade_documents(state: AgentState) -> AgentState:
    """Grade each retrieved chunk for relevance to the query."""
    return state


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the query to improve retrieval on next attempt."""
    return state


def generate(state: AgentState) -> AgentState:
    """Generate an answer from the relevant chunks."""
    return state


def check_hallucination(state: AgentState) -> AgentState:
    """Check whether the answer is grounded in retrieved context."""
    return state


def hitl_checkpoint(state: AgentState) -> AgentState:
    """Pause for human review if confidence is below threshold."""
    return state