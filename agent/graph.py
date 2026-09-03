"""
agent/graph.py

Florence — Literature RAG Agent
The LangGraph state machine. Defines the full agent graph:
nodes, edges, conditional routing, and the HITL checkpoint.
"""
import os
import sys
from typing import TypedDict, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    query: str
    rewritten_query: Optional[str]

    # Retrieval
    retrieved_docs: list
    doc_grades: list
    relevant_docs: list

    # Generation
    answer: Optional[str]
    sources: list
    user_feedback: Optional[str]
    rejection_count: int

    # Quality control
    hallucination_score: float
    confidence: float
    domain_valid: bool

    # Loop control
    retry_count: int
    generation_attempts: int

    # Metadata
    trace_id: Optional[str]
    error: Optional[str]


# ── Signal constants ───────────────────────────────────────────────────────────

VALID_QUERY      = "VALID_QUERY"
INVALID_QUERY    = "INVALID_QUERY"

SUFFICIENT       = "SUFFICIENT"
INSUFFICIENT     = "INSUFFICIENT"
EXHAUSTED        = "EXHAUSTED"

GROUNDED         = "GROUNDED"
UNGROUNDED       = "UNGROUNDED"
UNGROUNDED_FINAL = "UNGROUNDED_FINAL"


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_domain_check(state: AgentState) -> str:
    if not state.get("domain_valid", False):
        return INVALID_QUERY
    return VALID_QUERY


def route_after_grading(state: AgentState) -> str:
    relevant = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if len(relevant) >= 2:
        return SUFFICIENT
    elif retry_count < 2:
        return INSUFFICIENT
    else:
        return EXHAUSTED


def route_after_hallucination_check(state: AgentState) -> str:
    score = state.get("hallucination_score", 0.0)
    attempts = state.get("generation_attempts", 1)

    if score >= 0.7:
        return GROUNDED
    elif attempts < 2:
        return UNGROUNDED
    else:
        return UNGROUNDED_FINAL


def route_after_hitl(state: AgentState) -> str:
    """
    After hitl_checkpoint: if user rejected with feedback, loop back to generate.
    Otherwise (approved or didn't need review), end the graph.
    """
    if state.get("user_feedback"):
        return "generate"
    return END


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    from agent.nodes import (
        domain_router,
        retrieve,
        grade_documents,
        rewrite_query,
        generate,
        check_hallucination,
        hitl_checkpoint,
    )

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("domain_router",       domain_router)
    workflow.add_node("retrieve",            retrieve)
    workflow.add_node("grade_documents",     grade_documents)
    workflow.add_node("rewrite_query",       rewrite_query)
    workflow.add_node("generate",            generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("hitl_checkpoint",     hitl_checkpoint)

    # Entry point
    workflow.set_entry_point("domain_router")

    # domain_router → VALID_QUERY: retrieve | INVALID_QUERY: END
    workflow.add_conditional_edges(
        "domain_router",
        route_after_domain_check,
        {
            VALID_QUERY:   "retrieve",
            INVALID_QUERY: END,
        }
    )

    # retrieve → grade_documents
    workflow.add_edge("retrieve", "grade_documents")

    # grade_documents → SUFFICIENT: generate | INSUFFICIENT: rewrite_query | EXHAUSTED: END
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            SUFFICIENT:   "generate",
            INSUFFICIENT: "rewrite_query",
            EXHAUSTED:    END,
        }
    )

    # rewrite_query → retrieve (retry loop)
    workflow.add_edge("rewrite_query", "retrieve")

    # generate → check_hallucination
    workflow.add_edge("generate", "check_hallucination")

    # check_hallucination → GROUNDED: hitl_checkpoint | UNGROUNDED: generate | UNGROUNDED_FINAL: hitl_checkpoint
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            GROUNDED:         "hitl_checkpoint",
            UNGROUNDED:       "generate",
            UNGROUNDED_FINAL: "hitl_checkpoint",
        }
    )

    # hitl_checkpoint → if user rejected with feedback, loop back to generate; else END
    workflow.add_conditional_edges(
        "hitl_checkpoint",
        route_after_hitl,
        {
            "generate": "generate",
            END:        END,
        }
    )

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building Florence agent graph...")
    try:
        graph = build_graph()
        print("Graph compiled successfully.")
        print("\nNodes:")
        for node in graph.nodes:
            print(f"  - {node}")
        print("\nThe City of Florence is ready to work.")
    except Exception as e:
        print(f"Graph compilation failed: {e}")
        raise