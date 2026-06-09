"""
agent/graph.py

Florence — Literature RAG Agent
The LangGraph state machine. Defines the full agent graph:
nodes, edges, conditional routing, and the HITL checkpoint.

GRAPH FLOW:
===========

                    ┌─────────────────┐
                    │  domain_router  │
                    └────────┬────────┘
                             │
                    domain_valid?
                    │        │
                   NO       YES
                    │        │
                  END    ┌───▼────┐
                         │retrieve│◄──────────────┐
                         └───┬────┘               │
                             │                    │
                    ┌────────▼────────┐           │
                    │ grade_documents │           │
                    └────────┬────────┘           │
                             │                    │
                    enough relevant?              │
                    │        │                    │
                   NO       YES           ┌───────┴──────┐
                    │        │            │ rewrite_query │
                    └────────┼────────────┘              │
                    retry    │            (max 2 retries) │
                    count?───┘                           │
                             │                           │
                    ┌────────▼────────┐                  │
                    │    generate     │◄─────────────────┐
                    └────────┬────────┘                  │
                             │                           │
                    ┌────────▼────────────┐              │
                    │ check_hallucination │              │
                    └────────┬────────────┘              │
                             │                           │
                    grounded?                            │
                    │        │                           │
                   NO       YES                         │
                    │        │                           │
                    └────────┼───────────────────────────┘
                    retry    │
                    once     │
                             │
                    ┌────────▼────────┐
                    │ hitl_checkpoint │
                    └────────┬────────┘
                             │
                    confident?
                    │        │
                   NO       YES
                    │        │
                 interrupt  END
                 (human     (return
                 review)     answer)
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
# This is Florence's working memory for a single query.
# Every node reads from this and writes back to it.
# Think of it as a baton passed between runners in a relay race —
# each runner (node) carries it, adds something to it, and hands it on.

class AgentState(TypedDict):
    # ── Input ──
    query: str                          # original user query
    rewritten_query: Optional[str]      # query after rewriting, if needed

    # ── Retrieval ──
    retrieved_docs: list                # chunks retrieved from ChromaDB
    doc_grades: list                    # relevance grade per chunk (True/False)
    relevant_docs: list                 # only the chunks that passed grading

    # ── Generation ──
    answer: Optional[str]               # Florence's generated answer
    sources: list                       # citations with title, author, excerpt

    # ── Quality control ──
    hallucination_score: float          # 0.0-1.0, how grounded the answer is
    confidence: float                   # 0.0-1.0, overall answer confidence
    domain_valid: bool                  # is the query within the literary corpus?

    # ── Loop control ──
    retry_count: int                    # tracks retrieval retries (max 2)
    generation_attempts: int            # tracks generation retries (max 1)

    # ── Metadata ──
    trace_id: Optional[str]             # LangSmith run ID for observability
    error: Optional[str]               # error message if something went wrong


# ── Signal constants ───────────────────────────────────────────────────────────
# These are the signal strings returned by routing functions.
# SCREAMING_SNAKE_CASE distinguishes them from node names (lower_snake_case).
# Each signal is a decision label — not a node name.
# The mapping dictionary in add_conditional_edges translates signals → nodes.

VALID_QUERY      = "VALID_QUERY"      # query is in domain → prep for "retrieve" node next
INVALID_QUERY    = "INVALID_QUERY"    # query is out of domain → END

SUFFICIENT       = "SUFFICIENT"       # enough relevant docs → prep for "generate" node next
INSUFFICIENT     = "INSUFFICIENT"     # not enough docs, retries remain → prep for "rewrite_query" node next
EXHAUSTED        = "EXHAUSTED"        # not enough docs, retries gone → END

GROUNDED         = "GROUNDED"         # answer is well-grounded → prep for "hitl_checkpoint" node next
UNGROUNDED       = "UNGROUNDED"       # answer not grounded, retry → prep for "generate" node next
UNGROUNDED_FINAL = "UNGROUNDED_FINAL" # answer not grounded, no retries → prep for "hitl_checkpoint" node next

CONFIDENT        = "CONFIDENT"        # confidence high → END
LOW_CONFIDENCE   = "LOW_CONFIDENCE"   # confidence low → interrupt for human review

# ── Routing functions ──────────────────────────────────────────────────────────
# Each function reads the current state and returns a signal string.
# LangGraph calls these automatically after the corresponding node finishes.
# They never modify state — they only read it and return a signal.

def route_after_domain_check(state: AgentState) -> str:
    """
    Called after domain_router finishes.
    Returns VALID_QUERY or INVALID_QUERY.
    """
    if not state.get("domain_valid", False):
        return INVALID_QUERY
    return VALID_QUERY


def route_after_grading(state: AgentState) -> str:
    """
    Called after grade_documents finishes.
    Returns SUFFICIENT, INSUFFICIENT, or EXHAUSTED.
    """
    relevant = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if len(relevant) >= 2:
        return SUFFICIENT
    elif retry_count < 2:
        return INSUFFICIENT
    else:
        return EXHAUSTED


def route_after_hallucination_check(state: AgentState) -> str:
    """
    Called after check_hallucination finishes.
    Returns GROUNDED, UNGROUNDED, or UNGROUNDED_FINAL.
    """
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
    Called after hitl_checkpoint finishes.
    Returns CONFIDENT or LOW_CONFIDENCE.
    Note: the actual interrupt() call happens inside hitl_checkpoint node.
    This router handles post-interrupt routing.
    """
    confidence = state.get("confidence", 0.0)

    if confidence >= 0.7:
        return CONFIDENT
    else:
        return LOW_CONFIDENCE


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct the Florence agent graph.
    Node logic lives in nodes.py.
    Returns a compiled graph ready to run.
    """
    # Import node functions here to avoid circular imports
    from agent.nodes import (
        domain_router,
        retrieve,
        grade_documents,
        rewrite_query,
        generate,
        check_hallucination,
        hitl_checkpoint,
    )

    # Initialise the graph with our state schema
    workflow = StateGraph(AgentState)


    # ── Add nodes ──────────────────────────────────────────────────────────────
    # Syntax: workflow.add_node("node_name", function)
    # "node_name" is how this node is referenced in edges.
    # function is what runs when this node is visited.

    workflow.add_node("domain_router",       domain_router)
    workflow.add_node("retrieve",            retrieve)
    workflow.add_node("grade_documents",     grade_documents)
    workflow.add_node("rewrite_query",       rewrite_query)
    workflow.add_node("generate",            generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("hitl_checkpoint",     hitl_checkpoint)


    # ── Set entry point ────────────────────────────────────────────────────────
    # Every query enters the graph at this node.

    workflow.set_entry_point("domain_router")


    # ── Add edges ──────────────────────────────────────────────────────────────
    #
    # SIMPLE EDGES (no decision):
    # workflow.add_edge("A", "B")
    # After A finishes → always run B. No routing function. No signals.
    #
    # CONDITIONAL EDGES (decision required):
    # workflow.add_conditional_edges("A", routing_fn, { SIGNAL: "node" })
    # After A finishes → call routing_fn(state) → get signal → look up node.
    #
    # Again, signal legend for this graph:
    #   VALID_QUERY      → "retrieve"
    #   INVALID_QUERY    → END
    #   SUFFICIENT       → "generate"
    #   INSUFFICIENT     → "rewrite_query"
    #   EXHAUSTED        → END
    #   GROUNDED         → "hitl_checkpoint"
    #   UNGROUNDED       → "generate"
    #   UNGROUNDED_FINAL → "hitl_checkpoint"
    #   CONFIDENT        → END
    #   LOW_CONFIDENCE   → END (after human approval via interrupt)

    # domain_router → VALID_QUERY: retrieve | INVALID_QUERY: END
    workflow.add_conditional_edges(
        "domain_router",
        route_after_domain_check,
        {
            VALID_QUERY:   "retrieve",
            INVALID_QUERY: END,
        }
    )

    # retrieve → always → grade_documents
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

    # rewrite_query → always → retrieve (the retry loop)
    workflow.add_edge("rewrite_query", "retrieve")

    # generate → always → check_hallucination
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

    # hitl_checkpoint → CONFIDENT: END | LOW_CONFIDENCE: END (after human approval)
    workflow.add_conditional_edges(
        "hitl_checkpoint",
        route_after_hitl,
        {
            CONFIDENT:      END,
            LOW_CONFIDENCE: END,
        }
    )


    # ── Compile with memory ────────────────────────────────────────────────────
    # MemorySaver persists state between the interrupt() call in hitl_checkpoint
    # and the human's response — so the graph can resume exactly where it paused.

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Build the graph and confirm it compiles without errors.
    Does not run a query — just validates the structure.
    Usage: python agent/graph.py
    """
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