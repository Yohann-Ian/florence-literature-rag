"""
run_florence.py

Florence — Literature RAG Agent
Simple command-line runner to test the full agent graph end to end.
Usage: python run_florence.py
"""

import os
import sys
import uuid
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv(override=True)

from agent.graph import build_graph


def run_query(query: str):
    """Run a single query through the full Florence agent graph."""
    print("=" * 70)
    print("FLORENCE — Literature RAG Agent")
    print("=" * 70)
    print(f"\nQuery: {query}\n")

    graph = build_graph()

    # Each run needs a thread_id for the checkpointer (enables HITL)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Initial state — only the query is set, nodes fill in the rest
    initial_state = {
        "query": query,
        "rewritten_query": None,
        "retrieved_docs": [],
        "doc_grades": [],
        "relevant_docs": [],
        "answer": None,
        "sources": [],
        "hallucination_score": 0.0,
        "confidence": 0.0,
        "domain_valid": False,
        "retry_count": 0,
        "generation_attempts": 0,
        "trace_id": None,
        "error": None,
    }

    # Run the graph
    final_state = graph.invoke(initial_state, config=config)

    # Print the result
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    if final_state.get("error"):
        print(f"\nError: {final_state['error']}")
        return

    answer = final_state.get("answer")
    if answer:
        print(f"\nAnswer:\n{answer}\n")
        print(f"Confidence: {final_state.get('confidence', 0.0):.2f}")
        print(f"Faithfulness: {final_state.get('hallucination_score', 0.0):.2f}")

        sources = final_state.get("sources", [])
        if sources:
            print(f"\nSources ({len(sources)}):")
            for i, src in enumerate(sources):
                print(f"  {i+1}. {src['document']} — {src['author']}")
    else:
        print("\nNo answer generated.")


if __name__ == "__main__":
    # Start with a simple query we know should work
    test_query = "How does the Party in Nineteen Eighty-Four control what people think?"
    run_query(test_query)