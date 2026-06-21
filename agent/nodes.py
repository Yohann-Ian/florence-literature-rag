"""
agent/nodes.py

Florence — Literature RAG Agent
Node functions for the LangGraph state machine.

Each node:
  - Takes the full AgentState as input
  - Does exactly one job
  - Returns a dictionary of only the fields it changed
  - Never modifies fields it doesn't own

NODE RESPONSIBILITIES:
======================
domain_router       → sets domain_valid
retrieve            → sets retrieved_docs
grade_documents     → sets doc_grades, relevant_docs, retry_count
rewrite_query       → sets rewritten_query, retry_count
generate            → sets answer, sources, generation_attempts
check_hallucination → sets hallucination_score, confidence
hitl_checkpoint     → calls interrupt() if confidence too low
"""

import os
import sys
import json
from typing import Any
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

from agent.graph import AgentState
from agent.guardrails import check_prompt_injection, check_output_safety
from agent.tools import retrieve_chunks, get_parent_chunk

import chromadb
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

# ── Shared resources ───────────────────────────────────────────────────────────
# Loaded once when the module imports — not on every node call.
# This avoids reloading the embedding model on every single query.

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "florence_literature"
PARENT_COLLECTION_NAME = "florence_literature_parents"
TOP_K = int(os.getenv("TOP_K", "5"))
CONFIDENCE_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 0.7

# Literary domain keywords — used for lightweight domain checking
LITERARY_KEYWORDS = [
    "novel", "book", "author", "character", "theme", "narrative",
    "plot", "prose", "poem", "poetry", "epic", "tragedy", "symbol",
    "metaphor", "motif", "aeneid", "paradise lost", "milton", "virgil",
    "dostoyevsky", "tolstoy", "hemingway", "orwell", "huxley", "morrison",
    "steinbeck", "bulgakov", "raskolnikov", "winston", "anna karenina",
    "beloved", "nineteen eighty", "brave new world", "east of eden",
    "master and margarita", "farewell to arms", "crime and punishment",
    "nabokov", "forster", "guilt", "redemption", "duty", "empire",
    "satan", "aeneas", "dystopia", "totalitarian", "war", "love",
    "death", "suffering", "freedom", "soul", "god", "evil", "fate",
]

def _load_embedding_model() -> HuggingFaceEmbeddings:
    """Load the embedding model once and cache it."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _load_chroma_collection() -> chromadb.Collection:
    """Connect to the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)

def _load_llm() -> ChatAnthropic:
    """Load the Anthropic Claude model."""
    return ChatAnthropic(
        model="claude-opus-4-5",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
        temperature=0.3,
    )


def _load_grader_llm() -> ChatAnthropic:
    """Fast, cheap model for yes/no relevance grading.
    Haiku is plenty for a binary judgement and ~10x cheaper/faster than Opus."""
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=10,        # only needs to say YES or NO
        temperature=0.0,
    )


# ── Node 1: domain_router ──────────────────────────────────────────────────────

def domain_router(state: AgentState) -> dict:
    """
    Check whether the query is within Florence's literary domain.
    Also checks for prompt injection attempts.

    Writes: domain_valid, error
    """
    query = state["query"]

    print(f"\n[domain_router] Query: {query[:80]}...")

    # Check for prompt injection first
    if check_prompt_injection(query):
        print("[domain_router] Prompt injection detected — rejecting.")
        return {
            "domain_valid": False,
            "error": "Query rejected: prompt injection detected.",
        }

    # Lightweight domain check — does the query mention literary topics?
    query_lower = query.lower()
    keyword_match = any(kw in query_lower for kw in LITERARY_KEYWORDS)

    # Also check query length — very short queries are often out of domain
    long_enough = len(query.split()) >= 4

    domain_valid = keyword_match and long_enough

    if not domain_valid:
        print("[domain_router] Query appears out of literary domain — rejecting.")
        return {
            "domain_valid": False,
            "error": "Query is outside Florence's domain. Please ask about the literary corpus.",
        }

    print("[domain_router] Query is valid — proceeding to retrieval.")
    return {
        "domain_valid": True,
        "error": None,
        "retry_count": 0,
        "generation_attempts": 0,
        "retrieved_docs": [],
        "relevant_docs": [],
        "doc_grades": [],
        "sources": [],
        "answer": None,
        "hallucination_score": 0.0,
        "confidence": 0.0,
    }


# ── Node 2: retrieve ───────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB using semantic search.
    Uses the rewritten query if available, otherwise the original.

    Writes: retrieved_docs
    """
    # Use rewritten query if available
    query = state.get("rewritten_query") or state["query"]
    print(f"\n[retrieve] Searching with query: {query[:80]}...")

    try:
        docs = retrieve_chunks(query, top_k=TOP_K)
        print(f"[retrieve] Retrieved {len(docs)} chunks.")
        return {"retrieved_docs": docs}
    except Exception as e:
        print(f"[retrieve] Error during retrieval: {e}")
        return {"retrieved_docs": [], "error": str(e)}


# ── Node 3: grade_documents ────────────────────────────────────────────────────

def grade_documents(state: AgentState) -> dict:
    """
    Grade each retrieved chunk for relevance to the query.
    Runs all grading calls concurrently (thread pool) with a fast model.

    Writes: doc_grades, relevant_docs, retry_count
    """
    from concurrent.futures import ThreadPoolExecutor

    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    print(f"\n[grade_documents] Grading {len(docs)} chunks in parallel...")

    llm = _load_grader_llm()

    def grade_one(doc: dict) -> bool:
        """Grade a single chunk. Returns True if relevant."""
        grading_prompt = f"""You are grading whether a retrieved text chunk is relevant to a literary query.

Query: {query}

Retrieved chunk (from {doc.get('title', 'unknown')} by {doc.get('author', 'unknown')}):
{doc.get('text', '')[:500]}

Is this chunk relevant to answering the query?
Respond with only: YES or NO"""
        try:
            response = llm.invoke([HumanMessage(content=grading_prompt)])
            return "YES" in response.content.upper()
        except Exception as e:
            print(f"[grade_documents] grading error: {e} — treating as not relevant")
            return False

    # Fire all grading calls concurrently
    with ThreadPoolExecutor(max_workers=len(docs) or 1) as executor:
        grades = list(executor.map(grade_one, docs))

    relevant_docs = [doc for doc, keep in zip(docs, grades) if keep]

    for i, (doc, keep) in enumerate(zip(docs, grades)):
        status = "RELEVANT" if keep else "NOT RELEVANT"
        print(f"  Chunk {i+1}: {status} — {doc.get('title', 'unknown')}")

    print(f"[grade_documents] {len(relevant_docs)}/{len(docs)} chunks passed grading.")

    return {
        "doc_grades": grades,
        "relevant_docs": relevant_docs,
        "retry_count": retry_count + 1,
    }


# ── Node 4: rewrite_query ──────────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:
    """
    Rewrite the query to improve retrieval on the next attempt.
    Makes the query more specific and better suited to the corpus.

    Writes: rewritten_query
    """
    original_query = state["query"]
    previous_rewrite = state.get("rewritten_query")
    retry_count = state.get("retry_count", 0)

    print(f"\n[rewrite_query] Rewriting query (attempt {retry_count})...")

    llm = _load_llm()

    rewrite_prompt = f"""You are helping improve a search query for a literary corpus containing:
The Aeneid, Paradise Lost, Crime and Punishment, Anna Karenina, The Master and Margarita,
A Farewell to Arms, Nineteen Eighty-Four, Brave New World, East of Eden, Beloved,
Lectures on Russian Literature (Nabokov), and Aspects of the Novel (Forster).

Original query: {original_query}
{f'Previous rewrite (did not find enough results): {previous_rewrite}' if previous_rewrite else ''}

Rewrite the query to be more specific and likely to match relevant passages.
Focus on character names, themes, specific events, or literary concepts.
Return only the rewritten query....nothing else."""

    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten = response.content.strip()

    print(f"[rewrite_query] Rewritten: {rewritten[:80]}...")

    return {"rewritten_query": rewritten}


# ── Node 5: generate ───────────────────────────────────────────────────────────

def generate(state: AgentState) -> dict:
    """
    Generate an answer from the relevant chunks using Claude.
    Includes source citations in the response.

    Writes: answer, sources, generation_attempts
    """
    query = state.get("rewritten_query") or state["query"]
    relevant_docs = state.get("relevant_docs", [])
    generation_attempts = state.get("generation_attempts", 0)

    print(f"\n[generate] Generating answer (attempt {generation_attempts + 1})...")

    # Fetch parent chunks for richer context
    enriched_docs = []
    for doc in relevant_docs:
        parent_id = doc.get("parent_id")
        if parent_id:
            parent = get_parent_chunk(parent_id)
            if parent:
                enriched_docs.append(parent)
            else:
                enriched_docs.append(doc)
        else:
            enriched_docs.append(doc)

    # Build context string from enriched docs
    context_parts = []
    for i, doc in enumerate(enriched_docs):
        context_parts.append(
            f"[Source {i+1}] {doc.get('title', 'Unknown')} by {doc.get('author', 'Unknown')}:\n"
            f"{doc.get('text', '')[:1000]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build sources list for citations
    sources = []
    for doc in enriched_docs:
        sources.append({
            "document": doc.get("title", "Unknown"),
            "author": doc.get("author", "Unknown"),
            "excerpt": doc.get("text", "")[:200],
            "relevance_score": doc.get("distance", 0.0),
        })

    system_prompt = """You are Florence, a literary analysis agent specialising in comparative literature.
You have access to a corpus of twelve major literary works spanning antiquity to the twentieth century.

Your role is to answer literary questions with intellectual rigour and genuine engagement.
You reason across texts, identify connections, and construct interpretive arguments supported by textual evidence.
You also mustn't be afraid to express an idea even if it's dangerous in the 21st century politically correct
climate.

Rules:
- Only make claims supported by the provided context
- Cite specific passages or details from the sources
- Acknowledge genuine ambiguity — do not collapse tensions the texts preserve
- If the context is insufficient to answer fully, say so clearly
- Write in clear, precise prose — not bullet points"""

    user_prompt = f"""Using the following passages from the literary corpus, answer this question:

{query}

CORPUS PASSAGES:
{context}

Provide a thoughtful, well-supported answer. Reference specific details from the passages.
End your answer with a brief note on which sources were most relevant."""

    llm = _load_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    answer = response.content

    # Check output safety
    if not check_output_safety(answer):
        answer = "Florence was unable to generate a safe response for this query."

    print(f"[generate] Answer generated ({len(answer)} characters).")

    return {
        "answer": answer,
        "sources": sources,
        "generation_attempts": generation_attempts + 1,
    }


# ── Node 6: check_hallucination ────────────────────────────────────────────────

def check_hallucination(state: AgentState) -> dict:
    """
    Check whether the generated answer is grounded in the retrieved context.
    Uses Claude to score how well the answer is supported by the sources.

    Writes: hallucination_score, confidence
    """
    answer = state.get("answer", "")
    relevant_docs = state.get("relevant_docs", [])
    query = state["query"]

    print(f"\n[check_hallucination] Checking answer grounding...")

    if not answer or not relevant_docs:
        return {"hallucination_score": 0.0, "confidence": 0.0}

    # Build context for checking
    context = "\n\n".join([
        f"{doc.get('title', 'Unknown')}: {doc.get('text', '')[:500]}"
        for doc in relevant_docs
    ])

    llm = _load_llm()

    grounding_prompt = f"""You are evaluating whether a literary analysis answer is grounded in provided source texts.

Question: {query}

Source texts:
{context}

Generated answer:
{answer}

Evaluate the answer on two dimensions:

1. FAITHFULNESS (0.0-1.0): Are the claims in the answer supported by the source texts?
   - 1.0: Every claim is directly supported
   - 0.7: Most claims are supported, minor extrapolation
   - 0.5: Some claims are supported, some are not
   - 0.0: Claims contradict or ignore the sources

2. CONFIDENCE (0.0-1.0): How confident are you in the overall quality of this answer?
   - Consider: does it actually answer the question? Is the reasoning sound?

Respond in this exact JSON format:
{{
    "faithfulness": 0.0,
    "confidence": 0.0,
    "reasoning": "brief explanation"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=grounding_prompt)])
        # Parse JSON from response
        content = response.content.strip()
        # Find JSON in response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            scores = json.loads(content[start:end])
            hallucination_score = float(scores.get("faithfulness", 0.5))
            confidence = float(scores.get("confidence", 0.5))
            reasoning = scores.get("reasoning", "")
            print(f"[check_hallucination] Faithfulness: {hallucination_score:.2f} | Confidence: {confidence:.2f}")
            print(f"[check_hallucination] Reasoning: {reasoning}")
        else:
            hallucination_score = 0.5
            confidence = 0.5
            print("[check_hallucination] Could not parse scores — defaulting to 0.5")

    except Exception as e:
        print(f"[check_hallucination] Error: {e} — defaulting to 0.5")
        hallucination_score = 0.5
        confidence = 0.5

    return {
        "hallucination_score": hallucination_score,
        "confidence": confidence,
    }


# ── Node 7: hitl_checkpoint ────────────────────────────────────────────────────

def hitl_checkpoint(state: AgentState) -> dict:
    """
    Human-in-the-loop checkpoint.
    If confidence is below threshold, pause and wait for human review.
    Uses LangGraph's interrupt() to pause execution mid-graph.

    If confidence is high enough, passes through without interrupting.
    """
    confidence = state.get("confidence", 0.0)
    answer = state.get("answer", "")
    query = state["query"]

    print(f"\n[hitl_checkpoint] Confidence: {confidence:.2f} (threshold: {CONFIDENCE_THRESHOLD})")

    if confidence < CONFIDENCE_THRESHOLD:
        print("[hitl_checkpoint] Low confidence — requesting human review...")

        # interrupt() pauses the graph here.
        # The value passed to interrupt() is shown to the human reviewer.
        # When the human resumes the graph, execution continues from this point.
        human_decision = interrupt({
            "reason": "Low confidence answer requires human review",
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "instruction": "Review the answer above. Type 'approve' to send it, or provide a corrected answer.",
        })

        # If human provided a corrected answer, use it
        if isinstance(human_decision, str) and human_decision.lower() != "approve":
            print("[hitl_checkpoint] Human provided corrected answer.")
            return {"answer": human_decision, "confidence": 1.0}

        print("[hitl_checkpoint] Human approved the answer.")
        return {"confidence": CONFIDENCE_THRESHOLD}

    print("[hitl_checkpoint] Confidence sufficient — passing through.")
    return {}