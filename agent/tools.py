"""
agent/tools.py

Florence — Literature RAG Agent
Tools the agent uses to interact with the outside world:
  - retrieve_chunks: semantic search against ChromaDB
  - get_parent_chunk: fetch a parent chunk by ID for richer context

Connections and models are loaded once at module level and reused.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# COLLECTION_NAME = "florence_literature"
# PARENT_COLLECTION_NAME = "florence_literature_parents"
COLLECTION_NAME = "florence_literature_dev" #use a different collection name for dev ingestion, so we don't mess with the main one while testing
PARENT_COLLECTION_NAME = "florence_literature_dev_parents" #use a different collection name for dev ingestion, so we don't mess with the main one while testing



# ── Shared resources — loaded once ─────────────────────────────────────────────
# You don't wanna load the model or connect to the db every time the agent calls a tool

print("[tools] Loading embedding model...")
_embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("[tools] Connecting to ChromaDB...")
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_collection(COLLECTION_NAME)
_parent_collection = _client.get_collection(PARENT_COLLECTION_NAME)

print(f"[tools] Ready. {_collection.count():,} chunks | {_parent_collection.count():,} parents")


# ── Tool 1: retrieve_chunks ────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search against the child chunk collection.

    Args:
        query: the search query
        top_k: number of chunks to return

    Returns:
        List of dicts, each containing:
        text, title, author, filename, parent_id, distance
    """
    query_embedding = _embedding_model.embed_query(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "title": meta.get("title", "Unknown"),
            "author": meta.get("author", "Unknown"),
            "filename": meta.get("filename", ""),
            "parent_id": meta.get("parent_id", ""),
            "distance": dist,
        })

    return chunks


# ── Tool 2: get_parent_chunk ───────────────────────────────────────────────────

def get_parent_chunk(parent_id: str) -> dict | None:
    """
    Fetch a parent chunk by its ID.
    Used after retrieval — child chunks are precise for search,
    parent chunks give the LLM full context for generation.

    Args:
        parent_id: the chunk_id of the parent

    Returns:
        Dict with text, title, author — or None if not found.
    """
    if not parent_id:
        return None

    try:
        result = _parent_collection.get(
            ids=[parent_id],
            include=["documents", "metadatas"],
        )

        if not result["documents"]:
            return None

        return {
            "text": result["documents"][0],
            "title": result["metadatas"][0].get("title", "Unknown"),
            "author": result["metadatas"][0].get("author", "Unknown"),
            "filename": result["metadatas"][0].get("filename", ""),
            "parent_id": parent_id,
        }

    except Exception as e:
        print(f"[tools] Error fetching parent chunk {parent_id}: {e}")
        return None


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test the retrieval tools directly.
    Usage: python agent/tools.py
    """
    test_query = "How does the Party control what people are able to think?"
    print(f"\nTest query: {test_query}")
    print("=" * 60)

    chunks = retrieve_chunks(test_query, top_k=3)

    for i, chunk in enumerate(chunks):
        print(f"\nResult {i+1}:")
        print(f"  Book: {chunk['title']} — {chunk['author']}")
        print(f"  Distance: {chunk['distance']:.4f}")
        print(f"  Text: {chunk['text'][:150]}...")

        # Test parent fetching
        if chunk["parent_id"]:
            parent = get_parent_chunk(chunk["parent_id"])
            if parent:
                print(f"  Parent found: {len(parent['text'])} chars")