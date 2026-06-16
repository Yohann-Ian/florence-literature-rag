"""
ingestion/embed_index.py

Florence — Literature RAG Agent
Embeds chunks and indexes them into ChromaDB.

This is the final ingestion step. It:
1. Loads and parses the corpus
2. Chunks using the selected strategy
3. Embeds chunks using a local HuggingFace model
4. Stores everything in ChromaDB with full metadata

Run once to build the index. After that, the agent queries ChromaDB directly.

CODE FLOW:
==========

INPUT: Books/ folder containing EPUB and PDF files
OUTPUT: ChromaDB collection stored in chroma_db/ folder

FLOW:
    1. parse_corpus()
       └── reads all 12 books into ParsedDocument objects

    2. chunk_corpus()
       └── splits documents using selected strategy
       └── for hierarchical: stores parents separately

    3. embed_chunks()
       └── batches chunks into groups of 100
       └── converts each chunk text into a vector
       └── uses local HuggingFace model (no API cost)

    4. index_chunks()
       └── stores vectors + text + metadata in ChromaDB
       └── each chunk gets: id, embedding, text, metadata
       └── metadata carries title, author, strategy, parent_id

    5. verify_index()
       └── runs a test query against the index
       └── confirms retrieval is working before agent uses it

OUTPUT: chroma_db/ folder ready for agent queries
"""

import os
import sys
import json
import time
from typing import Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.parse import parse_corpus, ParsedDocument
from ingestion.chunk import chunk_corpus, chunk_hierarchical, Chunk

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

# BOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Books")
BOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Books_dev")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"


BATCH_SIZE = 100


# ── Embedding ──────────────────────────────────────────────────────────────────

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the local HuggingFace embedding model.
    Using a local model avoids API costs during ingestion....it's expensive you know.
    all-MiniLM-L6-v2 is "fast, small, and good enough for literary text retrieval" as they say.
    """
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding model loaded.")
    return model


def embed_chunks(
    chunks: list[Chunk],
    model: HuggingFaceEmbeddings,
) -> list[list[float]]:
    """
    Convert chunk texts into vectors using the embedding model.
    Processes in batches of BATCH_SIZE to avoid memory issues.
    Returns a list of embeddings in the same order as input chunks.
    """
    print(f"\nEmbedding {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
    all_embeddings = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_texts = [c.text for c in batch]
        batch_num = (i // BATCH_SIZE) + 1

        embeddings = model.embed_documents(batch_texts)
        all_embeddings.extend(embeddings)

        # Progress indicator
        print(f"  Batch {batch_num}/{total_batches} — {len(all_embeddings):,} chunks embedded")

    print(f"Embedding complete. {len(all_embeddings):,} vectors produced.")
    return all_embeddings


# ── ChromaDB indexing ──────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.
    Data is stored in chroma_db/ folder at the project root.
    Persistent means the index survives between runs — we only build it once.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client


def index_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection_name: str,
    client: chromadb.PersistentClient,
    parent_chunks: Optional[list[Chunk]] = None,
) -> chromadb.Collection:
    """
    Store chunks and their embeddings in ChromaDB.
    Each chunk is stored with:
      - id: unique chunk_id
      - embedding: the vector representation
      - document: the raw text (ChromaDB stores this for retrieval)
      - metadata: title, author, strategy, position, parent_id

    For hierarchical strategy, parent chunks are stored in a separate
    collection so the agent can fetch them by parent_id.
    """
    print(f"\nIndexing into ChromaDB collection: '{collection_name}'")

    # Delete existing collection if it exists — clean rebuild
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )

    # Index in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c.text for c in batch_chunks],
            metadatas=[{
                "title": c.title,
                "author": c.author,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "strategy": c.strategy,
                "parent_id": c.parent_id or "",
                "is_parent": str(c.is_parent),
                "word_count": c.word_count,
            } for c in batch_chunks],
        )

        print(f"  Indexed batch {batch_num}/{total_batches}")

    print(f"Indexing complete. {collection.count():,} chunks in collection.")

    # Store parent chunks in a separate collection for hierarchical retrieval
    if parent_chunks:
        parent_collection_name = f"{collection_name}_parents"
        print(f"\nIndexing {len(parent_chunks):,} parent chunks into '{parent_collection_name}'")

        try:
            client.delete_collection(parent_collection_name)
        except Exception:
            pass

        parent_collection = client.create_collection(
            name=parent_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed parents too
        parent_model = load_embedding_model()
        parent_embeddings = embed_chunks(parent_chunks, parent_model)

        for i in range(0, len(parent_chunks), BATCH_SIZE):
            batch = parent_chunks[i:i + BATCH_SIZE]
            batch_emb = parent_embeddings[i:i + BATCH_SIZE]

            parent_collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=batch_emb,
                documents=[c.text for c in batch],
                metadatas=[{
                    "title": c.title,
                    "author": c.author,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "strategy": c.strategy,
                    "is_parent": str(c.is_parent),
                    "word_count": c.word_count,
                } for c in batch],
            )

        print(f"Parent collection complete. {parent_collection.count():,} parent chunks indexed.")

    return collection


# ── Verification ───────────────────────────────────────────────────────────────

def verify_index(
    collection: chromadb.Collection,
    model: HuggingFaceEmbeddings,
    test_query: str = "What is the nature of guilt and redemption?",
) -> None:
    """
    Run a test query against the index to confirm retrieval is working.
    This is the smoke test — if this works, the agent can use the index.
    """
    print(f"\nVerifying index with test query:")
    print(f"  '{test_query}'")

    query_embedding = model.embed_query(test_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print("\nTop 3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n  Result {i+1}:")
        print(f"  Book: {meta['title']} — {meta['author']}")
        print(f"  Distance: {dist:.4f}")
        print(f"  Text: {doc[:200]}...")


# ── Main ingestion pipeline ────────────────────────────────────────────────────

def build_index(
    strategy: str = "hierarchical",
    # collection_name: str = "florence_literature",
    collection_name: str = "florence_literature_dev", #use a different collection name for dev ingestion, so we don't mess with the main one while testing
) -> None:
    """
    Full ingestion pipeline. Run this once to build Florence's index.
    Steps: parse → chunk → embed → index → verify
    """
    start_time = time.time()

    print("=" * 60)
    print("Florence — Literature RAG Agent")
    print("Building corpus index")
    print(f"Strategy: {strategy}")
    print(f"Collection: {collection_name}")
    print("=" * 60)

    # Step 1 — Parse
    print("\nStep 1: Parsing corpus...")
    documents = parse_corpus(BOOKS_DIR)
    if not documents:
        print("No documents found. Check your Books/ folder.")
        return

    # Step 2 — Chunk
    print("\nStep 2: Chunking corpus...")
    parent_chunks = []

    if strategy == "hierarchical":
        all_child_chunks = []
        for doc in documents:
            parents, children = chunk_hierarchical(doc)
            parent_chunks.extend(parents)
            all_child_chunks.extend(children)
        chunks = all_child_chunks
    else:
        chunks = chunk_corpus(documents, strategy=strategy)

    print(f"Total chunks to index: {len(chunks):,}")

    # Step 3 — Embed
    print("\nStep 3: Embedding chunks...")
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)

    # Step 4 — Index
    print("\nStep 4: Indexing into ChromaDB...")
    client = get_chroma_client()
    collection = index_chunks(
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
        client=client,
        parent_chunks=parent_chunks if strategy == "hierarchical" else None,
    )

    # Step 5 — Verify
    print("\nStep 5: Verifying index...")
    verify_index(collection, model)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Index built successfully in {elapsed:.1f} seconds.")
    print(f"ChromaDB stored at: {CHROMA_DIR}")
    print(f"Collection: {collection_name}")
    print(f"Total chunks indexed: {collection.count():,}")
    print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_index(
        strategy="hierarchical",
        # collection_name="florence_literature",
        collection_name="florence_literature_dev", #use a different collection name for dev ingestion, so we don't mess with the main one while testing
    )