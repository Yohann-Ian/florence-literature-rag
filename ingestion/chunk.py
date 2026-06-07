"""
ingestion/chunk.py

Florence — Literature RAG Agent
Three chunking strategies for comparative evaluation via Ragas. I have chosen a more robust strategy
here because literature is extremely contextual. Sentences don't mean anything without the situations
they're in. Meaning is nested within meaning, from once sentence to a paragraph, to a scene, to a chapter.

Config A — Fixed size (baseline)
Config B — Semantic chunking  (!run "pip install langchain-experimental==0.3.4")
Config C — Hierarchical parent-child

Each strategy produces a list of Chunk objects with full metadata
for citation and experiment tracking.


CHUNK.PY — CODE FLOW
====================

INPUT: ParsedDocument objects from parse.py
OUTPUT: list[Chunk] objects ready for embed_index.py

FLOW:
    1. ParsedDocument arrives from parse.py
       └── contains raw text, title, author, filename

    2. strip_gutenberg_header(text)
       └── removes legal boilerplate from public domain texts
       └── called first inside every chunking strategy below

    3. THREE CHUNKING STRATEGIES (we pick one per experiment):

       Config A — chunk_fixed(doc)
       └── splits text into fixed 512-character blocks
       └── 64-character overlap between blocks
       └── fast but not sophisticated, "dumb", breaks mid-sentence
       └── returns list[Chunk]

       Config B — chunk_semantic(doc)
       └── loads HuggingFace embeddings locally (no API cost)
       └── splits where embedding similarity drops sharply
       └── respects scene and argument boundaries
       └── returns list[Chunk]

       Config C — chunk_hierarchical(doc)  ← recommended
       └── first pass: large parent chunks (1024 characters)
       └── second pass: small child chunks (256 characters)
       └── each child carries parent_id linking back to parent
       └── agent retrieves children, generates from parents
       └── returns tuple(list[Chunk], list[Chunk])

    4. chunk_document(doc, strategy)
       └── single document router
       └── receives one ParsedDocument
       └── routes to A, B, or C based on strategy argument
       └── always returns flat list[Chunk] - flat because ChromaDB expects it

    5. chunk_corpus(documents, strategy)
       └── loops over all 12 ParsedDocuments
       └── calls chunk_document() on each
       └── collects and returns one flat list[Chunk]
       └── prints statistics: total, average, min, max chunk size

OUTPUT: flat list[Chunk] → consumed by embed_index.py


"""

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# We import ParsedDocument from parse.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse import ParsedDocument


# ── Data structure ─────────────────────────────────────────────────────────────
# We create a chunk class

@dataclass
class Chunk:
    """
    A single retrievable unit of text from the corpus.
    Every chunk carries its metadata so Florence can cite her sources.
    """
    chunk_id: str                    # unique identifier
    text: str                        # the actual text content
    title: str                       # book title
    author: str                      # book author
    filename: str                    # source filename
    chunk_index: int                 # position in the document
    total_chunks: int                # total chunks in this document
    strategy: str                    # 'fixed', 'semantic', or 'hierarchical'
    parent_id: Optional[str] = None  # for hierarchical — links child to parent
    is_parent: bool = False          # True if this is a parent chunk
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


# ── Gutenberg header stripper ──────────────────────────────────────────────────

def strip_gutenberg_header(text: str) -> str:
    """
    For houskeeping, we remove Project Gutenberg licence headers and footers, since we 
    obtained the books from them. These markers are in public domain texts and pollute embeddings
    with legal boilerplate.
    """
    # Find where the actual content starts
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "END OF THE PROJECT GUTENBERG",
    ]

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

        # Try standard header strip first
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            end_of_line = text.find("\n", idx)
            if end_of_line != -1:
                text = text[end_of_line + 1:]
            break

    # Strip footer
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Secondary pass — find where actual prose begins
    # Look for "Chapter 1" or "CHAPTER I" or "Part One, Chapter 1"
    # which signals the end of front matter and start of the novel
    chapter_patterns = [
        r'\nChapter 1\n',
        r'\nChapter I\n',
        r'\nCHAPTER 1\n',
        r'\nCHAPTER I\n',
        r'\nPART ONE\nChapter 1\n',
        r'\nBook One\n',
        r'\nBOOK I\n',
        r'\nI\.\n',          # for epics like the Aeneid, for examp
        r'\nBook First\n',
    ]

    import re
    for pattern in chapter_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[match.start():]
            break

    return text.strip()

# ── Config A — Fixed size ──────────────────────────────────────────────────────

def chunk_fixed(
    doc: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Config A: Fixed-size chunking with overlap.
    Baseline strategy. Fast and predictable but breaks mid-sentence,
    mid-scene, and mid-argument in literary text.
    """
    text = strip_gutenberg_header(doc.text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < 50:
            continue
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text.strip(),
            title=doc.title,
            author=doc.author,
            filename=doc.filename,
            chunk_index=i,
            total_chunks=total,
            strategy="fixed",
        ))

    return chunks


# ── Config B — Semantic chunking ───────────────────────────────────────────────

def chunk_semantic(
    doc: ParsedDocument,
    embeddings_model: Optional[object] = None,
    breakpoint_threshold: float = 95.0,
) -> list[Chunk]:
    """
    Config B: Semantic chunking using embedding similarity.
    Splits text where the semantic meaning shifts significantly.
    Better at respecting scene and argument boundaries than fixed chunking.
    """
    text = strip_gutenberg_header(doc.text)

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    splitter = SemanticChunker(
        embeddings=embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < 50:
            continue
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text.strip(),
            title=doc.title,
            author=doc.author,
            filename=doc.filename,
            chunk_index=i,
            total_chunks=total,
            strategy="semantic",
        ))

    return chunks


# ── Config C — Hierarchical parent-child ───────────────────────────────────────

def chunk_hierarchical(
    doc: ParsedDocument,
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
    parent_overlap: int = 128,
    child_overlap: int = 32,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Config C: Hierarchical parent-child chunking.
    
    Small child chunks are retrieved for precision grading.
    Large parent chunks are returned to the LLM for generation.
    
    This directly addresses the 'lost in the middle' problem:
    retrieval is precise, generation has full context.

    Returns: (parent_chunks, child_chunks)
    The agent retrieves child chunks, then fetches their parent for the answer.

    Here, meaning is defined by the small sentence within the context of the parent.

    """
    text = strip_gutenberg_header(doc.text)

    # Create parent chunks first
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # Create child splitter for smaller retrieval units
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_parents = parent_splitter.split_text(text)
    total_parents = len(raw_parents)

    parent_chunks = []
    child_chunks = []

    for i, parent_text in enumerate(raw_parents):
        if len(parent_text.strip()) < 100:
            continue

        # Create parent chunk
        parent_id = str(uuid.uuid4())
        parent_chunk = Chunk(
            chunk_id=parent_id,
            text=parent_text.strip(),
            title=doc.title,
            author=doc.author,
            filename=doc.filename,
            chunk_index=i,
            total_chunks=total_parents,
            strategy="hierarchical",
            is_parent=True,
        )
        parent_chunks.append(parent_chunk)

        # Split parent into children
        raw_children = child_splitter.split_text(parent_text)
        for j, child_text in enumerate(raw_children):
            if len(child_text.strip()) < 30:
                continue
            child_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=child_text.strip(),
                title=doc.title,
                author=doc.author,
                filename=doc.filename,
                chunk_index=j,
                total_chunks=len(raw_children),
                strategy="hierarchical",
                parent_id=parent_id,  # links back to parent
                is_parent=False,
            )
            child_chunks.append(child_chunk)

    return parent_chunks, child_chunks


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def chunk_document(
    doc: ParsedDocument,
    strategy: str = "hierarchical",
    **kwargs,
) -> list[Chunk]:
    """
    Route a document to the correct chunking strategy.
    Returns a flat list of chunks regardless of strategy.
    For hierarchical, returns child chunks (parents stored separately).
    """
    print(f"  Chunking: {doc.title} — strategy={strategy}")

    if strategy == "fixed":
        chunks = chunk_fixed(doc, **kwargs)
        print(f"  Done: {len(chunks)} chunks")
        return chunks

    elif strategy == "semantic":
        chunks = chunk_semantic(doc, **kwargs)
        print(f"  Done: {len(chunks)} chunks")
        return chunks

    elif strategy == "hierarchical":
        parents, children = chunk_hierarchical(doc, **kwargs)
        print(f"  Done: {len(parents)} parent chunks, {len(children)} child chunks")
        # Store parents in kwargs for embed_index to access if needed
        if "parent_store" in kwargs and kwargs["parent_store"] is not None:
            kwargs["parent_store"].extend(parents)
        return children

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose fixed, semantic, or hierarchical.")


def chunk_corpus(
    documents: list[ParsedDocument],
    strategy: str = "hierarchical",
    **kwargs,
) -> list[Chunk]:
    """
    Chunk all documents in the corpus using the specified strategy.
    Returns a flat list of all chunks across all documents.
    """
    print(f"\nChunking corpus — strategy: {strategy}")
    print("=" * 60)

    all_chunks = []
    parent_store = [] if strategy == "hierarchical" else None

    for doc in documents:
        if strategy == "hierarchical":
            chunks = chunk_document(doc, strategy=strategy, parent_store=parent_store, **kwargs)
        else:
            chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(chunks)

    print("=" * 60)
    print(f"Total chunks: {len(all_chunks):,}")

    if parent_store:
        print(f"Total parent chunks: {len(parent_store):,}")

    # Print chunk size statistics
    if all_chunks:
        sizes = [c.word_count for c in all_chunks]
        avg = sum(sizes) / len(sizes)
        print(f"Average chunk size: {avg:.0f} words")
        print(f"Min chunk size: {min(sizes)} words")
        print(f"Max chunk size: {max(sizes)} words")

    return all_chunks


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test all three chunking strategies against the corpus.
    Usage: python ingestion/chunk.py
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.parse import parse_corpus

    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Books"
    )

    print("Loading corpus...")
    docs = parse_corpus(corpus_path)

    if not docs:
        print("No documents found. Check your Books/ folder.")
        sys.exit(1)

    # Test with just the first document for speed
    test_doc = docs[0]
    print(f"\nTesting all strategies on: {test_doc.title}")
    print("=" * 60)

    # Config A
    print("\nConfig A — Fixed size:")
    fixed_chunks = chunk_fixed(test_doc)
    print(f"  {len(fixed_chunks)} chunks")
    print(f"  Sample: {fixed_chunks[0].text[:200]}")

    # Config B  
    print("\nConfig B — Semantic:")
    semantic_chunks = chunk_semantic(test_doc)
    print(f"  {len(semantic_chunks)} chunks")
    print(f"  Sample: {semantic_chunks[0].text[:200]}")

    # Config C
    print("\nConfig C — Hierarchical:")
    parents, children = chunk_hierarchical(test_doc)
    print(f"  {len(parents)} parent chunks, {len(children)} child chunks")
    print(f"  Parent sample: {parents[0].text[:600]}")
    print(f"  Child sample: {children[0].text[:200]}")